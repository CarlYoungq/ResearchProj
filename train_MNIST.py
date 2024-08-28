# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import csv
import time
from models import *
from utils import progress_bar
from models.randomaug import RandAugment
from models.convmixer import ConvMixer
from models.SETAdam import SETAdam
from models.VSAdamMin import VSAdamMin
import logging
from decimal import Decimal
from tqdm import tqdm
from datetime import datetime


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# parsers
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# NET--------------------------------------------------------------------------
parser.add_argument('--net', default='vit_small_reverse')
# NET--------------------------------------------------------------------------
# EPOCHS-----------------------------------------------------------------------
parser.add_argument('--n_epochs', type=int, default='50')
# EPOCHS-----------------------------------------------------------------------
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') #Vit..1e-4
parser.add_argument('--opt', default="setadam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--bs', default='128')
parser.add_argument('--size', default="28")
parser.add_argument('--run', default=0, type=int)
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
args = parser.parse_args()


eps_str='%.1E' % Decimal(str(args.eps))


# log the traning
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

outputs_dir = "outputs/MNIST"
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(outputs_dir, f"log_{time_now}_{args.net}_{args.n_epochs}.txt")
f_handler = logging.FileHandler(log_file_path)
f_handler.setLevel(logging.INFO)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.info('Logging is set up.')
logger.info(f'{args.net}_optim_{args.opt}_epoch_{args.n_epochs}_lr_{args.lr}_eps_{eps_str}')
  

bs = int(args.bs)
imsize = int(args.size)
use_amp = False
aug = args.noaug


device = 'cuda' if torch.cuda.is_available() else 'cpu'


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
max_memory_usage = 0  # max memory usage
total_memory_usage = 0  # total memory usage


# Data
logger.info('==> Preparing data..')
print('==> Preparing data..')
size = imsize
transform_train = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  
])

transform_test = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# Prepare dataset
trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
classes = tuple(str(i) for i in range(10))  


# Model factory..
logger.info('==> Building model..')
print('==> Building model..')
if args.net=="vit_small":
    from models.vit_small import ViT

    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 8,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels = 1
        )
elif args.net=="vit_small_bdia":
    from models.vit_small_inv_BDIA import ViT

    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 8,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels = 1
        )
elif args.net=='vit_small_reverse':
    from models.vit_small_reverse import ViT
    
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=10,
        dim=int(args.dimhead),
        depth=4,
        heads=8,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        channels = 1
        )
elif args.net=='vit_small_momentum':
    from models.vit_small_momentum import ViT
    
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=10,
        dim=int(args.dimhead),
        depth=4,
        heads=8,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        channels = 1
        )
logger.info(f"Model name: {args.net}")
logger.info(f"Total epochs: {args.n_epochs}")
print(f"Model name: {args.net}")
print(f"Total epochs: {args.n_epochs}")


# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    logger.info("Using CUDA and data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Criterion & Optimizer
criterion = nn.CrossEntropyLoss()
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps)    
elif args.opt == "vsadammin":
    optimizer = VSAdamMin(net.parameters(), lr=args.lr, eps=args.eps) 
elif args.opt == "setadam":
    optimizer = SETAdam(net.parameters(), lr=args.lr, eps=args.eps)        
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  

    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: \nmean={param.grad.abs().mean().item()}, \nmax={param.grad.abs().max().item()}, \nmin={param.grad.abs().min().item()}\n")
        else:
            print(f"{name}: No gradient")


##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    global max_memory_usage, total_memory_usage
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_memory_usage = 0  # epoch memory usage
    torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    with tqdm(total=len(trainloader), desc=f"Epoch {epoch}") as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
            loss_sum = loss # -0.001*KL_loss_sum
            scaler.scale(loss_sum).backward()
    
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            '''
            current_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # convert to MiB
            epoch_memory_usage = max(epoch_memory_usage, current_memory_usage)
            '''
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            pbar.update(1)
            pbar.set_postfix({
                'Loss': train_loss / (batch_idx + 1), 
                'Accuracy': 100. * correct / total
            })
    epoch_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)
    max_memory_usage = max(max_memory_usage, epoch_memory_usage)
    total_memory_usage += epoch_memory_usage

    print("tra_loss: ", train_loss/(batch_idx+1), "train_acc: ", 100.*correct/total), 
    return train_loss/(batch_idx+1), 100.*correct/total, epoch_memory_usage


##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(testloader), desc=f"Epoch {epoch} [Test]") as pbar:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
                pbar.update(1)
                pbar.set_postfix({
                    'Loss': test_loss / (batch_idx + 1), 
                    'Accuracy': 100. * correct / total
                })
    print("val_loss: ", test_loss/(batch_idx+1), "val_acc: ", 100.*correct/total) 
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return test_loss/(batch_idx+1), acc


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
epochs = []
memory_usages = []
   
net.cuda()
'''
from ptflops import get_model_complexity_info
from torchsummary import summary
net.train()

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(net, (1, 28, 28), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    #print(f"Model GFLOPs: {macs}, Parameters: {params}")

#summary(net, (1, 28, 28))
'''

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    train_loss, train_acc, train_memory_usage = train(epoch)
    val_loss, val_acc = test(epoch)
    
    scheduler.step(epoch-1)
    
    epochs.append(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    memory_usages.append(train_memory_usage)

    logger.info("epoch: %d train_loss: %.10f train_acc: %.4f val_loss:  %.10f val_acc: %.4f memory_usage: %.2f MiB",
                epoch, train_loss, train_acc, val_loss, val_acc, train_memory_usage)

logger.info(f"Best accuracy: {best_acc:.4f}")
logger.info(f"Max memory usage: {max_memory_usage:.2f} MB")
logger.info(f"Total memory usage: {total_memory_usage:.2f} MiB")



'''
outputs_npy = os.path.join(outputs_dir, f'results_{time_now}_{args.net}_{args.n_epochs}.npy')   
np.save(outputs_npy, {
    'epochs': epochs,
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'memory_usage (MiB)': memory_usages
    })
'''
results_df = pd.DataFrame({
    'epoch': epochs,
    'train_loss': train_losses,
    'train_acc': train_accuracies,
    'val_loss': val_losses,
    'val_acc': val_accuracies,
    'memory_usage (MiB)': memory_usages
    })
results_csv = os.path.join(outputs_dir, f'results_{time_now}_{args.net}_{args.n_epochs}.csv')
results_df.to_csv(results_csv, index=False)

with open(results_csv, 'a') as f:
    f.write(f'\nBest accuracy: {best_acc:.4f}\n')
    f.write(f'Max memory usage: {max_memory_usage:.2f} MiB\n')
    f.write(f'Total memory usage: {total_memory_usage:.2f} MiB\n')


