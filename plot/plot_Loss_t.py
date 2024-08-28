# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:40:22 2024

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''
file_paths = [
    '../outputs/CIFAR10/results_20240730_124850_vit_small_bdia_400.csv',
    '../outputs/CIFAR10/results_20240801_093648_vit_small_400.csv',
    '../outputs/CIFAR10/results_20240731_233654_vit_small_reverse_400.csv'
]
'''
file_paths = [
    '../outputs/MNIST/results_20240809_154128_vit_small_bdia_50.csv',
    '../outputs/MNIST/results_20240809_183439_vit_small_50.csv',
    '../outputs/MNIST/results_20240809_213700_vit_small_reverse_50.csv'
]

plt.figure(figsize=(12,8))

for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    model_name = df.iloc[-1, 0]
    df = df.iloc[:-5]  

    df['epoch'] = df['epoch'].astype(float)
    df['train_loss'] = df['train_loss'].astype(float)

    plt.plot(df['epoch'] + 1, df['train_loss'], label=f'{model_name} Train Loss', linewidth=5)

plt.ylim(0, 1.5) 
plt.yticks(np.arange(0, 1.6, 0.2))

#plt.title('CIFAR10 Train Loss Curves for Models', fontsize=24)
plt.title('MNIST Train Loss Curves for Models', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Train Loss', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
plt.grid(True)

#output_path = '../plot/CIFAR10/train_loss_curves.png'
output_path = '../plot/MNIST/train_loss_curves.png'
plt.savefig(output_path)

plt.show()