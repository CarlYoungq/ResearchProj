# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:40:22 2024

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np


'''
file_paths = [
    '../outputs/CIFAR10/results_20240730_124850_vit_small_bdia_400.csv',
    '../outputs/CIFAR10/results_20240801_093648_vit_small_400.csv',
    '../outputs/CIFAR10/results_20240731_233654_vit_small_reverse_400.csv'
]
offsets = [
    (-100, 0.3), 
    (-4, -3),   
    (-8, -6)  
]
'''
file_paths = [
    '../outputs/MNIST/results_20240809_154128_vit_small_bdia_50.csv',
    '../outputs/MNIST/results_20240809_183439_vit_small_50.csv',
    '../outputs/MNIST/results_20240809_213700_vit_small_reverse_50.csv'
]
offsets = [
    (-3, -2), 
    (-3, 1),
    (-15, -3)  
]



plt.figure(figsize=(12,8))

for idx, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    model_name = df.iloc[-1, 0]
    best_accuracy_str = df.iloc[-4, 0]
    best_accuracy = float(best_accuracy_str.split(':')[-1].strip())
    df = df.iloc[:-5] 

    df['epoch'] = df['epoch'].astype(float)
    df['val_acc'] = df['val_acc'].astype(float)

    plt.plot(df['epoch'] + 1, df['val_acc'], label=f'{model_name} Validation Accuracy', linewidth=5)

    best_epoch_idx = df['val_acc'].idxmax()
    best_epoch = df['epoch'].iloc[best_epoch_idx] + 1
    best_val_acc = df['val_acc'].iloc[best_epoch_idx]

    xytext_offset = offsets[idx]

    plt.annotate(
        f'Best Acc: {best_accuracy:.2f}%',
        xy=(best_epoch, best_val_acc),
        xytext=(best_epoch + xytext_offset[0], best_val_acc + xytext_offset[1]),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5)
    )

'''
plt.ylim(70, 90)
plt.yticks(np.arange(70, 90, 5))
plt.title('CIFAR-10 Validation Accuracy Curves for Models', fontsize=24)
'''
plt.ylim(90, 100)
plt.yticks(np.arange(90, 100, 5))
plt.title('MNIST Validation Accuracy Curves for Models', fontsize=24)

plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Validation Accuracy (%)', fontsize=20)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)

#output_path = '../plot/CIFAR10/validation_accuracy_curves.png'
output_path = '../plot/MNIST/validation_accuracy_curves.png'
plt.savefig(output_path)

plt.show()
