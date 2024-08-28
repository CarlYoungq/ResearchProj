# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:40:22 2024

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


'''
file_paths = [
    '../outputs/CIFAR10/results_20240730_124850_vit_small_bdia_400.csv',
    '../outputs/CIFAR10/results_20240801_093648_vit_small_400.csv',
    '../outputs/CIFAR10/results_20240731_233654_vit_small_reverse_400.csv'
]
offsets = [
    (0, 70), 
    (0, -100),   
    (0, 70)  
]
'''
file_paths = [
    '../outputs/MNIST/results_20240809_154128_vit_small_bdia_50.csv',
    '../outputs/MNIST/results_20240809_183439_vit_small_50.csv',
    '../outputs/MNIST/results_20240809_213700_vit_small_reverse_50.csv'
]
offsets = [
    (0, -70), 
    (0, -70),   
    (0, 70)  
]


plt.figure(figsize=(12,8))

for idx, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    model_name = df.iloc[-1, 0]
    max_memory_str = df.iloc[-3, 0]
    memory_usage = float(max_memory_str.split(':')[-1].strip().split()[0])
    epochs = df.iloc[-6, 0]
    epochs = int(epochs)+1
    df = df.iloc[:-5] 

    
    df['epoch'] = df['epoch'].astype(float)
    df['memory_usage (MiB)'] = df['memory_usage (MiB)'].astype(float)

    plt.plot(df['epoch']+1, df['memory_usage (MiB)'], label=f'{model_name} Memory Consumption ', linewidth=5)
    
    annotation_epoch = df['epoch'].iloc[int(epochs/2)]
    annotation_memory_value = df['memory_usage (MiB)'].iloc[int(epochs/2)]
    
    xytext_offset = offsets[idx]

    plt.annotate(
        f'Memory Usage: {memory_usage:.2f} MiB',
        xy=(annotation_epoch, annotation_memory_value),
        xytext=(annotation_epoch, annotation_memory_value + xytext_offset[1]),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5)
    )

#plt.title('CIFAR-10 Memory Usage Curves for Models', fontsize=24)
plt.title('MNIST Memory Usage Curves for Models', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Memory Usage (MiB)', fontsize=20)
plt.legend(loc='lower left', fontsize=12)
plt.grid(True)

#output_path = '../plot/CIFAR10/memory_usage_curves.png'
output_path = '../plot/MNIST/memory_usage_curves.png'
plt.savefig(output_path)

plt.show()