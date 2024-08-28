# vision-transformers

Let's train vision transformers for comparision! 

I use pytorch for implementation.

Python version: 3.8

CUDA version: 12.3.107

Torch version: 2.2.0

GPU: NVIDIA RTX 4060 laptop GPU (8GB RAM)

## THIS IS A MODIFICATED PROJECT FOR FINAL REPORT WRITTEN BY HAOLUN YANG 

# Requirements

SEE requirements.txt

# Usage example

Train on CIFAR-10 using `train_cifar10.py`

Train on CIFAR-10 using `train_MNIST.py` 

CHOOSE YOUR NET by editing the argument `--net`

`parser.add_argument('--net', default='vit_small')` -------------FOR ViT-small
`parser.add_argument('--net', default='vit_small_reverse')` -----FOR Reversiable ViT-small
`parser.add_argument('--net', default='vit_small_bdia')` --------FOR BDIA ViT-small

UNFINISHED MODIFICATION MODEL `vit_small_momentum`

# Results..

|                 | Accuracy | Epoch |
|:---------------:|:--------:|:-----:|
| ViT small       |  88.26%  |  400  |
| Reversiable ViT |  86.99%  |  400  |
| BDIA Vit        |  88.99%  |  400  |

# Cite

Special Thank you to:

Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)