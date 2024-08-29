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

# Results

|                 | Accuracy | Epoch |
|:---------------:|:--------:|:-----:|
| ViT small       |  88.26%  |  400  |
| Reversiable ViT |  86.99%  |  400  |
| BDIA Vit        |  88.99%  |  400  |

# BDIA

### Basic Concept of BDIA:
BDIA is an approach initially proposed for solving ordinary differential equations (ODEs) by integrating bidirectionally. This technique was later adapted for use in neural networks, specifically for improving the reversibility of Transformer models.

#### BDIA as ODE Solver
In the context of ODEs, BDIA provides a way to approximate the next state in a diffusion process by considering both forward and backward integration approximations. This dual-directional approach helps to maintain accuracy and stability when simulating time-reversible processes, such as diffusion.

#### Application in Neural networks
BDIA can be applied to transformer blocks by treating each block as a step in the Euler integration of an ODE. The technique uses a parameter 𝛾, which can take values of 0.5 or -0.5, to average two consecutive integration approximations. This averaging acts as a regularization mechanism, which can improve the stability and performance of the model during training.

### Implementation in Transformers:
The BDIA approach was specifically adapted to Vision Transformers (ViTs) and other Transformer architectures to achieve bit-level exact reversibility. Here's how it is implemented:

#### Transformer Blocks as ODE Approximations
Each transformer block is treated as an approximation step in solving an ODE. The BDIA method then incorporates this into the neural network's architecture by averaging the results from forward and backward integrations.

#### Activation quantization
To ensure bit-level reversibility, activation quantization is applied. This means that each transformation within the model can be exactly reversed, which is critical for enabling backpropagation without storing all intermediate activations. This approach is particularly useful in reducing memory consumption during training.

#### Light-weight Side information
Since activation quantization can lead to small information loss, some light-weight side information is stored during the forward pass. This stored information is used to perfectly reconstruct the intermediate states during the backward pass, ensuring that the training process is stable and precise.

# Cite

Special Thank you to:

On Exact Bit-level Reversible Transformers Without Changing Architectures [arxiv](https://arxiv.org/abs/2407.09093)

Exact Diffusion Inversion via Bi-directional Integration Approximation [arxiv](https://arxiv.org/abs/2307.10829)

Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)