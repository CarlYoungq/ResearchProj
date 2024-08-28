from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch.distributions import Categorical

from torch.autograd import Function as Function

import numpy as np

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        
        return self.net(x)
    

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)
        
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    

class ReversibleTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.block=nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ])  

    def forward(self, x):
        attn = self.block[0]
        ff = self.block[1]
        y = attn(x) + x
        z = ff(y) + y
        
        return z, x

    def backward_pass(self, z, x_pre, dz):
        attn = self.block[0]
        ff = self.block[1]

        with torch.enable_grad():
            x_pre.requires_grad_(True)
            attn_out = attn(x_pre)
            y = x_pre + attn_out
            
            y.requires_grad_(True)
            ff_out = ff(y)
            tmp = y + ff_out
            #tmp.backward(dz, retain_graph=True)
            try:
                tmp.backward(dz, retain_graph=True)
            except RuntimeError as e:
                print(f"Error during backward: {e}")
                return x_pre, dz        
            
        with torch.no_grad():
            if x_pre.grad is None:
                print("Warning: x_pre.grad is None, using dz as dx")
            dx = dz + x_pre.grad if x_pre.grad is not None else dz
            x_pre.grad = None

        return x_pre, dx


class RevBackProp(Function):
    @staticmethod
    def forward(ctx, x, layers):
        last_tensors = []
        last_tensors.append(x.detach())
        x_pre = None
        
        for layer in layers:
            x, x_pre = layer(x)
            last_tensors.append(x.detach())
        ctx.save_for_backward(*last_tensors)
        ctx.layers = layers
        
        return x

    @staticmethod
    def backward(ctx, dx):
        x = ctx.saved_tensors[-1]
        x_pre = ctx.saved_tensors[-2]
        layers = ctx.layers

        for i, layer in enumerate(layers[::-1]):
            x, dx = layer.backward_pass(x, x_pre, dx)
            if i < len(layers) - 1:
                x_pre = ctx.saved_tensors[-1 - i - 2]

        return dx, None
    

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        
        return self.to_patch_tokens(x_with_shifts)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.no_custom_backward = False

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([ReversibleTransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def vanilla_backward(x, layers):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """
        x_pre = None
        
        for layer in layers:
            x, x_pre = layer(x)
            
        return x        
            
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if not self.training or self.no_custom_backward:
            executing_fn = ViT.vanilla_backward
        else:
            executing_fn = RevBackProp.apply

        x = executing_fn(x, self.layers)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x)
