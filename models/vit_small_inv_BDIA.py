from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch.distributions import Categorical

import numpy as np

from torch.autograd import Function as Function

from scipy.io import savemat, loadmat 


class quantization_F(Function):
    @staticmethod
    def forward(ctx, input, fl):
        #v_max = (pow(2, bw-1)-1)*pow(2, -fl)
        #v_min = -1*pow(2, bw-1)*pow(2, -fl)
        #max_out_of_range = (input>v_max).float()
        #min_out_of_range = (input<v_min).float()
        #input *= 1-max_out_of_range
        #input *= 1-min_out_of_range
        #input += max_out_of_range*v_max
        #input += min_out_of_range*v_min
        input = torch.round(input/pow(2, -fl))
        return input*pow(2, -fl)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None   

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        #self.gamma = nn.Parameter(torch.tensor(0.0))
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

        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_v = nn.Linear(dim, inner_dim, bias = False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #qk = self.to_qk(x).chunk(2, dim = -1)
        q = self.to_q(x)
        k = self.to_k(x)

        
        v = self.to_v(x)

        #q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        
        dots = (torch.matmul(q, k.transpose(-1, -2)) ) * self.temperature.exp()
        
        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0., quant_l = 7):
        super().__init__()

        self.block=nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ])       
        self.quant_l = quant_l


    def forward(self, x, x_pre, numseedDropout, numseedGamma):
        attn=self.block[0]
        ff=self.block[1]

        if x_pre is None:
            if self.training:
                torch.manual_seed(numseedDropout)
                y = attn(x)
                z = x + quantization_F.apply(ff(x+y)+y, self.quant_l)
            else:
                y = attn(x)
                z = x + quantization_F.apply(ff(x+y)+y, self.quant_l)
            side = None                
        else:

            if self.training:               
                torch.manual_seed(numseedGamma)
                gammaVec=(torch.randint(2, tuple([x.shape[0]])).to('cuda')-0.5)
                gammaVec=gammaVec[:,None,None].expand([x.shape[0],1,1])

                side = x_pre/pow(2, -self.quant_l) % 2 == 1
                torch.manual_seed(numseedDropout)                
                y  = attn(x)
                
                z = quantization_F.apply(gammaVec*(x_pre+side*pow(2, -self.quant_l)), self.quant_l) + quantization_F.apply(((1.0-gammaVec)*x + (1.0+gammaVec)*(ff(y+x)+y)).float(), self.quant_l)

            else:
                y  = attn(x)
                z = quantization_F.apply(x+(ff(y+x)+y), self.quant_l)

                side = None

        del y 

        return z, x, side

    def backward_pass(
        self,
        flag,
        z,
        x,
        x_pre_gt,
        dz,
        dx,
        numseedDropout,
        numseedGamma,
        side,
    ):
        """
        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        attn=self.block[0]
        ff=self.block[1]


        torch.manual_seed(numseedGamma)
        gammaVec=(torch.randint(2, tuple([x.shape[0]])).to('cuda')-0.5)
        gammaVec=gammaVec[:,None,None].expand([x.shape[0],1,1])

        with torch.enable_grad():
            x.requires_grad = True
            

            if flag==False:
                torch.manual_seed(numseedDropout)
                
                y=attn(x)
                y = quantization_F.apply(((1.0-gammaVec)*x + (1.0+gammaVec)*(ff(y+x)+y)).float(), self.quant_l)

            else:
                torch.manual_seed(numseedDropout)
                y=attn(x)
                y = quantization_F.apply(((ff(y+x)+y)).float(), self.quant_l)
            

        with torch.no_grad():  
            if flag==False:      
                x_pre =(z-y)/gammaVec - side*pow(2, -self.quant_l) 
                if torch.mean(torch.pow(x_pre-x_pre_gt,2)) >0:
                    print("error ", torch.mean(torch.pow(x_pre-x_pre_gt,2)))
                    
        if flag==False:
            with torch.enable_grad():  
                x_pre.requires_grad = True            

                tmp = y+ quantization_F.apply(gammaVec*(x_pre+side*pow(2, -self.quant_l)), self.quant_l)
                
                tmp.backward(dz, retain_graph=True)

            with torch.no_grad():    
                dx = dx + x.grad
                dz = x_pre.grad 
                del y, z, tmp

                x.grad = None
                x = x.detach()
                x_pre.grad = None
                x_pre = x_pre.detach()

            return x, x_pre, dx, dz

        else:
            with torch.enable_grad():  

                tmp = y+x                

                tmp.backward(dz, retain_graph=True)

            with torch.no_grad():    
                dx = dx + x.grad
                del y, z, tmp, dz

                x.grad = None
                x = x.detach()


            return x, None, dx, None 
                                            

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

        self.quant_l = 9

        #self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads = heads, dim_head = dim_head, mlp_dim=mlp_dim, dropout = dropout, quant_l = self.quant_l))
            #self.layers.append(TransformerBlock(dim, heads = heads, dim_head = dim_head, mlp_dim=mlp_dim, dropout = 0))


        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    @staticmethod
    def vanilla_backward(x, layers):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """
        x_pre = None
        
        for block in layers:
            numseedGamma = np.random.randint(500)
            numseedDrop = np.random.randint(500)
            x, x_pre, side = block(x, x_pre, numseedGamma, numseedDrop)
        return x

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)


        # no need for custom backprop in eval/inference phase
        if not self.training or self.no_custom_backward:
            executing_fn = ViT.vanilla_backward
        else:
            executing_fn = RevBackProp.apply

        # This takes care of switching between vanilla backprop and rev backprop
        x = quantization_F.apply(x, self.quant_l)

        x = executing_fn(
            x,
            self.layers,
        )


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class RevBackProp(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        x,
        layers,
    ):
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        """

        #numseed = np.random.randint(500)
        #all_tensors = []
        seedDropArray = []
        seedGammaArray = []
        sideArray = []

        last_tensors = []
        last_tensors.append(x.detach())
        
        x_pre = None
        for block in layers:
            #print("forward numseed ", numseed)
            numseedDropout = np.random.randint(500)
            numseedGamma = np.random.randint(500)
            seedDropArray.append(numseedDropout)  
            seedGammaArray.append(numseedGamma) 
            x, x_pre, side = block(x, x_pre, numseedDropout=numseedDropout, numseedGamma=numseedGamma)
            last_tensors.append(x.detach())
            if side is not None:
                sideArray.append(side.detach())

            
        # saving only the final activations of the last reversible block
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*last_tensors)
        ctx.layers = layers
        ctx.seedDropArray = seedDropArray
        ctx.seedGammaArray = seedGammaArray
        
        ctx.sideArray = sideArray

        return x

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        #dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)
        #print(dx.shape)
        #input()
        # retrieve the last saved activations, to start rev recomputation
        #x, x_pre, x_pre_2nd, numseed = ctx.saved_tensors
        
        x = ctx.saved_tensors[-1]
        x_pre = ctx.saved_tensors[-2]
        
        seedDropArray = ctx.seedDropArray
        seedGammaArray = ctx.seedGammaArray 
        sideArray = ctx.sideArray

        # layer weights
        layers = ctx.layers

        dx_pre = torch.zeros_like(dx)
        for i, layer in enumerate(layers[::-1]):

            if i == len(layers)-1:
                x_pre_gt = None
                side = None
            else:
                x_pre_gt = ctx.saved_tensors[-1-i-2]
                side = sideArray[-1-i]

            
            x, x_pre, dx, dx_pre = layer.backward_pass(
                flag = (i==len(layers)-1),
                z = x,
                x= x_pre, 
                x_pre_gt = x_pre_gt,
                dz=dx,
                dx=dx_pre,
                numseedDropout = seedDropArray[-1-i],
                numseedGamma = seedGammaArray[-1-i],
                side = side,
            )


        del x, x_pre, dx_pre
        return dx, None, None