import math
import random
from random import random

'''
"contextlib" module provides utilities for working with context managers and "with" statements.
-"contextmanager" from "contextlib" is a decorator that manages resources.
'''
from contextlib import contextmanager

'''
"kornia" is a python package for Computer Vision.
-"filter2D" from "kornia" applies a 2D kernel to a tensor.
'''
from kornia.filters import filter2D

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

# Helper Classes and Functions

# calculates exponential moving averages
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

# residual connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

# channel normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# flattens a tensor
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

# applies a function at random
class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

# applies a 2nd order binominal filter for bilinear sampling
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

# checks if the given item exists
def exists(val):
    return val is not None

# null context
@contextmanager
def null_context():
    yield

# returns default value if the given value does not exist
def default(value, d):
    return value if exists(value) else d

# returns items from iterable
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# casts to list
def cast_list(el):
    return el if isinstance(el, list) else [el]

# checks if tensor is empty
def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

# raises Nan exception
def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

# accumulates contexts periodically
def gradient_accumulate_contexts(gradient_accumulate_every):
    contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

# loss is propagated backwards
def loss_backwards(loss, **kwargs):
    loss.backward(**kwargs)

'''
Gradient Penalty
-Applies gradient penalty to ensure stability in GAN training by preventing exploding gradients in the discriminator.
-Read about it at https://arxiv.org/pdf/1704.00028.pdf
-Watch about it at https://www.youtube.com/watch?v=5c57gnaPkA4
'''
def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# calculates perceptual path length to achieve feature disentanglement by determining the difference between successive images when interpolating between two noise inputs
def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]

    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

# returns random noise [N, latent_dim]
def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).to(device)

# returns a list of noise and layers [(N, latent_dim), num_layers]
def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

# returns a list of mixed noise generated at random and layers like the previous function
def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

# returns the style vector after passing latents through the Style Vectorizer or Mapping Network as referred to in the research paper
def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

# returns uniformly distributed noise [N, H, W, C]
def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)

'''
Leaky ReLU
-Leaky ReLU is an activation function that fixes the "dyling ReLU" problem - max(0.1x, x)
-Read about it at https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24#0863
-Watch a video about it at https://www.youtube.com/watch?v=Y-ruNSdpZ0Q
'''
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

# model evalutes the batches in chunks
def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

# concatenates all styles [(N, latent_dim), num_layers] --> [N, num_layers, latent_dim]
def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

'''
Spherical Linear Interpolation (SLERP)
-Spherical Linear Interpolation is a type of interpolation between two points on an arc.
-Read more about at https://en.wikipedia.org/wiki/Slerp
-Watch a video about interpolations at https://www.youtube.com/watch?v=ibkT5ao8kGY
'''
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


# Losses

'''
Hinge Loss
-Hinge Loss is usually used to train classifiers especially Support Vector Machines (SVMs).
-Read about at https://towardsdatascience.com/a-definitive-explanation-to-hinge-loss-for-support-vector-machines-ab6d8d3178f1
-Watch a video about it at https://www.youtube.com/watch?v=RBtgpKmdBlk
'''

# Hinge loss for generator
def gen_hinge_loss(fake, real):
    return fake.mean()

# Hinge loss for discriminator
def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()
