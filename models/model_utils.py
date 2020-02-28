"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR 2018
"""

import torch
import torch.nn as nn

from torch.nn import Parameter


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

class haar_psi(psi):
    def forward(self, inpt):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = inpt.shape[0], inpt.shape[1], inpt.shape[2] // bl, inpt.shape[3] // bl
        four = inpt.view(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, bl_sq, d, new_h, new_w)
        haar = torch.zeros(four.shape)
        haar[:,0] = four[:,0] + four[:,1] + four[:,2] + four[:,3]
        haar[:,1] = four[:,0] + four[:,1] - four[:,2] - four[:,3]
        haar[:,2] = four[:,0] - four[:,1] - four[:,2] + four[:,3]
        haar[:,3] = four[:,0] - four[:,1] + four[:,2] - four[:,3]
        return haar.reshape(bs, d * bl_sq, new_h, new_w)

    def inverse(self, outp):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d_height, new_height, new_width = outp.size()
        d = d_height // bl_sq
        thing = outp.view(bs, bl_sq, d, new_height, new_width)
        four = torch.zeros(thing.shape)
        four[:,0] = (thing[:,0] + thing[:,1] + thing[:,2] + thing[:,3])/4
        four[:,1] = (thing[:,0] + thing[:,1] - thing[:,2] - thing[:,3])/4
        four[:,2] = (thing[:,0] - thing[:,1] - thing[:,2] + thing[:,3])/4
        four[:,3] = (thing[:,0] - thing[:,1] + thing[:,2] - thing[:,3])/4
        return four.view(bs, bl, bl, d, new_height, new_width).permute(0, 3, 4, 1, 5, 2).reshape(bs, d, new_height*bl, new_width*bl)

class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)
