import torch
from torch import nn


def Conv2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    """Conv2d"""
    if 'padding' in kwargs:
        kwargs.pop('padding')
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, *args, **kwargs)


def ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    """ConvTranspose2d"""
    if 'padding' in kwargs:
        kwargs.pop('padding')
    if 'output_padding' in kwargs:
        kwargs.pop('output_padding')
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2, output_padding=stride-1, *args, **kwargs)
