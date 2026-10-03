import numpy as np

def conv_out_size(n, k, stride=1, padding=0, dilation=1):
    raise NotImplementedError

def dilated_conv2d(x, w, dilation):
    raise NotImplementedError

def strided_conv2d(x, w, stride):
    raise NotImplementedError

def conv_transpose2d(y, w, stride, out_shape):
    raise NotImplementedError
