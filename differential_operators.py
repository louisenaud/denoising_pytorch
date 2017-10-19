"""
Project:    
File:       differential_operators.py
Created by: louise
On:         10/9/17
At:         4:19 PM
"""
import torch
from torch.autograd import Variable


def forward_gradient(im, dtype=torch.FloatTensor):
    """
    Function to compute the forward gradient of the image I.
    Definition from: http://www.ipol.im/pub/art/2014/103/, p208
    :param im: torch variable [1xMxN], input image
    :param dtype: torch type, set to FloatTensor by default
    :return: torch variable [2xMxN], gradient of the input image, the first channel is the horizontal gradient, the second 
    is the vertical gradient. 
    """
    im_size = im.size()
    gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype))  # Allocate gradient array
    # Horizontal direction
    gradient[0, :, :-1] = im[0, :, 1:] - im[0, :, :-1]
    # Vertical direction
    gradient[1, :-1, :] = im[0, 1:, :] - im[0, :-1, :]

    return gradient


def forward_weighted_gradient(im, w, dtype=torch.FloatTensor):
    """
    Function to compute the forward gradient of the image I.
    Definition from: http://www.ipol.im/pub/art/2014/103/, p208
    :param im: torch tensor [MxN], input image
    :param w: pytorch variable [2xMxN], weights array
    :param dtype: torch type, set to FloatTensor by default
    :return: torch tensor [2xMxN], gradient of the input image, the first channel is the horizontal gradient, the second 
    is the vertical gradient. 
    """
    im_size = im.size()
    gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype))  # Allocate gradient array
    # Horizontal direction
    gradient[0, :, :-1] = im[0, :, 1:] - im[0, :, :-1]
    # Vertical direction
    gradient[1, :-1, :] = im[0, 1:, :] - im[0, :-1, :]
    gradient = gradient * Variable(w)
    return gradient


def backward_divergence(grad, dtype=torch.FloatTensor):
    """
    Function to compute the backward divergence.
    Definition in : http://www.ipol.im/pub/art/2014/103/, p208

    :param grad: numpy array [NxMx2], array with the same dimensions as the gradient of the image to denoise.
    :param dtype: torch type, set to FloatTensor by default
    :return: numpy array [NxM], backward divergence 
    """
    im_size = grad.size()
    div = torch.zeros((1, im_size[0], im_size[1])).type(dtype)  # Allocate divergence array
    # Horizontal direction
    d_h = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
    d_h[0, :, 0] = grad[0, :, 0]
    d_h[0, :, 1:-1] = grad[0, :, 1:-1] - grad[0, :, :-2]
    d_h[0, :, -1] = -grad[0, :, -2:-1]

    # Vertical direction
    d_v = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
    d_v[0, 0, :] = grad[1, 0, :]
    d_v[0, 1:-1, :] = grad[1, 1:-1, :] - grad[1, :-2, :]
    d_v[0, -1, :] = -grad[1, -2:-1, :]

    # Divergence
    div = d_h + d_v
    return div
