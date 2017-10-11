"""
Project:    
File:       proximal_operators.py
Created by: louise
On:         10/9/17
At:         5:32 PM
"""

import torch
from torch.autograd import Variable

def proximal_linf_ball(p, r=1.0):
    """
    Proximal operator for sum(gradient(x)).
    :param p: pytorch Variable [MxNx2], 
    :param r: float, radius of infinity norm ball.
    :return: numpy array, same dimensions as p
    """
    m1 = torch.max(torch.add(p.data, - r), torch.zeros(p.size()))
    m2 = torch.max(torch.add(torch.neg(p.data), - r), torch.zeros(p.size()))

    return p - Variable(m1 - m2)


def proximal_l1(x, f, clambda):
    """

    :param x: pytorch Variable, [MxN], primal variable,
    :param f: pytorch Variable, [MxN], observed image,
    :param clambda: float, parameter for data term.
    :return: pytorch Variable, [MxN]
    """
    return x + torch.clamp(f - x, -clambda, clambda)