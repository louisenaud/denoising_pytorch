"""
Project:    
File:       differential_operators_test.py
Created by: louise
On:         10/9/17
At:         5:01 PM
"""

import unittest
import numpy as np
import torch
from torch.autograd import Variable
from differential_operators import forward_gradient, backward_divergence


class TestAdjoint(unittest.TestCase):

    def test_adjoint_operator(self):
        Y = 200
        X = 100
        x = 1 + Variable(torch.randn((1, Y, X)).type(torch.FloatTensor))
        y_l = Variable(torch.randn((2, Y + 1, X + 1)).type(torch.FloatTensor))

        y_l[0, 1:, 1:-1] = 1 + torch.randn((1, Y, X - 1))
        y_l[1, 1:-1, 1:] = 1 + torch.randn((1, Y - 1, X))
        y = y_l[:, 1:, 1:]
        # Compute gradient and divergence
        gx = forward_gradient(x)
        dy = backward_divergence(y)

        check = abs((y.data.numpy()[:] * gx.data.numpy()[:]).sum() + (dy.data.numpy()[:]*x.data.numpy()[:]).sum())
        print(check)
        self.assertTrue(check < 1e-4)