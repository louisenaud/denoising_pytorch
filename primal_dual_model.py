"""
Project:    
File:       primal_dual_model.py
Created by: louise
On:         10/11/17
At:         3:02 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardGradient(nn.Module):
    def __init__(self):
        super(ForwardGradient, self).__init__()

    def forward(self, x, dtype=torch.FloatTensor):
        im_size = x.size()
        gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype))  # Allocate gradient array
        # Horizontal direction
        gradient[0, :, :-1] = x[0, :, 1:] - x[0, :, :-1]
        # Vertical direction
        gradient[1, :-1, :] = x[0, 1:, :] - x[0, :-1, :]
        return gradient


class ForwardWeightedGradient(nn.Module):
    def __init__(self):
        super(ForwardWeightedGradient, self).__init__()

    def forward(self, x, w, dtype=torch.FloatTensor):
        im_size = x.size()
        gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype)).cuda()  # Allocate gradient array
        # Horizontal direction
        gradient[0, :, :-1] = x[0, :, 1:] - x[0, :, :-1]
        # Vertical direction
        gradient[1, :-1, :] = x[0, 1:, :] - x[0, :-1, :]
        gradient = gradient * w
        return gradient


class BackwardDivergence(nn.Module):
    def __init__(self):
        super(BackwardDivergence, self).__init__()

    def forward(self, y, dtype=torch.FloatTensor):
        im_size = y.size()
        # Horizontal direction
        d_h = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_h[0, :, 0] = y[0, :, 0]
        d_h[0, :, 1:-1] = y[0, :, 1:-1] - y[0, :, :-2]
        d_h[0, :, -1] = -y[0, :, -2:-1]

        # Vertical direction
        d_v = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_v[0, 0, :] = y[1, 0, :]
        d_v[0, 1:-1, :] = y[1, 1:-1, :] - y[1, :-2, :]
        d_v[0, -1, :] = -y[1, -2:-1, :]

        # Divergence
        div = d_h + d_v
        return div


class ProximalLinfBall(nn.Module):
    def __init__(self):
        super(ProximalLinfBall, self).__init__()

    def forward(self, p, r):
        if p.is_cuda:
            m1 = torch.max(torch.add(p.data, - r), torch.zeros(p.size()).cuda())
            m2 = torch.max(torch.add(torch.neg(p.data), - r), torch.zeros(p.size()).cuda())
        else:
            m1 = torch.max(torch.add(p.data, - r), torch.zeros(p.size()))
            m2 = torch.max(torch.add(torch.neg(p.data), - r), torch.zeros(p.size()))
        return p - Variable(m1 - m2)


class ProximalL1(nn.Module):
    def __init__(self):
        super(ProximalL1, self).__init__()

    def forward(self, x, f, clambda):
        if x.is_cuda:
            res = x + torch.clamp(f - x, -clambda, clambda).cuda()
        else:
            res = x + torch.clamp(f - x, -clambda, clambda)
        return res


class ProximalL2(nn.Module):
    def __init__(self, x, f, clambda):
        super(ProximalL2, self).__init__()
        self.x = x
        self.f = f
        self.clambda = clambda

    def forward(self):
        return

class PrimalUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalUpdate, self).__init__()
        self.backward_div = BackwardDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs):
        x = (x + self.tau * self.backward_div.forward(y, dtype=torch.cuda.FloatTensor) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x

class PrimalRegularization(nn.Module):
    def __init__(self, theta):
        super(PrimalRegularization, self).__init__()
        self.theta = theta

    def forward(self, x, x_tilde, x_old):
        x_tilde = x + self.theta * (x - x_old)
        return x_tilde


class DualUpdate(nn.Module):
    def __init__(self, sigma):
        super(DualUpdate, self).__init__()
        self.forward_grad = ForwardGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        if y.is_cuda:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.cuda.FloatTensor)
        else:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.FloatTensor)
        return y


class DualWeightedUpdate(nn.Module):
    def __init__(self, sigma):
        super(DualWeightedUpdate, self).__init__()
        self.forward_grad = ForwardWeightedGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y, w):
        if y.is_cuda:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, w, dtype=torch.cuda.FloatTensor)
        else:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, w, dtype=torch.FloatTensor)
        return y


class PrimalDualNetwork(nn.Module):
    def __init__(self, max_it=20, lambda_rof=7.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        super(PrimalDualNetwork, self).__init__()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.energy_primal = PrimalEnergyROF()
        self.energy_dual = 0.0
        self.w = nn.Parameter()
        # self.x = img_obs.clone()
        # img_size = img_obs.size()
        # self.y = Variable(torch.zeros((img_size[0]+1, img_size[1], img_size[2]))).cuda()
        # self.x_tilde = img_obs.clone()

    def forward(self, img_obs):
        x = img_obs.clone().cuda()
        x_tilde = img_obs.clone().cuda()
        img_size = img_obs.size()
        x_old = x.clone().cuda()
        y = Variable(torch.zeros((img_size[0] + 1, img_size[1], img_size[2]))).cuda()
        self.w.data = torch.ones(y.size())

        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w.cuda())
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)

        return x_tilde


class PrimalEnergyROF(nn.Module):
    def __init__(self):
        super(PrimalEnergyROF, self).__init__()

    def forward(self):
        return


class DualEnergyROF(nn.Module):
    def __init__(self):
        super(DualEnergyROF, self).__init__()

    def forward(self):
        return
