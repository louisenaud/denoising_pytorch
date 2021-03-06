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
        """
        
        :param x: 
        :param dtype: 
        :return: 
        """
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
        """
        
        :param x: 
        :param w: 
        :param dtype: 
        :return: 
        """
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
        """
        
        :param y: 
        :param dtype: 
        :return: 
        """
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


class BackwardWeightedDivergence(nn.Module):
    def __init__(self):
        super(BackwardWeightedDivergence, self).__init__()

    def forward(self, y, w, dtype=torch.cuda.FloatTensor):

        im_size = y.size()
        y_w = w * y
        # Horizontal direction
        d_h = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_h[0, :, 0] = y_w[0, :, 0]
        d_h[0, :, 1:-1] = y_w[0, :, 1:-1] - y_w[0, :, :-2]
        d_h[0, :, -1] = -y_w[0, :, -2:-1]

        # Vertical direction
        d_v = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_v[0, 0, :] = y_w[1, 0, :]
        d_v[0, 1:-1, :] = y_w[1, 1:-1, :] - y_w[1, :-2, :]
        d_v[0, -1, :] = -y_w[1, -2:-1, :]

        # Divergence
        div = d_h + d_v
        return div


class ProximalLinfBall(nn.Module):
    def __init__(self):
        super(ProximalLinfBall, self).__init__()

    def forward(self, p, r):
        """
        
        :param p: 
        :param r: 
        :return: 
        """
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
        """
        
        :param x: 
        :param f: 
        :param clambda: 
        :return: 
        """
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
        """
        
        :param x: 
        :param y: 
        :param img_obs: 
        :return: 
        """
        x = (x + self.tau * self.backward_div.forward(y, dtype=torch.cuda.FloatTensor) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalWeightedUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalWeightedUpdate, self).__init__()
        self.backward_div = BackwardWeightedDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs, w):
        """

        :param x:
        :param y:
        :param img_obs:
        :param w:
        :return:
        """
        x = (x + self.tau * self.backward_div.forward(y, w, dtype=torch.cuda.FloatTensor) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalRegularization(nn.Module):
    def __init__(self, theta):
        super(PrimalRegularization, self).__init__()
        self.theta = theta

    def forward(self, x, x_tilde, x_old):
        """
        
        :param x: 
        :param x_tilde: 
        :param x_old: 
        :return: 
        """
        x_tilde = x + self.theta * (x - x_old)
        return x_tilde


class DualUpdate(nn.Module):
    def __init__(self, sigma):
        super(DualUpdate, self).__init__()
        self.forward_grad = ForwardGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        """
        
        :param x_tilde: 
        :param y: 
        :return: 
        """
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
        """
        
        :param x_tilde: 
        :param y: 
        :param w: 
        :return: 
        """
        if y.is_cuda:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, w, dtype=torch.cuda.FloatTensor)
        else:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, w, dtype=torch.FloatTensor)
        return y


class PrimalDualNetwork(nn.Module):
    def __init__(self, w, max_it=20, lambda_rof=7.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        super(PrimalDualNetwork, self).__init__()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.energy_primal = PrimalEnergyROF()
        self.energy_dual = 0.0
        self.w = w

    def forward(self, img_obs):
        """
        
        :param img_obs: 
        :return: 
        """
        x = img_obs.clone().cuda()
        x_tilde = img_obs.clone().cuda()
        img_size = img_obs.size()
        x_old = x.clone().cuda()
        y = Variable(torch.rand((img_size[0] + 1, img_size[1], img_size[2]))).cuda()

        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w.cuda())
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w.cuda())
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)

        return x_tilde


class PrimalEnergyROF(nn.Module):
    def __init__(self):
        super(PrimalEnergyROF, self).__init__()

    def forward(self, x, img_obs, clambda):
        """

        :param x: pytorch Variables, [MxN]
        :param img_obs: pytorch Variable [MxN], observed image
        :param clambda: float, lambda parameter
        :return: float, primal ROF energy
        """
        g = ForwardWeightedGradient()
        energy_reg = torch.sum(torch.norm(g.forward(x, torch.cuda.FloatTensor), 1))
        energy_data_term = torch.sum(0.5 * clambda * torch.norm(x - img_obs, 2))
        return energy_reg + energy_data_term


class DualEnergyROF(nn.Module):
    def __init__(self):
        super(DualEnergyROF, self).__init__()

    def forward(self, y, im_obs):
        """
        Compute the dual energy of ROF problem.
        :param y: pytorch Variable, [MxNx2]
        :param im_obs: pytorch Variables [MxN], observed image
        :return: float, dual energy
        """
        d = BackwardDivergence()
        nrg = -0.5 * (im_obs - d.forward(y, torch.cuda.FloatTensor)) ** 2
        nrg = torch.sum(nrg)
        return nrg


class PrimalDualGap(nn.Module):
    def __init__(self):
        super(PrimalDualGap, self).__init__()

    def forward(self, x, y, im_obs, clambda):
        """
        Compute the primal dual gap.
        :param x: pytorch Variable, [1xMxN]
        :param y: pytorch Variable, [2xMxN]
        :param im_obs: pytorch Variable, [1xMxN], observed image
        :param clambda: float > 0
        :return: float > 0
        """
        dual = DualEnergyROF()
        primal = PrimalEnergyROF()
        g = primal.forward(x, im_obs, clambda) - dual.forward(y, im_obs)
        return g


class MaxMarginLoss(nn.Module):
    def __init__(self):
        super(MaxMarginLoss, self).__init__()

    def forward(self, x, H, b, w):
        fg = ForwardWeightedGradient()
        output = torch.transpose(x, 0, 1) * x + torch.transpose(b, 0, 1) * x + torch.sum(torch.norm(fg.forward(x, w), 1))
        return output
