"""
Project:    
File:       main.py
Created by: louise
On:         10/9/17
At:         2:25 PM
"""
from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import skimage
from scipy.misc import face

from differential_operators import backward_divergence, forward_gradient
from proximal_operators import proximal_linf_ball

import data_set_patches

def dual_energy_tvl1(y, im_obs):
    """
    Compute the dual energy of TV-L1 problem.
    :param y: pytorch Variable, [MxNx2]
    :param im_obs: pytorch Variable, observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (im_obs - backward_divergence(y))**2
    nrg = torch.sum(nrg)
    return nrg


def dual_energy_rof(y, im_obs):
    """
    Compute the dual energy of ROF problem.
    :param y: pytorch Variable, [MxNx2]
    :param im_obs: pytorch Variables [MxN], observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (im_obs - backward_divergence(y))**2
    nrg = torch.sum(nrg)
    return nrg


def primal_energy_rof(x, img_obs, clambda):
    """

    :param x: pytorch Variables, [MxN]
    :param img_obs: pytorch Variable [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = torch.sum(torch.norm(forward_gradient(x), 1))
    energy_data_term = torch.sum(0.5*clambda * torch.norm(x - img_obs, 2))
    return energy_reg + energy_data_term


def primal_energy_tvl1(x, img_obs, clambda):
    """

    :param x: pytorch Variables, [MxN]
    :param img_obs: pytorch Variables [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = torch.sum(torch.norm(forward_gradient(x), 1))
    energy_data_term = torch.sum(clambda * torch.abs(x - img_obs))
    return energy_reg + energy_data_term


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


def imshow(tensor):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)

# cuda
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# images
imsize = 200  # desired size of the output image

loader = transforms.Compose([transforms.Scale(imsize),  # scale imported image
                             transforms.ToTensor()])  # transform it into a torch tensor

style = image_loader("images/picasso.jpg").type(dtype)
content = image_loader("images/dancing.jpg").type(dtype)

assert style.size() == content.size(), "we need to import style and content images of the same size"


# display
unloader = transforms.ToPILImage()  # reconvert into PIL image
fig = plt.figure()

plt.subplot(221)
imshow(style.data)
plt.subplot(222)
imshow(content.data)


if __name__ == '__main__':
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Create image to noise and denoise
    img_ = face(True)
    h, w = img_.shape
    img_.resize((h, w, 1))
    img_tensor = pil2tensor(img_.transpose(1, 0, 2))
    img_ref = Variable(img_tensor)
    img_obs = img_ref + Variable(torch.randn(img_ref.size()) * 0.1)
    # # Parameters
    # norm_l = 7.0
    # max_it = 200
    # theta = 1.0
    # tau = 0.01
    # sigma = 1.0 / (norm_l * tau)
    # #lambda_TVL1 = 1.0
    # lambda_rof = 7.0
    #
    # x = Variable(img_obs.data.clone())
    # x_tilde = Variable(img_obs.data.clone())
    # img_size = img_ref.size()
    # y = Variable(torch.zeros((img_size[0]+1, img_size[1], img_size[2])))
    #
    # p_nrg = primal_energy_rof(x, img_obs, lambda_rof)
    # print("Primal Energy = ", p_nrg)
    # #d_nrg = dual_energy_rof(y, img_obs)
    # #print("Dual Energy = ", d_nrg)
    #
    # # Solve ROF
    # primal = np.zeros((max_it,))
    # dual = np.zeros((max_it,))
    # gap = np.zeros((max_it,))
    # primal[0] = p_nrg.data[0]
    # #dual[0] = d_nrg.data[0]
    # y = forward_gradient(x)
    # for it in range(max_it):
    #     # Dual update
    #     y = y + sigma * forward_gradient(x_tilde)
    #     y = proximal_linf_ball(y, 1.0)
    #     # Primal update
    #     x_old = x
    #     x = (x + tau * backward_divergence(y) + lambda_rof * tau * img_obs) / (1.0 + lambda_rof * tau)
    #     # Smoothing
    #     x_tilde = x + theta * (x - x_old)
    #
    #     # Compute energies
    #     primal[it] = primal_energy_rof(x_tilde, img_obs, sigma).data[0]
    #     dual[it] = dual_energy_rof(y, img_obs).data[0]
    #     gap[it] = primal[it] - dual[it]
    #
    # plt.figure()
    # plt.plot(np.asarray(range(max_it)), primal, label="Primal Energy")
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(np.asarray(range(max_it)), dual, label="Dual Energy")
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(np.asarray(range(max_it)), gap, label="Gap")
    # plt.legend()
    #
    # # Plot reference, observed and denoised image
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # ax1.imshow(tensor2pil(img_ref.data))
    # ax1.set_title("Reference image")
    # ax2.imshow(tensor2pil(img_obs.data))
    # ax2.set_title("Observed image")
    # ax3.imshow(tensor2pil(x_tilde.data))
    # ax3.set_title("Denoised image")
    # plt.show()

# RNN for primal dual
data = data_set_patches.construct_patches(img_)
mean = [np.mean(data)]
std = [np.std(data)]
data.reshape(len(data), 8, 8)
normalize = transforms.Normalize(mean=mean, std=std)
preprocess = transforms.Compose([
   transforms.ToTensor(),
   normalize
])

trainloader = torch.utils.data.DataLoader(data, batch_size=4,
                                          shuffle=True, num_workers=2)
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs = data


    # wrap them in Variable
    inputs = Variable(inputs)

img_tensor = preprocess(data)
img_tensor.unsqueeze_(0)

img_variable = Variable(img_tensor)
fc_out = model(img_variable)




# content loss


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

# style loss


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion.forward(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

# load the cnn and build the model


cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()

# desired depth layers to compute style/content losses :
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# just in order to have an iterable access to or list of content/syle losses
content_losses = []
style_losses = []

model = nn.Sequential()  # the new Sequential module network
gram = GramMatrix()  # we need a gram module in order to compute style targets

# move these modules to the GPU if possible:
if use_cuda:
    model = model.cuda()
    gram = gram.cuda()

# weigth associated with content and style losses
content_weight = 1
style_weight = 1000

i = 1
for layer in list(cnn):
    if isinstance(layer, nn.Conv2d):
        name = "conv_" + str(i)
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, nn.ReLU):
        name = "relu_" + str(i)
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)

        i += 1

    if isinstance(layer, nn.MaxPool2d):
        name = "pool_" + str(i)
        model.add_module(name, layer)  # ***

print(model)

# input image

input = image_loader("images/dancing.jpg").type(dtype)
# if we want to fill it with a white noise:
#input.data = torch.randn(input.data.size()).type(dtype)

# add the original input image to the figure:
plt.subplot(223)
imshow(input.data)

# gradient descent

# this line to show that input is a parameter that requires a gradient
input = nn.Parameter(input.data)
optimizer = optim.LBFGS([input])

run = [0]
while run[0] <= 300:

    def closure():
        # correct the values of updated input image
        input.data.clamp_(0, 1)

        optimizer.zero_grad()
        model.forward(input)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.backward()
        for cl in content_losses:
            content_score += cl.backward()

        run[0]+=1
        if run[0] % 10 == 0:
            print("run " + str(run) + ":")
            print(style_score.data[0])
            print(content_score.data[0])

        return content_score+style_score

    optimizer.step(closure)

# a last correction...
input.data.clamp_(0, 1)

# finally enjoy the result:
plt.subplot(224)
imshow(input.data)
plt.show()