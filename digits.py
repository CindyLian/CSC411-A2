from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

def display_Part1 ():
    numbers = ["train0", "train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]
    f, axarr = plt.subplots (10, 10)
    for i in range (0,10):
        for j in range (0,10):
            axarr[i,j].imshow(M[numbers[i]][j].reshape((28,28)), cmap=cm.gray)
    show()    

def Part2 (w, x, b):
    o = np.matmul (w,x) + b
    p = exp (o)/np.sum (o)
    return o,p

def cost (w,x,b):
    o,p = Part2 (w,x,b)
    y = softmax (o)
    cost = -sum (y*log(p))
    return cost
    
def cost_grad (w,x,b):
    o = Part2 (w,x,b)
    p = softmax(o)
    
    return np.matmul (x,(p-o))    

def grad_approx (w,x,b,h,p,q):
    c = cost (w,x,b)
    w[p][q] += h
    
    return (cost(w,x,b)-c)/h

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    

#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl", "rb"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################