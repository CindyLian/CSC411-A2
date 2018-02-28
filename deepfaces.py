
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from pylab import *
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

import torch.nn as nn

# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'] 
act2 = ['bra','gil','har','bal','had','car']

def get_data (file):
    
    x= np.zeros((0,9216))
    y = np.zeros ((0,6))
    # model_orig = torchvision.models.alexnet(pretrained=True)
    model = MyAlexNet()
    model.eval()
    softmax = torch.nn.Softmax()
    
    for line in open (file + ".txt"):
        try:
            im = imread ("color227/" + line.strip())[:,:,:3]
        except:
            continue
            
        im = im - np.mean (im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis (im, -1).astype(np.float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)
        newX = softmax(model.forward(im_v)).data.numpy()[0]
        i = act2.index(line[:3])
        one_hot = np.zeros (6)
        one_hot[i] = 1
        x = np.vstack ((x, newX))
        y = np.vstack ((y, one_hot))
    
    return x,y


def Part10():
    train_x, train_y = get_data ("training")
    test_x, test_y = get_data ("test")
    val_x, val_y = get_data ("validation")
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    dim_h = 100
    dim_x = 9216
    dim_k = len(act) 
    
    train_idx = np.random.permutation(range(train_x.shape[0]))
    x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)
    
    x_val = Variable(torch.from_numpy(val_x), requires_grad=False).type(dtype_float)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.Sigmoid(),
        torch.nn.Linear(dim_h, dim_k),
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cost = np.zeros (10000)
    performance1 = np.zeros (10000)
    
    for t in range(10000):
        if t%100 == 0:
            print t
        y_pred = model(x)
        y_val = model (x_val).data.numpy()
        loss = loss_fn(y_pred, y_classes)
        
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to make a step
        cost [t] = loss [0]
        performance1 [t] = np.mean(np.argmax(y_val, 1) == np.argmax(val_y, 1))
    
    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    performance = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))

    print performance
    
    figure (0)
    plt.semilogy(range(10000), cost.squeeze())
    figure (1)
    plt.semilogy(range(10000), performance1.squeeze())
    plt.show()
    
    return model


