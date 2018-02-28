import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

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

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

act = ['bra','gil','har','bal','had','car']

def get_data (set):
    k = len (act)
    x = np.zeros ((0, 64*64))
    y = np.zeros ((0,k))
    
    for line in open (set + ".txt"):
        try:
            newX = imread ("cropped64/"+ line.strip())
        except IOError as e:
            continue
        newX = newX [:,:,0].flatten()/255.0 #take only 1 channel and flatten
        x = np.vstack((x, newX))
        one_hot = np.zeros (k)
        i = act.index(line[:3])
        one_hot[i] = 1
        y = np.vstack((y, one_hot))
    
    return x, y
        
def Part8 ():

    train_x, train_y = get_data ("training")
    test_x, test_y = get_data ("test")
    val_x, val_y = get_data ("validation")
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    dim_h = 100
    dim_x = 64*64
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
    
    
    #Part 9
    h1 = np.argmax (model[2].weight.data.numpy()[4,:])
    h2 = np.argmax (model[2].weight.data.numpy()[3,:])
    
    figure (3)
    plt.imshow(model[0].weight.data.numpy()[h1].reshape((64,64)),cmap=plt.cm.coolwarm)
    figure (4) 
    plt.imshow(model[0].weight.data.numpy()[h2].reshape((64,64)),cmap=plt.cm.coolwarm)
    
    plt.show ()
    
    return model
        
