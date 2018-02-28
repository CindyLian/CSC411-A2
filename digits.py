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
import pickle
import cPickle

import os
from scipy.io import loadmat

M = loadmat("mnist_all.mat")

def display_Part1 ():
    numbers = ["train0", "train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]
    f, axarr = plt.subplots (10, 10)
    for i in range (0,10):
        for j in range (0,10):
            axarr[i,j].imshow(M[numbers[i]][j].reshape((28,28)), cmap=cm.gray)
    show()    


#o: size by 10
#x: size by 784
#w: 784 by 10
#b: 1 by 10
def Part2 (w, x, b):
    o = np.dot(x, w) + np.tile(b, (x.shape[0],1))
    p = np.exp(o)/np.sum(np.exp(o), axis=1).reshape((o.shape[0],1))
    return o,p

def Cost (x,y,p):
    cost = 0
    for i in range (p.shape[0]):
        cost += array(y[i]).dot (log(p[i]))  
    return -cost/p.shape[0]

def dC_dW (x,y,p):
    return np.dot (x.T, (p - y))

def dC_db (x,y,p):
    return np.dot (np.ones ((1, p.shape[0])), p - y)

def grad_approx (w,x,b,h,i,j,y):
    p = Part2(w,x,b)[1]
    c = Cost (x,y,p)
    w[i][j] += h
    p = Part2(w,x,b)[1]
    new_c = Cost (x,y,p)
    
    return (new_c-c)/h
    
def Part3b ():
    x = load_data (M)[0]
    y = load_data (M)[1]
    w = array([[0.00001 for i in xrange(10)] for i in xrange(784)])
    b = array([[0 for i in xrange(10)] for i in xrange(1)])
    
    p = Part2(w,x,b)[1]
    grad = dC_dW (x,array(y), p)
    

    h = 0.01
    for k in range (0,5):
        i = randint (0,784)
        j = randint (0,10)
        approx = grad_approx (w,x,b,h,i,j,y)*x.shape[0]
        print ("%d, %d, %f, %f, %f" %(i, j, grad[i][j], approx, (grad[i][j]-approx)*100/grad[i][j]))

def load_data (dict):
    x_training, y_training, x_test, y_test = [], [], [], []
    
    for i in range (0,10):
        label = np.zeros (10)
        label[i] = 1
        
        for j in range (0, len(M["train"+str(i)])):
            x_training.append (M["train"+str(i)][j])
            y_training.append (label)
        for j in range (0, len(M["test"+str(i)])):
            x_test.append (M["test"+str(i)][j])
            y_test.append (label)
    
    x_training = np.asarray(x_training)/255.0
    x_test = np.asarray(x_test)/255.0
                
    return x_training, np.asarray(y_training), x_test, np.asarray(y_test)
    
def test (w, x_test, y_test, b):
    p = Part2(w,x_test, b)[1]
    total = 0.
    correct = 0.
    
    for i in range (len(p)):
        if np.argmax(p[i]) == np.argmax (y_test[i]):
            correct += 1
        total +=1
    
    return correct/total

def gradient_descent (x,y,x_test, y_test, init_w, init_b, alpha,  EPS, max_iter):
    
    prev_w = init_w-10*EPS
    prev_b = init_b-10*EPS
    w = init_w.copy ()
    b = init_b.copy ()
    iter = 0
    iter_arr, accuracy, c = [], [], []

    while norm (w-prev_w) > EPS and norm (w-prev_w) > EPS and iter < max_iter:
        p = Part2(w,x,b)[1]
        prev_w = w.copy ()
        prev_b = b.copy ()
    
        w -= alpha*dC_dW(x,y,p)
        b -= alpha*dC_db(x,y,p)

        iter_arr.append (iter)
        accuracy.append (test(w,x_test, y_test,b))
        cost = Cost (x,y,p)
        c.append (cost)

        if iter % 10 == 0:
            print "Iteration", iter
            print 'Cost:', cost, '\n'
        
        iter += 1
    
    return w, b, iter_arr, accuracy, c
    
    
def Part4 ():
    x_training, y_training, x_test, y_test = load_data (M)
        
    w = np.asarray( [[0. for i in xrange(10)] for i in xrange(784)])
    b = np.zeros ((1,10))
    
    w, b, iter, acc, cost = gradient_descent (x_training, y_training, x_test, y_test, w, b, 0.00001, 0.000001, 500)
    
    
    for i in range (10):
        plt.figure(i)
        imshow(w[:,i].reshape((28,28)))
        
    plt.figure (10)
    plt.plot(iter, cost)
    plt.figure (11)
    plt.plot (iter, acc)
    plt.show ()
    
def Part5 ():
    x_training, y_training, x_test, y_test = load_data (M)
        
    w = np.asarray( [[0. for i in xrange(10)] for i in xrange(784)])
    b = np.zeros ((1,10))
    
    w, b, iter, acc, cost = gradient_descent_momentum(x_training, y_training, x_test, y_test, w, b, 0.00001, 0.000001, 500, 0.9)
    
    for i in range (10):
        plt.figure(i)
        imshow(w[:,i].reshape((28,28)))
        
    plt.figure (10)
    plt.plot(iter, cost)
    plt.figure(11)
    plt.plot(iter, acc)
    plt.show ()
    
def gradient_descent_momentum(x,y, x_test, y_test, init_w, init_b, alpha,  EPS, max_iter, gamma):
    prev_w = init_w-10*EPS
    prev_b = init_b-10*EPS
    w = init_w.copy ()
    b = init_b.copy ()
    iter = 0
    iter_arr, accuracy, c = [], [], []
    v = np.asarray( [[0. for i in xrange(10)] for i in xrange(784)])
    u = np.zeros ((1,10))

    while norm (w-prev_w) > EPS and norm (w-prev_w) > EPS and iter < max_iter:
        p = Part2(w,x,b)[1]
        prev_w = w.copy ()
        prev_b = b.copy ()
    
        v = gamma * v + alpha * dC_dW(x,y,p)
        w = w - v
        u = gamma * u + alpha * dC_db(x,y,p)
        b = b - u

        iter_arr.append (iter)
        accuracy.append (test(w,x_test, y_test,b))
        cost = Cost (x,y,p)
        c.append (cost)

        if iter % 10 == 0:
            print "Iteration", iter
            print 'Cost:', cost, '\n'
        
        iter += 1
    
    return w, b, iter_arr, accuracy, c
    
def Part6a():
    x_training, y_training, x_test, y_test = load_data (M)
    W = pickle.load(open('part_5.pickle', 'rb'))    
    w, b = W['w'], W['b']
    x, y = w.shape[0], w.shape[1]
    x1, y1 = 400, 5
    x2, y2 = 500, 6
    w1 = w[x1, y1]
    w2 = w[x2, y2]
        
    w1_axis = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2_axis = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    X,Y = np.meshgrid(w1_axis, w2_axis)
    Z = cost_6 (X, Y, {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x': x_training, 'y': y_training, 'w': w, 'b': b})

    plt.figure(1)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    title('Contour plot')
    plt.show()
    
@np.vectorize
def cost_6 (w1, w2, weights):
    weights['w'][weights['x1'], weights['y1']] = w1
    weights['w'][weights['x2'], weights['y2']] = w2
    p = Part2(weights['w'], weights['x'], weights['b'])[1]
    cost = Cost(0, weights['y'], p)
    return cost
    
def Part6bc():
    x_training, y_training, x_test, y_test = load_data (M)
    W = pickle.load(open('part_5.pickle', 'rb'))    
    w, b = W['w'], W['b']
    x, y = w.shape[0], w.shape[1]
    x1, y1 = 400, 5
    x2, y2 = 500, 6
    w1 = w[x1, y1]
    w2 = w[x2, y2]
        
    w1_axis = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2_axis = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    X,Y = np.meshgrid(w1_axis, w2_axis)
    Z = cost_6 (X, Y, {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x': x_training, 'y': y_training, 'w': w, 'b': b})

    plt.figure(1)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    title('Contour plot')
    
    w[x1, y1] = 0
    w[x2, y2] = 0
    
    van = gradient_descent_6(x_training, y_training, x_test, y_test, w, b, x1, y1, x2, y2, 0.0005, 0.0000001, 50)
    van1, van2 = van[2], van[3]
    
    mom = gradient_descent_6_momentum(x_training, y_training, x_test, y_test, w, b, x1, y1, x2, y2, 0.0006, 0.9, 0.0000001, 50)
    mom1, mom2 = mom[2], mom[3]
    
    plt.figure(2)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    plt.plot(van1, van2, 'bo-', label="Vanilla")
    plt.legend()
    
    plt.figure(3)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    plt.plot(mom1, mom2, 'go-', label="Momentum")
    plt.legend()

    plt.figure(4)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    plt.plot(van1, van2, 'bo-', label="Vanilla")
    plt.plot(mom1, mom2, 'go-', label="Momentum")
    plt.legend()
    
    plt.show()

def Part6e():
    x_training, y_training, x_test, y_test = load_data (M)
    W = pickle.load(open('part_5.pickle', 'rb'))    
    w, b = W['w'], W['b']
    x, y = w.shape[0], w.shape[1]
    x1, y1 = 100, 1
    x2, y2 = 100, 7
    w1 = w[x1, y1]
    w2 = w[x2, y2]
        
    w1_axis = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2_axis = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    X,Y = np.meshgrid(w1_axis, w2_axis)
    Z = cost_6 (X, Y, {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x': x_training, 'y': y_training, 'w': w, 'b': b})

    plt.figure(1)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    title('Contour plot')
    
    w[x1, y1] = 0
    w[x2, y2] = 0
    
    van = gradient_descent_6(x_training, y_training, x_test, y_test, w, b, x1, y1, x2, y2, 0.002, 0.0000001, 50)
    van1, van2 = van[2], van[3]
    
    mom = gradient_descent_6_momentum(x_training, y_training, x_test, y_test, w, b, x1, y1, x2, y2, 0.0002, 0.9, 0.0000001, 50)
    mom1, mom2 = mom[2], mom[3]
    
    plt.figure(2)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    plt.plot(van1, van2, 'bo-', label="Vanilla")
    plt.legend()
    
    plt.figure(3)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    plt.plot(mom1, mom2, 'go-', label="Momentum")
    plt.legend()

    plt.figure(4)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel ("w1")
    plt.ylabel ("w2")
    plt.plot(van1, van2, 'bo-', label="Vanilla")
    plt.plot(mom1, mom2, 'go-', label="Momentum")
    plt.legend()
    
    plt.show()

    
def gradient_descent_6 (x, y, x_test, y_test, init_w, init_b, x1, y1, x2, y2, alpha, EPS, max_iter):
    prev_w = init_w-10*EPS
    prev_b = init_b-10*EPS
    w = init_w.copy ()
    b = init_b.copy ()
    iter = 0
    iter_arr, accuracy, c = [], [], []
    w1 = [w[x1,y1]]
    w2 = [w[x2,y2]]

    while norm (w-prev_w) > EPS and norm (w-prev_w) > EPS and iter < max_iter:
        p = Part2(w,x,b)[1]
        prev_w = w.copy ()
        prev_b = b.copy ()
    
        w[x1, y1] -= alpha*dC_dW(x, y, p)[x1, y1]
        w[x2, y2] -= alpha*dC_dW(x, y, p)[x2, y2]
        b[0,y1] -= alpha*dC_db(x, y, p)[0,y1]
        b[0,y2] -= alpha*dC_db(x, y, p)[0,y2]
        w1.append(w[x1][y1])
        w2.append(w[x2][y2])
        iter_arr.append (iter)
        accuracy.append (test(w,x_test, y_test,b))
        cost = Cost (x,y,p)
        c.append (cost)

        if iter % 10 == 0:
            print "Iteration", iter
            print 'Cost:', cost, '\n'
        
        iter += 1
    
    return w, b, w1, w2, iter_arr, accuracy, c
    
def gradient_descent_6_momentum (x,y,x_test, y_test, init_w, init_b, x1, y1,x2, y2, alpha, gamma, EPS, max_iter):
    prev_w = init_w-10*EPS
    prev_b = init_b-10*EPS
    w = init_w.copy ()
    b = init_b.copy ()
    iter = 0
    iter_arr, accuracy, c = [], [], []
    w1 = [w[x1,y1]]
    w2 = [w[x2,y2]]
    v1, v2, u1, u2 = 0, 0, 0, 0

    while norm (w-prev_w) > EPS and norm (w-prev_w) > EPS and iter < max_iter:
        p = Part2(w,x,b)[1]
        prev_w = w.copy ()
        prev_b = b.copy ()
        
        v1 = v1 * gamma + alpha*dC_dW(x, y, p)[x1,y1]
        v2 = v2 * gamma + alpha*dC_dW(x, y, p)[x2,y2]
        u1 = u1 * gamma + alpha*dC_db(x, y, p)[0,y1]
        u2 = u2 * gamma + alpha*dC_db(x, y, p)[0,y2]
        w[x1, y1] -= v1
        w[x2, y2] -= v2
        b[0,y1] -= u1
        b[0,y2] -= u2
        w1.append(w[x1][y1])
        w2.append(w[x2][y2])
        iter_arr.append (iter)
        accuracy.append (test(w,x_test, y_test,b))
        cost = Cost (x,y,p)
        c.append (cost)

        if iter % 10 == 0:
            print "Iteration", iter
            print 'Cost:', cost, '\n'
        
        iter += 1
    
    return w, b, w1, w2, iter_arr, accuracy, c