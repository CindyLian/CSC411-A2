from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'] 

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

#baldwin121, bracco4, bracco150, carell71, carell 184, hader8, hader123, harmon82
bad_images = ['ce0080a17a4035b022cb79076ad961ae674acc585ede6afbf3023017c3fe5cce', 'cf7f2d4bdc1adcd3a970e88f3871291e077667011b5478ad486b525a91a8f6af', '0b3bc5d0e0d430bd2742c38e77fd5cfd47f839cd8f2cfb21f751cce892c314e2', 'd656de4f0632781593d6bbb406c70e18f1b5cda6b7055560ebcf81505c658bd8', '4b8621dd29e406641008213d893c96506a8a4f8fde50fce8bc362b98e12355fc', 'ea5b7ebd0c70c8041113862a7f53fae408d23d97b11b8c5afbca128de42178ff', '4f9a8a6f1377b03133bc4df0cc240d924e938bc4f163703d4222c796c5d0bd92','dbd552e744e53bc61f9e442ef21a60280aa599bd8d97bf952f9f17f874b9f609']
'''
for a in act:
    name = a.split()[1].lower()
    i = 0
    j = 0
    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            i += 1
            
            try:
                SHA = line.split ()[6]
            except:
                continue 
                
            if SHA in bad_images:
                continue
                
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 150)
            if not os.path.isfile("uncropped/"+filename):
                continue
            print filename

            try:                                                       
                img = imread("uncropped/"+filename)
            except:
                continue
                
            coordinates = map(int, line.split()[5].split(',')) 
            cropped = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]        
            resize = imresize(cropped, (64, 64))           
            try:
                image  = rgb2gray(resize)      
            except:
                image = resize
            
            imsave("cropped64/"+filename, image, cmap = 'gray')   #save file
            
            if j<20:
                f = open ('test.txt', 'a')
            elif j<30:
                f = open ('validation.txt','a')
            else:
                f = open ('training.txt', 'a')
                
            f.write ('\n' + filename)
            f.close ()
                
            j+=1

'''
#only works after images have already been downloaded to the correct folder
for a in act:
    name = a.split()[1].lower()
    i = 0
    j = 0
    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            i += 1
            
            try:
                SHA = line.split ()[6]
            except:
                continue 
                
            if SHA in bad_images:
                continue

            print filename

            try:                                                       
                img = imread("uncropped/"+filename)
            except:
                continue
                
            coordinates = map(int, line.split()[5].split(',')) 
            cropped = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]        
            resize = imresize(cropped, (64, 64))           
            
            imsave("color64/"+filename, resize)   #save file
