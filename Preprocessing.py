from __future__ import print_function
import sys, os
import pickle, gzip
import numpy as np
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt 
from utils import *


N_DATA = 20000
N_TRAIN = int(N_DATA*0.9)    
dataX = []

for i in range(N_DATA):
    path = "Data/%d.jpg"%(i) 
    img = misc.imread(path).astype(np.float) # load image 
    grayim = np.dot(img[...,:3],[0.299,0.587,0.114]) # gray scale
    dataX.append(grayim) 

f = open('Data/pass.txt')
labelY = f.read().split(' ')[:N_DATA]

dataY = []
for y in labelY:
    dataY.append(str2onehot(y))

trX = dataX[:N_TRAIN] # from 0 - N_TRAIN
teX = dataX[N_TRAIN:] # from N_TRAIN to end
trY = dataY[:N_TRAIN]
teY = dataY[N_TRAIN:]

pickle.dump((trX, teX, trY, teY), gzip.open( "data.pkl.gz", "wb" ))
