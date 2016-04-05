
import sys
import warnings
warnings.filterwarnings('ignore','.*')


import h5py
import cStringIO as StringIO
from PIL import Image

import numpy as np
import matplotlib.pyplot as pl
from path import Path
import pandas as pd
import cv2

import theano
import theano.tensor as T
import lasagne
import scipy.io

import pydnn
import aman_sandbox
import pyimg
from aman_sandbox.pseudofcn_rgbd import PseudoFcn_RGBD
#from pydnn.models.pseudofcn import PseudoFcn8_2
#from pydnn.expr2d import masked_error_rate_2d
import scipy.io

#Creating list to store the path of image and structured_labes
image_frames = []
gt_image_frames = []
hha_image_frames = []


#Base directory where data is available
base_dir = Path('/home/aman/data/NYUv2_HHA/')

print 'Reading Dataset'

# Getting all the pathes of files reading frome each directory

for base_dir_current in sorted(base_dir.dirs()):
    image_frames.append(base_dir_current/'image.jpg')
    #gt_image_frames.append(base_dir_current/'structure_labels.png')
    hha_image_frames.append(base_dir_current/'HHA.png')

#Loading the split data
split = scipy.io.loadmat('splits_NYUD.mat')
testNdxs = split.pop('testNdxs')
trainNdxs =  split.pop('trainNdxs')

print 'testing data length:', len(testNdxs)
print 'training data length:', len(trainNdxs)
test_idx = testNdxs.flatten()
train_idx = trainNdxs.flatten()
test_idx -= 1
train_idx -= 1

print 'Length of image_frames:', len(image_frames)
print 'Length of gt_frames:', len(gt_image_frames)
print 'Length of hha_frames:', len(hha_image_frames)

rng = np.random.RandomState(seed=28)
rng.shuffle(train_idx)
rng.shuffle(test_idx)

N_train = len(train_idx)
N_test = len(test_idx)

print 'Reading a sample image and HHA components'
img = cv2.imread(image_frames[300])
hha_img = cv2.imread(hha_image_frames[300], cv2.IMREAD_UNCHANGED) 
#gt_img = cv2.imread(gt_image_frames[300], cv2.IMREAD_UNCHANGED)

# Calculating mean values of R, G , B and H, H, A
mean_bgr = np.zeros(img.shape, dtype=np.float64)
mean_hha = np.zeros(hha_img.shape, dtype=np.float64)

for n in xrange(N_test):
    BGR = cv2.imread(image_frames[test_idx[n]])
    HHA = cv2.imread(hha_image_frames[test_idx[n]], cv2.IMREAD_UNCHANGED)
    mean_bgr += BGR
    mean_hha += HHA
mean_bgr /= N_train
mean_hha /= N_train

mean_bgr_px = mean_bgr.mean((0,1), keepdims=True)
mean_hha_px = mean_hha.mean((0,1), keepdims=True)

print 'Total number of images:', N_test

with h5py.File(base_dir/('means_test.h5'), 'w') as f:
    f.create_dataset('mean_bgr', data=mean_bgr)
    f.create_dataset('mean_bgr_px', data=mean_bgr_px)
    f.create_dataset('mean_hha', data=mean_hha)
    f.create_dataset('mean_hha_px', data=mean_hha_px)

