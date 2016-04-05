import sys
import warnings
warnings.filterwarnings('ignore','.*')
import h5py

#import cStringIO as StringIO
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
from pyml_util import metrics_db
import aman_sandbox
import pyimg
from aman_sandbox.pseudofcn_rgbd import PseudoFcn_RGBD

#Creating list to store the path of image and structured_labes
image_frames = []
gt_image_frames = []
hha_image_frames = []


#Base directory where data is available
base_dir = Path('/home/aman/data/NYUv2_HHA/')
#for color mapping of labels and prediction
rgb_to_dict = {}
rgb_to_dict[0] = np.array((0,0,0), np.uint8) #black(void)
rgb_to_dict[1] = np.array((255, 128, 0), np.uint8) #orange
rgb_to_dict[2] = np.array((0,255,0), np.uint8) #green
rgb_to_dict[3] = np.array((0,0,255), np.uint8) #blue
rgb_to_dict[4] = np.array((255,0,255), np.uint8) #pink
rgb_remapper = pyimg.cyimg.RgbRemapper(rgb_to_dict)


print 'Reading Dataset'

# Getting all the pathes of files reading frome each directory

for base_dir_current in sorted(base_dir.dirs()):
    image_frames.append(base_dir_current/'image.jpg')
    gt_image_frames.append(base_dir_current/'structure_labels.png')
    hha_image_frames.append(base_dir_current/'HHA.png')


#loads the split
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

N = len(image_frames)

rng = np.random.RandomState(seed=28)
rng.shuffle(train_idx)
rng.shuffle(test_idx)

N_train = len(train_idx)
N_test = len(test_idx)

print 'Analyzing model parameters'

batch_size = 5
num_class = 4
model = PseudoFcn_RGBD(batch_size, num_class)

print 'Model layers with output data blob'
layers = lasagne.layers.helper.get_all_layers(model.l_out)
for layer in layers:
    print layer.name, layer.get_output_shape()

#sys.exit()
print 'Reading a sample image and HHA components'
img = cv2.imread(image_frames[300])
gt_img = cv2.imread(gt_image_frames[300], cv2.IMREAD_UNCHANGED)
hha_img = cv2.imread(hha_image_frames[300], cv2.IMREAD_UNCHANGED) # to get in meters

#mean_px_dp = np.mean(dp_img) #Calculating the mean of depth image
#std_px_dp = np.std(dp_img)  #Calculating standard deviation of depth image
#print 'Original depth: \n', dp_img
#print ' \n Mean pixel value:', mean_px_dp
#dp_img_meanstd = (dp_img - mean_px_dp)/std_px_dp # Applying mean and standard deviation
#print '\n Standard Deviation:', std_px_dp
#print '\n Sub mean and Div std: \n', dp_img_meanstd

#dp_scaled = dp_img_meanstd + 255  #Scaling depth image
#print '\n Scaled: \n', dp_scaled
#print '\n RGB Image:', img[:,:,0]
pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); pl.axis('off'); pl.title('Original Image'); pl.savefig("image.png"); pl.close();
gt_img_rgb = rgb_remapper(gt_img)
pl.imshow(gt_img_rgb); pl.axis('off'); pl.title('Ground Truth'); pl.savefig("gt.png");pl.close();

#HHA components
hozdisp = hha_img[:,:,2]
heignd = hha_img[:,:,1]
ang_grav = hha_img[:,:,0]

#Plotting HHA components
pl.imshow(hozdisp); pl.axis('off'); pl.title('Horizontal Disparity'); pl.savefig("hozdips.png");
pl.close();
pl.imshow(heignd); pl.axis('off'); pl.title('Height above ground'); pl.savefig("heignd.png");
pl.close();
pl.imshow(ang_grav); pl.axis('off'); pl.title('Angle with gravity'); pl.savefig("ang_grav.png");
pl.close();
#print gt_img[:,600]

H, W = model.input_height, model.input_width # H=227, W = 227
h, w = model.l_out.get_output_shape()[2:]  # h=, w=
X_data = np.zeros((N, 6, H, W), dtype=np.float32)
y_data = np.zeros((N, 1, h, w), dtype=np.float32)

model.l_out.get_output_shape() # Shape of final output with batch_size=8

img_jit = pydnn.preprocessing.ImageBatchJitterer(x_output_shape=(H,W), y_output_shape=(h,w))


with h5py.File(base_dir/('means.h5'), 'r') as f:
    mean_bgr_px = f['mean_bgr_px'].value
    mean_hha_px = f['mean_hha_px'].value

print 'Loading dataset ...'
for n in xrange(N):
    img = cv2.imread(image_frames[n])
    hha = cv2.imread(hha_image_frames[n], cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (H, W), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    hha = cv2.resize(hha, (H, W), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)-mean_bgr_px
    hha = hha.astype(np.float32)-mean_hha_px

    img_hha = np.dstack((img, hha))

    gt_img = cv2.imread(gt_image_frames[n], cv2.IMREAD_UNCHANGED)
    gt_img = np.asarray(cv2.resize(gt_img,(h,w), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)).copy()

    X_data[n] = np.rollaxis(img_hha, 2).astype(np.float32)  # shape is (c,H,W)
    #X_data[n] = np.asarray(img_hha).astype(np.float32)
    y_data[n,0] = np.asarray(gt_img).astype(np.float32) #shape is (1,h,w)

print 'X_data Shape:', X_data.shape
print 'y_data Shape:', y_data.shape


#visualizing a smaple input ground truth image after dimension adjustment
newgt = y_data[300,0]
newgt_rgb = rgb_remapper(newgt)
pl.imshow(newgt_rgb); pl.axis('off'); pl.title('New Ground Truth'); pl.savefig("new_gt.png");
pl.close();
#pl.imshow(gt_img_rgb); pl.axis('off'); pl.title('Ground Truth'); pl.show();

print 'Training data'
X_data_j, y_data_j = img_jit.preprocess(X_data[train_idx], y_data[train_idx]) # preprocessing training data
print 'Shape of X_data_j :', X_data_j.shape
print 'Shape of y_data_j :', y_data_j.shape

X_shared = theano.shared(X_data_j)

# Creating the theano shared variable and copying training data to GPU

y_shared = theano.shared(y_data_j)

#Defining Theano varaibles
print 'Creating theano variables & model..'
X_batch = model.X
y_batch = model.y1

batch_index = T.iscalar('b_ix')
batch_slice = slice(batch_index*model.batch_size, (batch_index+1)*batch_size)

#Defining outputs
out = model.l_out.get_output(X_batch) # defining training output
dout = model.get_deterministic_output(X_batch) #defining testing output

#Defining loss
loss = model.get_loss(out, y_batch) + 0.0005*model.get_l2_norm()

#Defining error
err = model.get_error_rate(out, y_batch)
err_test = model.get_error_rate(dout,y_batch)

#Getting and updating all the paramters of model
all_params = model.get_params()
updates = lasagne.updates.momentum(loss, all_params, 0.001, 0.9)

#Compiling prediction function
pred_fn = theano.function([batch_index],[model.get_prediction_image(dout),err_test],
                          givens={X_batch: X_shared[batch_slice], y_batch: T.cast(y_shared[batch_slice], 'int32')})


print 'Prediction before training'
yi = 3
X_data_j, y_data_j = img_jit.preprocess(X_data[test_idx], y_data[test_idx], deterministic=True)
X_shared.set_value(X_data_j, borrow=True) #settin the data to testing set
y_shared.set_value(y_data_j, borrow=True)
[yhat_b,error] = pred_fn(0) # output shape=
yhat00 = yhat_b[yi,0]+1  #adding 1 to avoid void presence
img = np.asarray(Image.open(image_frames[test_idx[yi]]).resize((256, 256)))
gt_img = np.asarray(Image.open(gt_image_frames[test_idx[yi]]).resize((256, 256)))

 
#Showing results
pl.subplot(1, 3, 1)
pl.imshow(img); pl.axis('off'); pl.title('Input');
pl.subplot(1, 3, 2)
gt_img_rgb = rgb_remapper(gt_img)
pl.imshow(gt_img_rgb, interpolation='nearest'); pl.axis('off'); pl.title('Ground Truth')
pl.subplot(1, 3, 3)
yhat00_rgb = rgb_remapper(yhat00)
pl.imshow(yhat00_rgb, interpolation='nearest'); pl.axis('off'); pl.title('Prediction')
pl.savefig("naive_predict.png"); pl.close();
print 'Prediction Error:', error

#sys.exit()
#Batch training takes input batch index and gives loss and error as output, apply updates to loss
iter_train = theano.function([batch_index], [loss, err], updates=updates, allow_input_downcast=True,
                            givens={X_batch: X_shared[batch_slice], y_batch: T.cast(y_shared[batch_slice], 'int32')})


num_batches = len(train_idx)//batch_size
print 'number of training batches:', num_batches

#stores loss and erro at each training example
metrics_log1 = metrics_db.ShelveDataLogger(base_dir/('train_metrics_all.shelf'), reinitialize=True)
metrics_log2 = metrics_db.ShelveDataLogger(base_dir/('train_metrics_epoch.shelf'), reinitialize=True)


print('saving model')
print 'Model parameters:', model.get_params()
weights = model.get_param_values()
weights_fname = base_dir/('epoch_00000.h5')
print('saving %s'%weights_fname)
with h5py.File( weights_fname, 'w') as f:
    for ix,w in enumerate( weights ):
        f.create_dataset( 'w%03d'%ix , data=w, compression='gzip')


##initialzing some model layes with weights
params = lasagne.layers.get_all_params(model.l_out)
names = [p.name for p in params if p.name]

'''
print('Opening saved model')
weight_fnames = '/home/aman/data/NYUv2_HHA/epoch_00012.h5'
with h5py.File( weight_fnames, 'r') as f:
    param_dict = {}
    for ix in xrange(0,len(names)):
        param_dict[names[ix]] = f['w%03d'%ix].value
'''

print('loading weights of VGG-16 caffe model...')
caffe_model = 'caffe_weight.h5'

#dict to store all weight before assigning
param_dict = {}
with h5py.File( caffe_model, 'r') as f:
    for ix in xrange(4,28):
        param_dict[names[ix]] = f[names[ix]].value
        if names[ix] == 'conv1_2.W':
            print 'assigning values to conv6_2.W and conv6_3.W here'
            param_dict['conv6_2.W'] = f['conv1_2.W'].value
            param_dict['conv6_3.W'] = f['conv1_2.W'].value
        if names[ix] == 'conv1_2.b':
            print 'assigning values to conv6_2.b and conv6_3.b here'
            param_dict['conv6_2.b'] = f['conv1_2.b'].value
            param_dict['conv6_3.b'] = f['conv1_2.b'].value
f.close

#assigning rest layer weights from previous training
print('Loading some weights from previous model...')
epoch_final = 'epoch_final.h5'
with h5py.File( epoch_final, 'r') as f:
    for ix in xrange(0,4):
        param_dict[names[ix]] = f['w%03d'%ix].value

    #assigning weights to remaining layers
    param_dict[names[28]] = f['w028'].value
    param_dict[names[29]] = f['w029'].value 
    param_dict[names[30]] = f['w030'].value 
    param_dict[names[31]] = f['w031'].value 
    param_dict[names[36]] = f['w036'].value 
    param_dict[names[37]] = f['w037'].value 

f.close
#Setting up weights for all layers
print('Assigning all weights now...')
print 'total weights initialized:', len(param_dict)

model.set_param_values_from_dict(param_dict)

num_batches_validate = len(test_idx)//batch_size

# Training of model starts
print 'Training model....'
for epoch in xrange(120):
    X_data_j, y_data_j = img_jit.preprocess(X_data[train_idx], y_data[train_idx])
    X_shared.set_value(X_data_j, borrow=True) # setting the data to training set
    y_shared.set_value(y_data_j, borrow=True)
    losses = []
    errors = []
    for bi in xrange(num_batches):
        los, erro = iter_train(bi)
        losses.append(los)
        errors.append(erro)
        metrics_log1.log({'loss':los, 'error':erro, 'epoch':epoch, 'batch':bi})
        print('epoch: {}, batch: {}, loss: {}'.format(epoch,bi,los))

    #Saving model after every 2 epoch
    if(epoch%1 == 0):
        print('saving model')
        weights = model.get_param_values()
        weights_fname = base_dir/('epoch_%05d.h5'%(epoch))
        print('saving %s'%weights_fname)
        with h5py.File( weights_fname, 'w') as f:
            for ix,w in enumerate( weights ):
                f.create_dataset( 'w%03d'%ix , data=w, compression='gzip')

    print('epoch {}, average loss: {}, average error: {} \n'.format(epoch, np.mean(losses), np.mean(errors))),
    #metrics_log2.log({'epoch':epoch, 'average loss':np.mean(losses), 'average error':np.mean(errors)})
    
    print('Validating the model ...')
    X_data_j, y_data_j = img_jit.preprocess(X_data[test_idx], y_data[test_idx], deterministic=True)
    X_shared.set_value(X_data_j, borrow=True)
    y_shared.set_value(y_data_j, borrow=True)
    avg_error = []
    for bi in xrange(num_batches_validate):
        [yhat_b, erro] = pred_fn(bi)
        avg_error.append(erro)
    print 'Validation error:', np.mean(avg_error)
    metrics_log2.log({'epoch':epoch, 'average loss':np.mean(losses), 'average error':np.mean(errors), 'test_error': np.mean(avg_error)})
   
print 'Training Completed'
