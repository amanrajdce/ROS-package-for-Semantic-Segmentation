import sys
import warnings
warnings.filterwarnings('ignore','.*')
import h5py

from PIL import Image

import numpy as np
import matplotlib.pyplot as pl
from path import Path
import cv2

import theano
import theano.tensor as T
import lasagne
import scipy.io

import pydnn
import aman_sandbox
import pyimg
from aman_sandbox.pseudofcn_rgbd import PseudoFcn_RGBD
from pydnn.models import SimpleDnsModel2
from draw_net import draw_to_file
from pyml_util import metrics_db

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

N = len(image_frames)

rng = np.random.RandomState(seed=28)
rng.shuffle(train_idx)
rng.shuffle(test_idx)

N_train = len(train_idx)
N_test = len(test_idx)

print 'Analyzing model parameters'

batch_size = 10
num_class = 4
model = PseudoFcn_RGBD(batch_size, num_class)

#model = SimpleDnsModel2(batch_size,num_class)

print 'Model layers with output data blob'
layers = lasagne.layers.helper.get_all_layers(model.l_out)
for layer in layers:
    print layer.name, layer.get_output_shape()

layers = lasagne.layers.get_all_layers(model.l_out)
draw_to_file(layers, 'network.pdf', output_shape=True, verbose=True)

#sys.exit()
    
H, W = model.input_height, model.input_width # 
h, w = model.l_out.get_output_shape()[2:]  # 
X_data = np.zeros((N, 6, H, W), dtype=np.float32)
y_data = np.zeros((N, 1, h, w), dtype=np.float32)

model.l_out.get_output_shape() # Shape of final output with batch_size

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
    y_data[n,0] = np.asarray(gt_img).astype(np.float32) #shape is (1,h,w)

print 'X_data Shape:', X_data.shape
print 'y_data Shape:', y_data.shape


print 'Testing data'
X_data_j, y_data_j = img_jit.preprocess(X_data[test_idx], y_data[test_idx]) # preprocessing training data
print 'Shape of X_data_j :', X_data_j.shape
print 'Shape of y_data_j :', y_data_j.shape

# Creating the theano shared variable and copying training data to GPU
X_shared = theano.shared(X_data_j)
y_shared = theano.shared(y_data_j)

#Defining Theano varaibles
print 'Creating theano variables & model..'
X_batch = model.X
y_batch = model.y1
batch_index = T.iscalar('b_ix')
batch_slice = slice(batch_index*model.batch_size, (batch_index+1)*batch_size)

#Defining test output
dout = model.get_deterministic_output(X_batch) #difference b/w both outputs

#Defining error
err_test = model.get_error_rate(dout,y_batch)

#Compiling prediction function
pred_fn = theano.function([batch_index],[model.get_prediction_image(dout),err_test],
                          givens={X_batch: X_shared[batch_slice], y_batch: T.cast(y_shared[batch_slice], 'int32')})


metrics_log =  metrics_db.ShelveDataLogger(base_dir/('test_metrics.shelf'), reinitialize=True)

print('Opening saved model')
#fnames = sorted(base_dir.files( 'epoch*.h5'))
#weights_fname = fnames[-1]
weights_fname = '/home/aman/data/NYUv2_HHA/epoch_00012.h5'
with h5py.File( weights_fname,'r') as f:
    orig_weights = model.get_param_values()
    weights = []
    for ix,w in enumerate( orig_weights ):
        weights.append(f['w%03d'%ix].value)
    model.set_param_values(weights)
    print 'Total weights loaded:', len(weights)
   
#sys.exit()
# Testing model
print 'Testing model ....'
num_batches_validate = len(train_idx)//batch_size
print 'number of testing batches:', num_batches_validate

X_data_j, y_data_j = img_jit.preprocess(X_data[train_idx], y_data[train_idx], deterministic=True)
X_shared.set_value(X_data_j, borrow=True)
y_shared.set_value(y_data_j, borrow=True)

for bi in xrange(num_batches_validate):
    [yhat_b,erro] = pred_fn(bi) #testing is also batch type??
    print('batch {}, error: {}'.format(bi, erro))
    metrics_log.log({'batch':bi, 'average error':erro})
    #printing result every 10th validation batch
    if(bi%10)==0:
        for yi in xrange(batch_size):
            yhat00 = yhat_b[yi,0]+1
            name = 'pred_bat'+ str(bi)+'_yi'+str(yi)+'.png'
            #name1 = 'pred_bat'+str(bi)+'_yi'+str(yi)+'.jpg'
            #cv2.imwrite(name,yhat00)
            img = np.asarray(Image.open(image_frames[train_idx[(bi*batch_size)+yi]]).resize((320,320)))
            gt_img = np.asarray(Image.open(gt_image_frames[train_idx[(bi*batch_size)+yi]]).resize((320,320)))
            #print 'output shape:', yhat_b.shape
            #Plotting test results
            yhat00_rgb = rgb_remapper(yhat00)
            gt_img_rgb = rgb_remapper(gt_img)
            #pl.imshow(yhat00_rgb);pl.axis('off'), pl.title('Prediction');
            #pl.savefig(name1);pl.close();
            
            pl.subplot(1, 3, 1)
            pl.imshow(img); pl.axis('off'); pl.title('Input');
            pl.subplot(1, 3, 2)
            pl.imshow(gt_img_rgb); pl.axis('off'); pl.title('Ground Truth');
            pl.subplot(1, 3, 3)
            pl.imshow(yhat00_rgb); pl.axis('off'); pl.title('Prediction');
            pl.savefig(name); pl.close();
