import h5py
import caffe
from path import Path
import numpy as np
import sys

'''
Reads a caffe model file and stores it's weights in h5py file to be used for theano models

'''

caffe_root = Path('/home/aman/caffe-master')
sys.path.insert(0,caffe_root/'python')

#defining paths for deploy and model file
deploy_path = Path('/home/aman/mavscout_semantic/src/aman_sandbox/notebooks/deploy.prototxt')
caffe_path = Path('/home/aman/mavscout_semantic/src/aman_sandbox/notebooks/VGG_ILSVRC_16_layers.caffemodel')

#defining the net
net = caffe.Net(str(deploy_path), str(caffe_path),caffe.TEST)

#parameters in the net
print net.params.items()

#Reading weights and biases of each layer 
keys = []

for k,v in net.params.items():
    weight,bias = v
    print k, weight.data.shape
    print k, bias
    #storing all layer names 
    keys.append(k)


#Analyzing the data blobs and output shapes
for k, v in net.blobs.items():
    print k, v.channels, v.width, v.height


#Function to extract conv_params from model
def extract_conv_params(net,k):
    w = net.params[k][0].data
    b = np.ascontiguousarray(np.squeeze(net.params[k][1].data))
    return b, w

#dict stores weight and bias of chosen layers
param_dict = {}

#Storing the chosen weights anb bias in h5py file
with h5py.File('caffe_weight.h5','w') as f:
    for ix in xrange(1,13):
        b,w = extract_conv_params(net,keys[ix])
        param_dict[keys[ix] + '.W'] = w
        param_dict[keys[ix] + '.b'] = b
        f.create_dataset(keys[ix]+'.W', data=w, compression='gzip')
        f.create_dataset(keys[ix]+'.b', data=b, compression = 'gzip')
f.close
    
#verifying file
with h5py.File('caffe_weight.h5','r') as f:
    for ix in xrange(1,13):
        print f[keys[ix]+'.W'].value
        print f[keys[ix]+'.b'].value
f.close


