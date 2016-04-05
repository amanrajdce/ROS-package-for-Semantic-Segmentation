#
# @author  Aman Raj
# @Code adpated from Daniel Maturana's pydnn package
# @year    2015
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#

import sys
import numpy as np

from path import Path

import cv2
import theano
from theano import tensor as T
import lasagne
import lasagne.layers.dnn

import pydnn
import aman_sandbox

"""
Very Deep Covnets
"""
#Base Class having all the network parameters

class PseudoFcn8Base(object):

    def __init__(self, batch_size, num_classes):
        self.X = T.ftensor4('X')
        self.y1 = T.TensorType('int32', [False]*4)('y1')
        self.y2 = T.TensorType('int32', [False]*4)('y2')
        # we assume input has been cropped
        self.input_width = 320
        self.input_height    = 320
        self.channels = 6
        self.num_classes = num_classes
        self.batch_size = batch_size
        self._build()

    def get_output(self, X_batch):
        return self.l_out.get_output(X_batch)

    def get_deterministic_output(self, X_batch):
        return self.l_out.get_output(X_batch, deterministic=True)

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_out)

    def get_loss(self, output, y_batch):
        loss = pydnn.expr2d.masked_logloss_2d( output, T.cast(y_batch, 'int32') )
        return loss

    def get_multi_loss(self, out1, out2, y1_batch, y2_batch):
        # TODO needs downsample for y
        #loss1 = pydnn.expr2d.logloss_2d( out1, T.cast(y1_batch, 'int32') )
        #loss2 = pydnn.expr2d.logloss_2d( out2, T.cast(y2_batch, 'int32') )
        loss1 = pydnn.expr2d.masked_logloss_2d( out1, T.cast(y1_batch, 'int32') )
        loss2 = pydnn.expr2d.masked_logloss_2d( out2, T.cast(y2_batch, 'int32') )
        return loss1+loss2

    def get_prediction_image(self, dout):
        #dout = self.get_deterministic_output(X_batch)
        return pydnn.expr2d.predict_2d( dout )

    def get_error_rate(self, output, y_batch):
        err_rate = pydnn.expr2d.masked_error_rate_2d( output, T.cast(y_batch, 'int32') )
        return err_rate

    def get_param_values(self):
        return lasagne.layers.get_all_param_values(self.l_out)

    def get_l2_norm(self):
        return lasagne.regularization.regularize_layer_params(self.l_out,
                lasagne.regularization.l2)

    def set_param_values(self, values):
        lasagne.layers.set_all_param_values(self.l_out, values)

    def set_param_values_from_dict(self, param_dict):
        params = lasagne.layers.get_all_params(self.l_out)
        names = [p.name for p in params if p.name]
        # TODO this may not be an error
        if len(names) != len(set(names)):
            raise Exception('duplicate layer names')

        updated = []
        for param in params:
            if not (param.name is None) and param.name in param_dict:
                param.set_value(param_dict[param.name])
                updated.append(param.name)
        if not len(updated)==len(param_dict.keys()):
            raise Exception('not all values in param_dict used')

#VGG-net with changes RGB+D channel use
class PseudoFcn_RGBD(PseudoFcn8Base):

    def _build(self):
        self.l_input = lasagne.layers.InputLayer((self.batch_size, self.channels,self.input_height, self.input_width))
        #self.rgb_slice = lasagne.layers.SliceLayer(self.l_input, indices=slice(0,3), axis=1, name='rgb_slice')
	
	#Starts the RGB slice process
        self.conv1_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_input,64,filter_size=(3,3),stride=1,pad=100, name='conv1_1', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv1_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv1_1, 64, filter_size=(3,3), stride=1, pad=1, name='conv1_2', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool1= lasagne.layers.dnn.Pool2DDNNLayer(self.conv1_2,pool_size=2,stride=2, mode='max', name='pool1')

        self.conv2_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool1,128,filter_size=(3,3), pad=1, name='conv2_1', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv2_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv2_1,128,filter_size=(3,3), pad=1, name='conv2_2', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool2= lasagne.layers.dnn.Pool2DDNNLayer(self.conv2_2, pool_size=2, stride=2, mode='max',name='pool2')

        self.conv3_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool2,256,filter_size=(3,3),pad=1,name='conv3_1', stride=1, nonlinearity=lasagne.nonlinearities.rectify)
        self.conv3_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_1, 256,filter_size=(3,3), pad=1, stride=1, name='conv3_2', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv3_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_2, 256, filter_size=(3,3), pad=1, stride=1, name='conv3_3', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool3 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv3_3, pool_size=2, stride=2,mode='max',name='pool3')

        self.conv4_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool3, 512,filter_size=(3,3), pad=1, stride=1, name='conv4_1', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv4_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_1, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv4_2', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv4_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_2, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv4_3', nonlinearity = lasagne.nonlinearities.rectify)

        self.pool4 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv4_3, pool_size=2, stride=2, mode='max', name = 'pool4')

        self.conv5_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool4, 512, filter_size=(3,3), pad=1, stride=1,  name = 'conv5_1', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv5_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_1, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv5_2', nonlinearity = lasagne.nonlinearities.rectify)
	self.conv5_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_2, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv5_3', nonlinearity = lasagne.nonlinearities.rectify)

        self.pool5 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv5_3, pool_size=2, stride=2, mode='max', name = 'pool5')
	self.score =  lasagne.layers.dnn.Conv2DDNNLayer(self.pool5, 64, filter_size = (3,3), pad=1, stride=1, name='score')
	'''
	#Starts the HHA slice process
	self.hha_slice = lasagne.layers.SliceLayer(self.l_input, indices = slice(3,6), axis=1, name='hha_slice')
        self.conv1_1hha = lasagne.layers.dnn.Conv2DDNNLayer(self.hha_slice,64,filter_size=(3,3),stride=1,pad=50, name='conv1_1_hha', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv1_2hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv1_1hha, 64, filter_size=(3,3), stride=1, pad=1, name='conv1_2_hha', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool1hha= lasagne.layers.dnn.Pool2DDNNLayer(self.conv1_2hha,pool_size=2,stride=2, mode='max', name='pool1_hha')

        self.conv2_1hha = lasagne.layers.dnn.Conv2DDNNLayer(self.pool1hha,128,filter_size=(3,3), pad=1, name='conv2_1_hha', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv2_2hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv2_1hha,128,filter_size=(3,3), pad=1, name='conv2_2_hha', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool2hha= lasagne.layers.dnn.Pool2DDNNLayer(self.conv2_2hha, pool_size=2, stride=2, mode='max',name='pool2_hha')

        self.conv3_1hha = lasagne.layers.dnn.Conv2DDNNLayer(self.pool2hha,256,filter_size=(3,3),pad=1,name='conv3_1_hha', stride=1, nonlinearity=lasagne.nonlinearities.rectify)
        self.conv3_2hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_1hha, 256,filter_size=(3,3), pad=1, stride=1, name='conv3_2_hha', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv3_3hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_2hha, 256, filter_size=(3,3), pad=1, stride=1, name='conv3_3_hha', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool3hha = lasagne.layers.dnn.Pool2DDNNLayer(self.conv3_3hha, pool_size=2, stride=2,mode='max',name='pool3_hha')

        self.conv4_1hha = lasagne.layers.dnn.Conv2DDNNLayer(self.pool3hha, 512,filter_size=(3,3), pad=1, stride=1, name='conv4_1_hha', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv4_2hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_1hha, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv4_2_hha', 
								nonlinearity = lasagne.nonlinearities.rectify)
        self.conv4_3hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_2hha, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv4_3_hha', 
								nonlinearity = lasagne.nonlinearities.rectify)

        self.pool4hha = lasagne.layers.dnn.Pool2DDNNLayer(self.conv4_3hha, pool_size=2, stride=2, mode='max', name = 'pool4_hha')

        self.conv5_1hha = lasagne.layers.dnn.Conv2DDNNLayer(self.pool4hha, 512, filter_size=(3,3), pad=1, stride=1,  name = 'conv5_1_hha', 
								nonlinearity = lasagne.nonlinearities.rectify)
        self.conv5_2hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_1hha, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv5_2_hha', 
								nonlinearity = lasagne.nonlinearities.rectify)
        self.conv5_3hha = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_2hha, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv5_3_hha', 
								nonlinearity = lasagne.nonlinearities.rectify)

        self.pool5hha = lasagne.layers.dnn.Pool2DDNNLayer(self.conv5_3hha, pool_size=2, stride=2, mode='max', name = 'pool5_hha')
	self.score_hha =  lasagne.layers.dnn.Conv2DDNNLayer(self.pool5hha, 64, filter_size = (1,1), pad=1, stride=1, name='score_hha')
	'''
        #Fusing the outputs of both processes
	#self.elem_sum = lasagne.layers.ElemwiseSumLayer([self.score_rgb, self.score_hha], name='fuse_rgb_hha')    
       
	self.Up1 = pydnn.layers.Upsample2dLayer(self.score, times=2, name='Up1')
	#self.deconv = aman_sandbox.layers.DeConv2DDNNLayer(self.Up1, 64, (3,3), pad=1, stride=1, name='deconv', nonlinearity = lasagne.nonlinearities.identity)
	
	#scale-1
        self.l_conv1 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_input, 96, filter_size=(3,3), pad=10, stride=3, name='l_conv1',nonlinearity= lasagne.nonlinearities.rectify)
        self.l_conv_pool1 = lasagne.layers.dnn.Pool2DDNNLayer(self.l_conv1, pool_size = 2, stride = 2, mode='max', name = 'l_conv_pool1')
        self.l_conv_poolpad = lasagne.layers.PadLayer(self.l_conv_pool1, 4, name='l_conv_poolPad')
	
	#Concate input with scale-1 output        
	self.l_concat1 = lasagne.layers.ConcatLayer( [self.l_conv_poolpad, self.Up1], axis=1, name='l_concat1')
        
	self.conv6_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_concat1, 64 , filter_size=(3,3), pad=1, stride=1, name ='conv6_1', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv6_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv6_1, 64, filter_size=(3,3), pad=1, stride=1, name='conv6_2', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv6_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv6_2, 64, filter_size=(3,3), pad=1, stride=1, name='conv6_3', nonlinearity = lasagne.nonlinearities.rectify)

	#self.conv6_4 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv6_3, 4, filter_size=(3,3), pad=1, stride=1, name='conv6_4', nonlinearity = lasagne.nonlinearities.rectify)
	'''        
	self.Up2 = pydnn.layers.Upsample2dLayer(self.conv6_4, times=1, name='Up2')
	#Scale-2
	self.l_conv2 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_input, 96, filter_size=(3,3), pad=8, stride=3, name='l_conv2',nonlinearity= lasagne.nonlinearities.rectify)
        #self.l_conv_pool2 = lasagne.layers.dnn.Pool2DDNNLayer(self.l_conv2, pool_size = 2, stride = 2, mode='max', name = 'l_conv_pool2')
	
	#Concate input with scale-2 output
	self.l_concat2 = lasagne.layers.ConcatLayer([self.Up2,self.l_conv2], name='l_concat2')
	self.conv7_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_concat2, 64 , filter_size=(3,3), pad=1, stride=1, name ='conv7_1', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv7_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv7_1, 64, filter_size=(3,3), pad=1, stride=1, name='conv7_2', nonlinearity = lasagne.nonlinearities.rectify)
	self.conv7_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv7_2, 4, filter_size=(3,3), pad=1, stride=1, name='conv7_3', nonlinearity = lasagne.nonlinearities.rectify)
	'''	
	self.conv_out_nin = lasagne.layers.NINLayer(self.conv6_3, self.num_classes, nonlinearity=None, name='conv_out_nin')
        self.l_out = self.conv_out_nin

# VGG16 net as proposed in paper
class PseudoFcn_RGBD_VGG(PseudoFcn8Base):

    def _build(self):
        self.l_input = lasagne.layers.InputLayer((self.batch_size, self.channels,self.input_height, self.input_width))
        	
	#Starts the RGB slice process
        self.conv1_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_input,64,filter_size=(3,3),stride=1,pad=100, name='conv1_1', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv1_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv1_1, 64, filter_size=(3,3), stride=1, pad=1, name='conv1_2', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool1= lasagne.layers.dnn.Pool2DDNNLayer(self.conv1_2,pool_size=2,stride=2, mode='max', name='pool1')

        self.conv2_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool1,128,filter_size=(3,3), pad=1, name='conv2_1', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv2_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv2_1,128,filter_size=(3,3), pad=1, name='conv2_2', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool2= lasagne.layers.dnn.Pool2DDNNLayer(self.conv2_2, pool_size=2, stride=2, mode='max',name='pool2')

        self.conv3_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool2,256,filter_size=(3,3),pad=1,name='conv3_1', stride=1, nonlinearity=lasagne.nonlinearities.rectify)
        self.conv3_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_1, 256,filter_size=(3,3), pad=1, stride=1, name='conv3_2', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv3_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_2, 256, filter_size=(3,3), pad=1, stride=1, name='conv3_3', nonlinearity=lasagne.nonlinearities.rectify)

        self.pool3 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv3_3, pool_size=2, stride=2,mode='max',name='pool3')

        self.conv4_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool3, 512,filter_size=(3,3), pad=1, stride=1, name='conv4_1', nonlinearity=lasagne.nonlinearities.rectify)
        self.conv4_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_1, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv4_2', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv4_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_2, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv4_3', nonlinearity = lasagne.nonlinearities.rectify)

        self.pool4 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv4_3, pool_size=2, stride=2, mode='max', name = 'pool4')

        self.conv5_1 = lasagne.layers.dnn.Conv2DDNNLayer(self.pool4, 512, filter_size=(3,3), pad=1, stride=1,  name = 'conv5_1', nonlinearity = lasagne.nonlinearities.rectify)
        self.conv5_2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_1, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv5_2', nonlinearity = lasagne.nonlinearities.rectify)
	self.conv5_3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_2, 512, filter_size=(3,3), pad=1, stride=1, name = 'conv5_3', nonlinearity = lasagne.nonlinearities.rectify)
	#self.pool5 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv5_3, pool_size=2, stride=2, mode='max', name = 'pool5')
	#self.reshap = lasagne.layers.ReshapeLayer(self.pool5,((self.batch_size,))
	#self.score =  lasagne.layers.dnn.Conv2DDNNLayer(self.reshape, 64, filter_size = (3,3), pad=1, stride=1, name='score')
	self.conv_out_nin = lasagne.layers.NINLayer(self.conv5_3, self.num_classes, nonlinearity=None, name='conv_out_nin')
        self.l_out = self.conv_out_nin


#FCN_Experiment with RGB+D channel use

class PseudoFcn_RGBD_Exp(PseudoFcn8Base):

    def _build(self):
        self.l_input = lasagne.layers.InputLayer((self.batch_size, self.channels,
            self.input_height, self.input_width))

        # TODO
        #lrn = True

        ## Trying to replicate alexnet layers here
        # output input height width
        self.conv1 = lasagne.layers.dnn.Conv2DDNNLayer(self.l_input, 196, (7,7), stride=3, border_mode = 'same', name = 'conv1', nonlinearity = lasagne.nonlinearities.rectify)
        #self.pool1 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv1, pool_size = 3, stride = 2, pad=0, mode = 'max', name = 'pool1')
        self.conv1_drop = lasagne.layers.DropoutLayer(self.conv1, name='conv1drop')

        self.conv2 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv1_drop, 196, (5,5), stride=1, border_mode = 'same',name = 'conv2', nonlinearity = lasagne.nonlinearities.rectify)
        #self.pool2 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv2, pool_size = 3, stride = 2 , pad=2, mode = 'max', name = 'pool2')
        self.conv2_drop = lasagne.layers.DropoutLayer(self.conv2, name='conv2drop')

        self.conv3 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv2_drop, 384, (3,3), stride=1, border_mode = 'same',name = 'conv3', nonlinearity = lasagne.nonlinearities.rectify)
        #self.pool3 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv3, pool_size =1, stride = 1, pad=1,mode = 'max', name = 'pool3')
        self.conv3_drop =  lasagne.layers.DropoutLayer(self.conv3, name='conv3drop')

        self.conv4 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv3_drop, 384 , (3,3), stride=1, border_mode = 'same', name = 'conv4', nonlinearity = lasagne.nonlinearities.rectify)
        #self.pool4 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv4, pool_size = 1, stride =1 , pad=1, mode ='max', name ='pool4')
        self.conv4_drop = lasagne.layers.DropoutLayer(self.conv4, name='conv4drop')

        self.conv5 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv4_drop, 256, (3,3), stride=1, border_mode = 'same', name = 'conv5', nonlinearity= lasagne.nonlinearities.rectify)
        #self.pool5 = lasagne.layers.dnn.Pool2DDNNLayer(self.conv5, pool_size =3, stride=2, pad=1, name = 'pool5')
        self.conv5_drop = lasagne.layers.DropoutLayer(self.conv5, name='conv5drop')

        self.conv6 = lasagne.layers.dnn.Conv2DDNNLayer(self.conv5_drop, 256, (7,7), stride=2, border_mode='same', name = 'conv6')
        self.conv6_nin =  lasagne.layers.NINLayer(self.conv6, 128, name='conv6_nin')

        #self.conv7 = lasagne.layers.NINLayer(self.conv6, 4096, name='conv7')
        #self.conv7drop= lasagne.layers.DropoutLayer(self.conv7, name='conv7drop')
        #self.conv7drop_nin = lasagne.layers.NINLayer(self.conv7drop, 512, name='conv7drop_nin')
        #self.conv7drop_nin_pad = lasagne.layers.PadLayer(self.conv7drop_nin, 5, name='conv7_pad')

        #self.pool4_nin =  lasagne.layers.NINLayer(self.pool4, 512, name='pool4_nin')
        #self.pool4_up = pydnn.layers.Upsample2dLayer(self.pool4_nin, times=1, name='pool4up')
        #self.pool4_up_crop= pydnn.layers.Crop2dLayer(self.pool4_up, (18,18), name='pool4up_crop')

        #self.fcn = lasagne.layers.ElemwiseSumLayer([self.pool4_up_crop, self.conv7drop_nin], name = 'fcn')
        #self.fcn_up =  pydnn.layers.Upsample2dLayer(self.fcn , times=1, name='fcn_up')
        #self.fcn_up_crop = pydnn.layers.Crop2dLayer(self.fcn_up,(28,28), name='fcn_up_crop')

        #self.pool1_nin = lasagne.layers.NINLayer(self.pool1, 512, name='pool1_nin')
        #self.fcn1 =  lasagne.layers.ElemwiseSumLayer([self.conv7drop_nin_pad, self.pool1_nin], name='fcn1')
        #self.fcn1_drop = lasagne.layers.DropoutLayer(self.fcn1, name='fcn1_drop')

        #self.conv4_up = pydnn.layers.Upsample2dLayer(self.conv4_nin_drop, times=1, name='conv4_up')
        #self.conv4_up2 = pydnn.layers.Crop2dLayer(lasagne.layers.PadLayer(self.conv4_up, 1), (27, 27), name='conv4_up2')
        #self.conv1_nin = lasagne.layers.NINLayer(self.pool1, 512, name='conv1_nin')
        #self.fcn = lasagne.layers.ElemwiseSumLayer([self.conv1_nin, self.conv4_up2], name='fcn')
        #self.fcn_drop = lasagne.layers.DropoutLayer(self.conv4_up2)
        self.conv_out_nin = lasagne.layers.NINLayer(self.conv6_nin, self.num_classes, nonlinearity=None, name='conv_out_nin')
        self.l_out = self.conv_out_nin
	#self.l_nin2 = lasagne.layers.NINLayer(self.l_drop2_2, self.num_classes)
        #self.l_out2 = pydnn.layers.Softmax2dLayer(self.l_nin2)
        #self.l_out = self.fcn32


