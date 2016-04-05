#
# @author  Daniel Maturana
# @year    2015
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#
# @=

from lasagne.layers import Layer
from theano.sandbox.cuda import dnn

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers.conv import conv_output_length
from lasagne.utils import as_tuple


__all__ = [\
        'DeConv2DDNNLayer',
        ]


def upsample_filt(filter_shape):
    # output channels, input channels, filter width, filter height    
    factor = (filter_shape[2] + 1) // 2
    if filter_shape[2] % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_shape[2], :filter_shape[3]]
    twoD_filter=(1 - abs(og[0] - center) / factor) * \
                (1 - abs(og[1] - center) / factor)
    fourD_filters=np.zeros(filter_shape)
    for n in xrange(filter_shape[0]):
        fourD_filters[n,n]=twoD_filter

    return fourD_filters

class DeConv2DDNNLayer(Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 border_mode=None, untie_biases=False, W=upsample_filt([64,64,3,3]).astype(np.float32),
                 pad=None, nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, **kwargs):
        super(DeConv2DDNNLayer, self).__init__(incoming, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
            
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and "
                               "'pad'. To avoid ambiguity, please specify "
                               "only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0)
            self.border_mode = 'valid'
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0)
                self.border_mode = 'valid'
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] - 1)
                self.border_mode = 'full'
            elif border_mode == 'same':
                # dnn_conv does not support same, so we just specify
                # padding directly.
                # only works for odd filter size, but the even filter size
                # case is probably not worth supporting.
                self.pad = ((self.filter_size[0] - 1) // 2,
                            (self.filter_size[1] - 1) // 2)
                self.border_mode = None
            else:
                raise RuntimeError("Invalid border mode: '%s'" % border_mode)
        else:
            self.pad = as_tuple(pad, 2)
            self.border_mode = None

        #self.W = self.add_param(W, self.get_W_shape(), name="W")
        #W = upsample_filt([num_filters,num_filters,filter_size[0],filter_size[0]]).astype(np.float32)
        self.W = self.add_param(W, self.get_W_shape(), name='W', trainable=False, regularizable=False)

	'''
    def upsample_filt(filter_shape):
        # output channels, input channels, filter width, filter height    
        factor = (filter_shape[2] + 1) // 2
        if filter_shape[2] % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:filter_shape[2], :filter_shape[3]]
        twoD_filter=(1 - abs(og[0] - center) / factor) * \
                    (1 - abs(og[1] - center) / factor)
        fourD_filters=np.zeros(filter_shape)
        for n in xrange(filter_shape[0]):
            fourD_filters[n,n]=twoD_filter

        return fourD_filters
	'''
    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         'pad', self.pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            'pad', self.pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        # if 'border_mode' is one of 'valid' or 'full' use that.
        # else use pad directly.
        border_mode = (self.border_mode if (self.border_mode is not None)
                       else self.pad)

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode
                              )
        activation = conved
        return self.nonlinearity(activation)

