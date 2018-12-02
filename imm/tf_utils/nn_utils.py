"""
nn_utils.py
  Utility functions for defining neural-networks.

@Author: Ankush Gupta
@Date: 16 August 2016
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.contrib.layers import batch_norm as batch_norm_layer
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages



def _variable_with_weight_decay(name, shape, stddev, wd, dtype, device):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    dtype: tf datatypes (e.g.: tf.float16, tf.float32)
    device: device for the place of the VARIABLES (not the OPS).
    collection: [optional] string, name of the collection to which
                the variable should be added.

  Returns:
    Variable Tensor
  """
  regularizer = None
  if wd is not None:
    regularizer = l2_regularizer(wd)
  init = tf.truncated_normal_initializer(stddev=stddev,dtype=dtype)
  # init = tf.random_uniform_initializer(minval=-1.0,maxval=1.0,dtype=dtype)
  return variables.model_variable(name, shape=shape, dtype=dtype,
                                  initializer=init, regularizer=regularizer,
                                  device=device)

def my_batch_norm(x,is_train,dtype=tf.float32,reuse=False,scope=None,device=None):
  """
  My batch normalization.
  Adds the update ops to tf.GraphKets.UPDATE_OPS collection.
    --> Collect the ops there and run them during training.
  """
  with tf.variable_scope(scope,default_name='BNorm',values=[x],reuse=reuse) as sc:
    params_shape = [x.get_shape()[-1]]
    beta = variables.model_variable('beta', shape=params_shape, dtype=dtype,
                                     initializer=tf.constant_initializer(0.0, dtype),device=device)
    gamma = variables.model_variable('gamma', shape=params_shape, dtype=dtype,
                                     initializer=tf.constant_initializer(1.0, dtype),device=device)
    if is_train:
      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
      moving_mean = variables.model_variable('moving_mean', shape=params_shape, dtype=dtype,
                                              initializer=tf.constant_initializer(0.0, dtype),
                                              trainable=False,device=device)
      moving_variance = variables.model_variable('moving_variance', shape=params_shape, dtype=dtype,
                                                  initializer=tf.constant_initializer(1.0, dtype),
                                                  trainable=False,device=device)
      mu_update_op = moving_averages.assign_moving_average(moving_mean,mean,0.99)
      var_update_op = moving_averages.assign_moving_average(moving_variance,variance,0.99)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,mu_update_op)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,var_update_op)
    else:
      mean = variables.model_variable('moving_mean', shape=params_shape, dtype=dtype,
                                      initializer=tf.constant_initializer(0.0, dtype),trainable=False,device=device)
      variance = variables.model_variable('moving_variance', shape=params_shape, dtype=dtype,
                                          initializer=tf.constant_initializer(1.0, dtype),trainable=False,device=device)
    # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
    y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-4)
    y.set_shape(x.get_shape())
    return y

def _conv(x,shape,stride,padding,dilation_rate=None,w_name='w',b_name='b',
          std=0.01,wd=None,dtype=tf.float32,add_bias=True,device=None):
  """
  Define a Convolutional layer with (optional) bias term.
  For documentation, see `conv_block`.

  If DILATION_RATE is specified, ATROU-conv is used.
    In this case, the STRIDE parameter is ignored, as the
    stride is set to one.
  """
  w = _variable_with_weight_decay(w_name,shape=shape,stddev=std,
                                  wd=wd,dtype=dtype,device=device)
  if dilation_rate is None:
    out = tf.nn.conv2d(x,w,strides=stride,padding=padding)
  else:
    out = tf.nn.atrous_conv2d(x,w,dilation_rate,padding=padding)
  # [optional] bias:
  if add_bias:
    b = variables.model_variable(b_name,shape=shape[-1:],dtype=dtype,
                                initializer=tf.constant_initializer(0.0),
                                device=device)
    out = tf.nn.bias_add(out,b)
  return out


def fc_layer(opts,x,out_dim,layer_name,
              w_name='w',b_name='b',
              scope=None, reuse=False,
              dtype=tf.float32,std=0.01,wd=None,batch_norm=False,
              dropout_keeprate=None,add_bias=False,device=None):
  """
  Implements fully-connected layer using convolutions with optional bias
  and optional dropout.
  For an input tensor of size: [B,H,W,C], the output size is: [B,1,1,OUT_DIM]

  Args:
    x: (tensor) input to this layer
    out_dim: (integer) the output dimension (output number of channels)
    {w,b}_name: names of the filters and bias
    dtype: (datatype; default = tf.float32) tensorflow datatype
    std: (float) std for initializing the weight matrix
    wd: (float) weight-decay for the weight matrix
    dropout_keeprate: (float) rate with which the units are ON
    add_bias: (bool) if we want to add a bias
    device: device for the place of the VARIABLES (not the OPS).

  Returns:
    The tensor output.
  """
  x_shape = x.get_shape().as_list()
  # get the shape of the filters of the convolutional layers:
  f_shape = x_shape[1:] + [out_dim]
  with tf.variable_scope(layer_name,default_name='FCLayer',values=[x],reuse=reuse) as sc:
    # convolution operation (for the fully-connected op):
    y = _conv(x,f_shape,[1,1,1,1],'VALID',1,w_name,b_name,
              std,wd,dtype,add_bias,device)
    # [optional] dropout:
    if dropout_keeprate is not None:
      y = tf.nn.dropout(y, dropout_keeprate)
    # [optional batch-norm]:
    if batch_norm:
      y = batch_norm_layer(y,decay=0.9,reuse=False,is_training=opts['train_switch'])
  return y

def conv_block(opts,x,shape,stride,padding,dilation_rate=None,
               w_name='w',b_name='b',conv_scope=None,
               share_conv=False,batch_norm=False,layer_norm=False,
               activation=tf.nn.relu,
               preactivation=None,
               add_bias=True,
               device=None):
  """
  Returns a conv-batchNorm-relu block.

  If DILATION_RATE is not None, then DILATED-CONV is performed.
  In this case, the STRIDE parameter is ignored, as the stride
  is set to one.

  Args:
    opts: dictionary of options:
        dtype: data-type of the filters, e.g. tf.float16, tf.float32
        wd: (float) weight-decay multiplier (or None for no weight-decay)
        std: (float) standard-dev of weights initialization
        is_training: (Python boolean) same truth-value as train_switch

    x : input variable/ placeholder
    shape: 4-tuple of the filter-sizes [H,W,IN,OUT]
    stride: (4-tuple) stride off the conv [batch, height, width, channels]
    padding: (string) one of ['SAME', 'VALID']
    {w,b}_name: names of {weights,bias} to be used in this conv-block.
    conv_scope: (string) [optional] name of the scope for the conv layer
    share_conv: (boolean) Whether to re-use variables in the conv-scope
    batch_norm: (optional) add batch-normalization between conv and relu [default: True]
    add_bias: (optional) add a bias to conv [default: True]
    device: device for the place of the VARIABLES (not the OPS).

  Output:
    tf.Tensor: relu(batch-norm(conv(x))) (or without batch-norm)
  """
  if layer_norm and batch_norm:
    raise ValueError('Both layer and batch norm cannot be applied.')

  if preactivation is not None:
    raise ValueError('preactivation option is deprecated.')

  # conv op with optional scope:
  with tf.variable_scope(conv_scope,default_name='ConvBlock',values=[x],reuse=share_conv) as sc:
    out_c = _conv(x,shape,stride,padding,dilation_rate,w_name,b_name,
                opts['std'],opts['wd'],opts['dtype'],add_bias,device)
    # [optional] batch-normalization:
  out = out_c
  if batch_norm:
    #out = batch_norm_layer(out,decay=0.9,reuse=False,is_training=opts['train_switch'])
    # NOTE: specify device?
    out_b = tf.layers.batch_normalization(out_c, training=opts['training_pl'],
                                          fused=True)
    out = out_b
  if layer_norm:
    out_b = tf.contrib.layers.layer_norm(out_c, variables_collections=tf.GraphKeys.MODEL_VARIABLES)
    out = out_b
  # relu:
  if activation is not None:
    out = activation(out)
  return out,out_c