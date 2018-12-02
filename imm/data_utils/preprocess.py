"""
Data pre-processing methods.

Author: Ankush Gupta
Date: 23 March, 2017.
"""
import tensorflow as tf
import numpy as np
import scipy.ndimage as scim


def gaussian_kernel(sz,sigma,dtype=np.float32):
  """
  SZ: Integer (odd) -- size of the Gaussian window.
  sigma: [max-value=0.5], actual sigma = sigma * SZ//2.

  Returns a gaussian kernel of SZxSZ.
  """
  sz = int(sz)
  if sz%2 != 1:
    raise ValueError('Gaussian kernel size should be odd, got: %d.'%sz)
  # if sigma <= 0 or sigma > 0.5:
  #   raise ValueError('Sigma not in (0,0.5] range: %.2f'%sigma)
  im = np.zeros((sz,sz),dtype=dtype)
  im[sz//2,sz//2] = 1.0
  sigma = sz//2 * sigma
  g = scim.filters.gaussian_filter(im,sigma=sigma)
  return g


def global_contrast_norm(x,eps=1.0):
  """
  Given a 4D tensor,
  performs per-channel whitening.

  X: [B,H,W,C] tensor.
  """
  x = tf.convert_to_tensor(x)
  ndims = x.get_shape().ndims
  assert ndims==4, 'Unknown shape.'
  # get the mean and variance:
  mu,v = tf.nn.moments(x,[1,2],keep_dims=True)
  inv_std = tf.rsqrt(tf.maximum(v,eps**2))
  x_c = tf.multiply(tf.subtract(x,mu),inv_std)
  return x_c


def local_contrast_norm(x,sz=21,eps=1.0):
  """
  Local contrast normalization, as per LeCun:
    http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf

  X : [B,H,W,C] tensor, which is contrast normalized.
  SZ: integer, size of the neighbourhoood for pooling statistics.
      must be odd.

  Reflection padding at the edges.
  """
  sz = int(sz)
  if sz%2 != 1:
    raise ValueError('Neighborhood size must be odd, got: %d'%sz)

  x = tf.convert_to_tensor(x)
  with tf.name_scope("lcn", [x]) as name:
    # reflection padding at the edges:
    padding = np.zeros((4,2),dtype=np.int32)
    padding[1:3,:] = sz//2
    x_pad = tf.pad(x,padding,mode='REFLECT',name='pad_mu')
    # get a gaussian kernel for weighting:
    w = gaussian_kernel(sz,sigma=0.7)
    w = np.reshape(w,[sz,sz,1,1])
    w = tf.tile(w,[1,1,3,1])
    # get the mean and standard dev "images":
    mu = tf.nn.depthwise_conv2d(x_pad,w,[1,1,1,1],padding='VALID')
    x_c = x - mu # mean-centering
    x_c_pad = tf.pad(x_c,padding,mode='REFLECT',name='pad_std')
    std = tf.nn.depthwise_conv2d(tf.square(x_c_pad),w,[1,1,1,1],padding='VALID')
    std = tf.sqrt(tf.maximum(eps**2,std))
    mu_std = tf.reduce_mean(std,axis=[1,2],keep_dims=True)
    std = tf.maximum(mu_std,std)
    x = tf.div(x_c, std)
  return x
