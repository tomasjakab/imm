"""
Code for colorization network adapted from
Colorization as a Proxy Task for Visual Understanding, Larsson, Maire, Shakhnarovich, CVPR 2017
https://github.com/gustavla/self-supervision
"""

import os
import tensorflow as tf
import deepdish as dd

from imm.models.selfsup import info
from imm.models.selfsup import vgg16

def build_vgg16(input, reuse=False, pretrained_file=None):
  with tf.variable_scope('vgg16', reuse=reuse):
    data = dd.io.load(pretrained_file, '/data')
    inf = info.create(scale_summary=True)
    testing = True

    input_raw = input
    # convert to grayscale
    input = tf.reduce_mean(input, 3, keep_dims=True)
    # normalize
    input = input / 255.0
    # centre
    input = input - 114.451 / 255.0
    net = vgg16.build_network(input, info=inf, parameters=data,
                             final_layer=False,
                             phase_test=testing,
                             pre_adjust_batch_norm=True,
                             use_dropout=True)

    # replace the input with the original input in RGB
    net['input'] = input_raw
    return net


if __name__ == '__main__':
  pretrained_file = '/users/tomj/minmaxinfo/data/models/vgg16.caffemodel.h5'
  input = tf.placeholder(tf.float32, [None, 128, 128, 1])
  net = build_vgg16(input, pretrained_file=pretrained_file)
