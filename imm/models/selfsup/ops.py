"""
Code for colorization network adapted from
Colorization as a Proxy Task for Visual Understanding, Larsson, Maire, Shakhnarovich, CVPR 2017
https://github.com/gustavla/self-supervision
"""

import tensorflow as tf
import numpy as np
from .util import DummyDict

from tensorflow.python.framework import ops as tfops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


def max_pool(x, size, stride=None, name=None, info=DummyDict(), padding='SAME'):
    if stride is None:
        stride = size

    z = tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding,
                          name=name)

    info['activations'][name] = z
    return z


def avg_pool(x, size, stride=None, name=None, info=DummyDict(), padding='SAME'):
    if stride is None:
        stride = size

    z = tf.nn.avg_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding,
                          name=name)

    info['activations'][name] = z
    return z


def dropout(x, drop_prob, phase_test=None, name=None, info=DummyDict()):
    assert phase_test is not None
    with tf.name_scope(name):
        keep_prob = tf.cond(phase_test,
                            lambda: tf.constant(1.0),
                            lambda: tf.constant(1.0 - drop_prob))

        z = tf.nn.dropout(x, keep_prob, name=name)
    info['activations'][name] = z
    return z


def scale(x, name=None, value=1.0):
    s = tf.get_variable(name, [], dtype=tf.float32,
                        initializer=tf.constant_initializer(value))
    return x * s


def inner(x, channels, info=DummyDict(), stddev=None,
          activation=tf.nn.relu, name=None):
    with tf.name_scope(name):
        f = channels
        features = np.prod(x.get_shape().as_list()[1:])
        xflat = tf.reshape(x, [-1, features])
        shape = [features, channels]

        if stddev is None:
            W_init = tf.contrib.layers.variance_scaling_initializer()
        else:
            W_init = tf.random_normal_initializer(0.0, stddev)
        b_init = tf.constant_initializer(0.0)

        with tf.variable_scope(name):
            W = tf.get_variable('weights', shape, dtype=tf.float32,
                                initializer=W_init)
            b = tf.get_variable('biases', [f], dtype=tf.float32,
                                initializer=b_init)

        z = tf.nn.bias_add(tf.matmul(xflat, W), b)

    if activation is not None:
        z = activation(z)

    if info.get('scale_summary'):
        with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    info['activations'][name] = z
    if 'weights' in info:
        info['weights'][name + ':weights'] = W
        info['weights'][name + ':biases'] = b
    return z


def atrous_avg_pool(value, size, rate, padding, name=None, info=DummyDict()):
    with tfops.op_scope([value], name, "atrous_avg_pool") as name:
        value = tfops.convert_to_tensor(value, name="value")
        if rate < 1:
            raise ValueError("rate {} cannot be less than one".format(rate))

        if rate == 1:
            value = nn_ops.avg_pool(value=value,
                                                                strides=[1, 1, 1, 1],
                                                                ksize=[1, size, size, 1],
                                                                padding=padding)
            return value

        # We have two padding contributions. The first is used for converting "SAME"
        # to "VALID". The second is required so that the height and width of the
        # zero-padded value tensor are multiples of rate.

        # Padding required to reduce to "VALID" convolution
        if padding == "SAME":
            filter_height, filter_width = size, size

            # Spatial dimensions of the filters and the upsampled filters in which we
            # introduce (rate - 1) zeros between consecutive filter values.
            filter_height_up = filter_height + (filter_height - 1) * (rate - 1)
            filter_width_up = filter_width + (filter_width - 1) * (rate - 1)

            pad_height = filter_height_up - 1
            pad_width = filter_width_up - 1

            # When pad_height (pad_width) is odd, we pad more to bottom (right),
            # following the same convention as avg_pool().
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        elif padding == "VALID":
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0
        else:
            raise ValueError("Invalid padding")

        # Handle input whose shape is unknown during graph creation.
        if value.get_shape().is_fully_defined():
            value_shape = value.get_shape().as_list()
        else:
            value_shape = array_ops.shape(value)

        in_height = value_shape[1] + pad_top + pad_bottom
        in_width = value_shape[2] + pad_left + pad_right

        # More padding so that rate divides the height and width of the input.
        pad_bottom_extra = (rate - in_height % rate) % rate
        pad_right_extra = (rate - in_width % rate) % rate

        # The paddings argument to space_to_batch includes both padding components.
        space_to_batch_pad = [[pad_top, pad_bottom + pad_bottom_extra],
                                                    [pad_left, pad_right + pad_right_extra]]

        value = array_ops.space_to_batch(input=value,
                                                                         paddings=space_to_batch_pad,
                                                                         block_size=rate)

        value = nn_ops.avg_pool(value=value, ksize=[1, size, size, 1],
                                                            strides=[1, 1, 1, 1],
                                                            padding="VALID",
                                                            name=name)

        # The crops argument to batch_to_space is just the extra padding component.
        batch_to_space_crop = [[0, pad_bottom_extra], [0, pad_right_extra]]

        value = array_ops.batch_to_space(input=value,
                                                                         crops=batch_to_space_crop,
                                                                         block_size=rate)

    info['activations'][name] = value
    return value


def conv(x, channels, size=3, strides=1, activation=tf.nn.relu, name=None, padding='SAME',
         info=DummyDict(), output_shape=None):
    with tf.name_scope(name):
        features = x.get_shape().as_list()[3]
        f = channels
        shape = [size, size, features, f]

        W_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.0)

        W = tf.get_variable(name + '/weights', shape, dtype=tf.float32,
                            initializer=W_init)
        b = tf.get_variable(name + '/biases', [f], dtype=tf.float32,
                            initializer=b_init)
        z = tf.nn.conv2d(
                x,
                W,
                strides=[1, strides, strides, 1],
                padding=padding)

        z = tf.nn.bias_add(z, b)
        if activation is not None:
            z = activation(z)
        info['weights'][name + ':weights'] = W
        info['weights'][name + ':biases'] = b
        info['activations'][name] = z
        if output_shape is not None:
            assert list(output_shape) == list(z.get_shape().as_list())
        return z


def upconv(x, channels, size=3, strides=1, output_shape=None, activation=tf.nn.relu, name=None, padding='SAME',
         info=DummyDict()):
    with tf.name_scope(name):
        features = x.get_shape().as_list()[3]
        f = channels
        shape = [size, size, f, features]

        W_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.0)

        W = tf.get_variable(name + '/weights', shape, dtype=tf.float32,
                            initializer=W_init)
        b = tf.get_variable(name + '/biases', [f], dtype=tf.float32,
                            initializer=b_init)
        z = tf.nn.conv2d_transpose(
                x,
                W,
                output_shape=output_shape,
                strides=[1, strides, strides, 1],
                padding=padding)

        z = tf.nn.bias_add(z, b)
        if activation is not None:
            z = activation(z)
        info['weights'][name + ':weights'] = W
        info['weights'][name + ':biases'] = b
        info['activations'][name] = z
        return z


