"""
Abstract model class.

Author: Ankush Gupta
Date: 26 Jan, 2018.
"""

import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod
from tensorflow.contrib.framework.python.ops import variables

from imm.tf_utils import nn_utils as nnu


class BaseModel(object):
  """A simple class for handling data sets."""
  __metaclass__ = ABCMeta
  num_instances = 0

  def __init__(self, dtype, name):
    self.dtype = dtype
    """Initialize dataset using a subset and the path to the data."""
    # assert subset in self.available_subsets(), self.available_subsets()
    self._name = name
    # operations for moving-"averaging" (for e.g. accuracy estimates):
    self._avg_ops = []
    # opts for conv layers:
    self._opts = None
    # keep a count of how many instances of this class have been instantiated:
    self.__class__.num_instances += 1

  def _decay(self,scope=None):
    """Aggregates the various L2 weight decay losses."""
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    sum_decay = tf.add_n(reg_loss)
    return sum_decay

  def _exp_running_avg(self, x, training_pl, init_val=0.0, rho=0.99, name='x'):
    x_avg = variables.model_variable(name+'_agg', shape=x.shape,
                                     dtype=x.dtype,
                                     initializer=tf.constant_initializer(init_val, x.dtype),
                                     trainable=False,device='/cpu:0')
    w_update = 1.0 - rho
    x_new = x_avg + w_update * (x - x_avg)
    update_op = tf.cond(training_pl,
                        lambda: tf.assign(x_avg, x_new),
                        lambda: tf.constant(0.0))
    with tf.control_dependencies([update_op]):
      return tf.identity(x_new)

  def _add_cost_summary(self, cost, name):
    """
    Adds moving average + raw cost summaries:
    """
    if self.__class__.num_instances == 1:
      cost_avg = tf.train.ExponentialMovingAverage(0.99, name=name+'_movavg', )
      self._avg_ops.append(cost_avg.apply([cost]))
      tf.summary.scalar(name+'_avg', cost_avg.average(cost), family='train')
      tf.summary.scalar(name+'_raw', cost, family='train')

  def _get_opts(self, training_pl):
    if self._opts is None:
      opts = {'dtype': self.dtype,
              'wd': 1e-5,
              'std': 0.01,
              'training_pl': training_pl}
      self._opts = opts
    return self._opts

  def get_bnorm_ops(self,scope=None):
    """
    Return any batch-normalization / other "moving-average" ops.
    ref: https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-236068575
    """
    updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope)
    # print updates
    return tf.group(*updates)

  def uncertainty_weighted_mtl(self, losses, name='uw_mtloss'):
    """
    Implements "uncertainty-weighted" multi-task loss [Kendall et al., 2017].
    Loss-total = Sum_i  1/s_i^2 * loss_i + log(s_i)
    """
    uw_losses =[]
    with tf.variable_scope(name,default_name='uw_mtloss') as sc:
      for i, loss in enumerate(losses):
        i_log_s = variables.model_variable('loss%d'%i, shape=(1,),
                    dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                    device='/cpu:0')
        s = tf.exp(-i_log_s[0])
        i_loss = s * loss + i_log_s[0]
        uw_losses.append(i_loss)
    return tf.add_n(uw_losses, name='uwmt_loss')

  def conv_block(self, opts, x, filter_hw, out_channels,
                 stride=(1,1,1,1), padding='SAME', add_bias=True,
                 batch_norm=True, layer_norm=False, preactivation=None,
                 activation=tf.nn.relu,
                 name='cblock', var_device='/cpu:0'):
    """
    Convenience function which figures out the shape of the filters.
    """
    if layer_norm and batch_norm:
      raise ValueError('Both layer and batch norm cannot be applied.')

    with tf.variable_scope(name,default_name='mconv') as sc:
      f_h, f_w = filter_hw
      in_channels = x.get_shape().as_list()[-1] # number of input channels
      f_shape = [f_h,f_w,in_channels,out_channels] # shape of the filters of the first conv-layer
      y,_ = nnu.conv_block(opts, x, f_shape, stride, padding,
                           add_bias=add_bias, batch_norm=batch_norm,
                           layer_norm=layer_norm,
                           preactivation=preactivation,
                           activation=activation,
                           conv_scope=name, device=var_device)
    return y

  @abstractmethod
  def build(self, inputs, training_pl):
    """This is the method called by the model factory."""
    pass
