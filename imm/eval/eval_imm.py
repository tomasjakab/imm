# ==========================================================
# Author: Tomas Jakab, Ankush Gupta
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import time
from datetime import datetime

import os

import metayaml

from imm.utils.box import Box
from imm.train.cnn_train_multi import get_test_summaries

from tensorflow.contrib.framework.python.ops import variables

from imm.utils.colorize import *



def evaluate(dataset_instance, net, net_config, net_file, training_opts,
             batch_size=100, random_seed=0, eval_tensors=None,
             eval_loss=False, eval_summaries=False, eval_metrics=False):
  np.random.seed(random_seed)

  with tf.Graph().as_default() as graph:
    test_dataset = dataset_instance.get_dataset(batch_size, repeat=False,
                                                shuffle=False,
                                                num_preprocess_threads=12)

    global_step = variables.model_variable('global_step', shape=[],
                                           initializer=tf.constant_initializer(
        0),
        trainable=False)
    training_pl = tf.placeholder(tf.bool)
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(
        handle_pl, test_dataset.output_types, test_dataset.output_shapes)
    inputs = base_iterator.get_next()

    net_instance = net(net_config)
    _, loss, _, tensors = net_instance.build(inputs, training_pl=training_pl,
                                             output_tensors=True,
                                             build_loss=eval_loss)

    tensors_col = tf.get_collection('tensors')
    tensors_col = {k: v for k, v in tensors_col}
    tensors.update(tensors_col)
    if eval_tensors is not None:
      tensors_ = {x: tensors[x] for x in eval_tensors}
      tensors = tensors_
    tensors_names, tensors_ops = [list(x) for x in zip(*tensors.items())]

    test_summary_op = tf.summary.merge(
        get_test_summaries(tf.contrib.framework.get_name_scope()))

    test_iterator = test_dataset.make_initializable_iterator()

    # start a new session:
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = training_opts.allow_growth
    session = tf.Session(config=session_config)

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    session.run([global_init, local_init])

    test_handle = session.run(test_iterator.string_handle())

    summary_logdir = training_opts.logdir + '_test'
    summary_writer = tf.summary.FileWriter(summary_logdir, graph=session.graph)

    net_file = os.path.join(training_opts.logdir, net_file)

    # restore checkpoint:
    if tf.gfile.Exists(net_file) or tf.gfile.Exists(net_file + '.index'):
      print('RESTORING MODEL from: ' + net_file)
      checkpoint_fname = net_file
      reader = tf.train.NewCheckpointReader(checkpoint_fname)
      vars_to_restore = tf.global_variables()
      checkpoint_vars = reader.get_variable_to_shape_map().keys()
      vars_ignored = [
          v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
      print(colorize('vars-IGNORED (not restoring):', 'blue', bold=True))
      print(colorize(', '.join(vars_ignored), 'blue'))
      vars_to_restore = [
          v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
      restorer = tf.train.Saver(var_list=vars_to_restore)
      restorer.restore(session, checkpoint_fname)
    else:
      raise Exception('model file does not exist at: ' + net_file)

    step = session.run(global_step)
    feed_dict = {handle_pl: test_handle, training_pl: False}
    metrics_reset_ops = tf.get_collection('metrics_reset')
    metrics_update_ops = tf.get_collection('metrics_update')
    session.run(metrics_reset_ops)
    session.run(test_iterator.initializer)
    test_iter = 0
    tensors_results = {k: [] for k in tensors_names}
    ops_to_run = {'tensors': tensors_ops}
    if eval_loss:
      ops_to_run['loss'] = loss
    if eval_metrics:
      ops_to_run['metrics'] = metrics_update_ops
    while True:
      try:
        start_time = time.time()
        if test_iter == 0 and eval_summaries:
          results, summary_str = session.run([ops_to_run, test_summary_op],
                                             feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
        else:
          results = session.run(ops_to_run, feed_dict=feed_dict)
        duration = time.time() - start_time

        tensors_values = results['tensors']
        loss_value = results['loss'] if eval_loss else 0
        for name, value in zip(tensors_names, tensors_values):
          tensors_results[name].append(value)

        examples_per_sec = batch_size / float(duration)
        format_str = 'test: %s: step %d, loss = %.4f (%.1f examples/sec) %.3f sec/batch'
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, duration))
      except tf.errors.OutOfRangeError:
        print('iteration through test set finished')
        break
      test_iter += 1

    metrics_summaries_ops = tf.get_collection('metrics_summaries')
    if metrics_summaries_ops:
      summary_str = session.run(tf.summary.merge(metrics_summaries_ops))
      summary_writer.add_summary(summary_str, step)

    summary_writer.flush()  # write to disk now

    return tensors_results


def load_configs(file_names):
  """
  Loads the yaml config files.
  """
  config = Box(metayaml.read(file_names))
  return config