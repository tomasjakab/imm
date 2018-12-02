"""
Train models using multiple GPU's with synchronous updates.
Adapted from inception_train.py

This is modular, i.e. it is not tied to any particular
model or dataset.

@Author: Ankush Gupta, Tomas Jakab
@Date: 25 Aug 2016
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path as osp
import time
import numpy as np
import tensorflow as tf

from ..utils.colorize import colorize
from ..utils import utils


def get_train_summaries(scope):
  summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
  return summaries


def get_test_summaries(scope):
  summaries = tf.get_collection('test_summaries', scope)
  return summaries


def tower_loss(inputs, training_pl, model,scope):
  """
  Calculate the total loss on a single tower running the model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size,H,W,C].
    labels: 1-D integer Tensor of [batch_size,EXTRA_DIMS (optional)].
    model: object which defines the model. Needs to have a `build` function.
    scope: unique prefix string identifying the tower, e.g. 'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Build Graph. Note,we force the variables to lie on the CPU,
  # required for multi-gpu training (automatically placed):
  _, loss, avg_ops = model.build(inputs, training_pl,
                               costs_collection='costs',
                               scope=scope, var_device='/cpu:0')
  # we want to do the averaging before, the GPUs are synchronized,
  # so that the averages are computed independently on each GPU:
  if avg_ops:
    with tf.control_dependencies(avg_ops):
      loss = tf.identity(loss)
  return loss

def average_gradients(tower_grads,clip_value=None):
  """
  Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    if grad_and_vars[0][0] is None: continue

    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads,axis=0)
    grad = tf.reduce_mean(grad,axis=[0])
    if clip_value is not None:
      # if grad is not None:
      with tf.name_scope('grad_clip') as scope:
        grad = tf.clip_by_norm(grad, clip_value+0.0)#(-clip_value+0.0), (clip_value+0.0))

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train_multi(opts,graph,optim,inputs, training_pl, model_factory,global_step,
                clip_value=None):
  """
  Train on dataset for a number of steps.
  Args:
    opts: dict, dictionary with the following options:
      gpu_ids: list of integer indices of the GPUs to use
      batch_size: integer: total batch size
                  (each GPU processes batch_size/num_gpu instances)
    graph: tf.Graph instance
    model_factory: function which creates TFModels.
                   Multiple such models are created
                   for each GPU.
                   create_optimizer(lr): returns an optimizer
  """
  num_gpus = len(opts['gpu_ids'])
  # Get images and labels for ImageNet and split the batch across GPUs.
  assert opts['batch_size'] % num_gpus == 0, ('Batch size must be divisible by number of GPUs')

  with graph.as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    # Split the batch of images and labels for towers.
    inputs_splits = utils.split_tensors(inputs, num_gpus,axis=0)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Calculate the gradients for each model tower.
    tower_grads = []
    losses = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % opts['gpu_ids'][i]):
          # note: A NAME_SCOPE only affects the names of OPS
          #       and not of variables:
          with tf.name_scope('tower_%d'%i) as scope:
            print(colorize('building graph on: tower_%d'%i,'blue',bold=True))
            model_i = model_factory.create()
            loss = tower_loss(inputs_splits[i], training_pl, model_i, scope)
            losses.append(loss)
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            # Retain summaries and other updates from ONLY THE LAST TOWER:
            # Note: Its ok for batch-norm too (don't worry)
            if i ==0:
              train_summaries = get_train_summaries(scope)
              test_summaries = get_test_summaries(scope)
            bnorm_updates = model_i.get_bnorm_ops(scope)
            # Calculate the gradients for the batch of data on this tower:
            grads = optim.compute_gradients(loss)
            tower_grads.append(grads)

    # We must calculate the mean of each gradient.
    # >>> Note that this is the **SYNCHRONIZATION POINT** across all towers.
    grads = average_gradients(tower_grads,clip_value)
    # Apply the gradients (this it the MAIN LEARNING OP):
    apply_gradient_op = optim.apply_gradients(grads, global_step=global_step)
    # Group all updates to into a single train op:
    train_op = tf.group(apply_gradient_op, bnorm_updates)
    # if bnorm_updates:
    #   with ops.control_dependencies(bnorm_updates):
    #     barrier = control_flow_ops.no_op(name='update_barrier')
    #   train_op = control_flow_ops.with_dependencies([barrier], train_op)

    # get the average loss across all towers (for printing):
    avg_tower_loss = tf.reduce_mean(losses)

    # Add a summaries for the input processing and global_step.
    train_summaries.extend(input_summaries)
    test_summaries.extend(input_summaries)
    # summaries.append(tf.summary.scalar('learning_rate', lr))
    # add a histogram summary for ALL the trainable variables:
    """
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))
    # add a summary for tracking the GRADIENTS of all the variables:
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))
    """
    # Build the summary operation from the last tower summaries:
    train_summary_op = tf.summary.merge(train_summaries)
    test_summary_op = tf.summary.merge(test_summaries)

    return avg_tower_loss,train_op, train_summary_op, test_summary_op, model_i


def train_single(opts,graph,optim,inputs, training_pl, model_factory,global_step,
                 clip_value=None):
  """
  Train on dataset for a number of steps.
  Args:
    opts: dict, dictionary with the following options:
      gpu_ids: list of integer indices of the GPUs to use
      batch_size: integer: total batch size
                  (each GPU processes batch_size/num_gpu instances)
    graph: tf.Graph instance
    model_factory: function which creates TFModels.
                   Multiple such models are created
                   for each GPU.
                   create_optimizer(lr): returns an optimizer
  """
  num_gpus = len(opts['gpu_ids'])
  assert num_gpus==1, ('Found more than one gpus in train_single')

  with graph.as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Calculate the gradients for each model tower.
    with tf.device('/gpu:%d' % opts['gpu_ids'][0]):
      # note: A NAME_SCOPE only affects the names of OPS
      #       and not of variables:
      with tf.name_scope('tower_0') as scope:
        print(colorize('building graph','blue',bold=True))
        model_i = model_factory.create()
        loss = tower_loss(inputs, training_pl, model_i, scope)

        # summaries and batch-norm updates:
        train_summaries = get_train_summaries(scope)
        test_summaries = get_test_summaries(scope)
        bnorm_updates = model_i.get_bnorm_ops(scope)
        # get the training op:
        grads_and_vars = optim.compute_gradients(loss)
      if clip_value is not None:
        with tf.name_scope('grad_clip') as scope:
          clipped_grads_and_vars = []
          for grad,var in grads_and_vars:
            if grad is not None:
              grad = tf.clip_by_norm(grad, clip_value+0.0)#(-clip_value+0.0), (clip_value+0.0))
            clipped_grads_and_vars.append((grad,var))
        grads_and_vars = clipped_grads_and_vars

      apply_grad_op = optim.apply_gradients(grads_and_vars,global_step=global_step)
      # Group all updates to into a single train op:
      train_op = tf.group(apply_grad_op, bnorm_updates)
      # Add a summaries for the input processing and global_step.
      train_summaries.extend(input_summaries)
      test_summaries.extend(input_summaries)
      train_summary_op = tf.summary.merge(train_summaries)
      test_summary_op = tf.summary.merge(test_summaries)

    return loss,train_op,train_summary_op,test_summary_op,model_i

def train_single_cpu(opts,graph,optim,inputs, training_pl, model_factory,global_step,
                     clip_value=None):
  """
  Train on dataset for a number of steps.
  Args:
    opts: dict, dictionary with the following options:
      gpu_ids: list of integer indices of the GPUs to use
      batch_size: integer: total batch size
                  (each GPU processes batch_size/num_gpu instances)
    graph: tf.Graph instance
    model_factory: function which creates TFModels.
                   Multiple such models are created
                   for each GPU.
                   create_optimizer(lr): returns an optimizer
  """
  num_gpus = len(opts['gpu_ids'])
  assert num_gpus==0, ('Found more non-zero GPU ids')

  with graph.as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Calculate the gradients for each model tower.
    # note: A NAME_SCOPE only affects the names of OPS
    #       and not of variables:
    with tf.name_scope('cpu_tower') as scope:
      print(colorize('building graph','blue',bold=True))
      model_i = model_factory.create()
      loss = tower_loss(inputs, training_pl, model_i, scope)

      # summaries and batch-norm updates:
      train_summaries = get_train_summaries(scope)
      test_summaries = get_test_summaries(scope)
      bnorm_updates = model_i.get_bnorm_ops(scope)
      # get the training op:
      grads_and_vars = optim.compute_gradients(loss)
      if clip_value is not None:
        with tf.name_scope('grad_clip') as scope:
          grads_and_vars = [(tf.clip_by_norm(grad, clip_value+0.0), var) for grad, var in grads_and_vars]

      apply_grad_op = optim.apply_gradients(grads_and_vars,global_step=global_step)
      # Group all updates to into a single train op:
      train_op = tf.group(apply_grad_op, bnorm_updates)
      # Add a summaries for the input processing and global_step.
      train_summaries.extend(input_summaries)
      test_summaries.extend(input_summaries)
      train_summary_op = tf.summary.merge(train_summaries)
      test_summary_op = tf.summary.merge(test_summaries)

  return loss,train_op,train_summary_op, test_summary_op,model_i

def train_split_gpus(opts, graph, optim, inputs, training_pl, model_factory,
                      global_step, clip_value):
  """
  Network components are assumed to have been split across
  multiple devices, hence manual averaging of gradient is not done.
  Instead grads are co-located with ops, and just applied to the
  vars through the optimizer.
  """
  with graph.as_default(), tf.device('/cpu:0'):
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Calculate the gradients for each model tower.
    with tf.name_scope('split_gpus') as scope:
      model = model_factory.create()
      loss = tower_loss(inputs, training_pl, model, scope)
      # summaries and batch-norm updates:
      train_summaries = get_train_summaries(scope)
      test_summaries = get_test_summaries(scope)
      bnorm_updates = model.get_bnorm_ops(scope)
      # get the training op:
      grads_and_vars = optim.compute_gradients(loss, colocate_gradients_with_ops=True)
      if clip_value is not None:
        with tf.name_scope('grad_clip') as scope:
          clipped_grads_and_vars = []
          for grad,var in grads_and_vars:
            if grad is not None:
              grad = tf.clip_by_norm(grad, clip_value+0.0)#(-clip_value+0.0), (clip_value+0.0))
            clipped_grads_and_vars.append((grad,var))
        grads_and_vars = clipped_grads_and_vars

      apply_grad_op = optim.apply_gradients(grads_and_vars,global_step=global_step)
      # Group all updates to into a single train op:
      train_op = tf.group(apply_grad_op, bnorm_updates)
      # Add a summaries for the input processing and global_step.
      train_summaries.extend(input_summaries)
      test_summaries.extend(input_summaries)
      train_summary_op = tf.summary.merge(train_summaries)
      test_summary_op = tf.summary.merge(test_summaries)
    return loss, train_op, train_summary_op, test_summary_op, model

def setup_training(opts, graph, optim, inputs, training_pl, model_factory,
                   global_step, clip_value=None,
                   split_gpus=False):
  """
  SPLIT_GPUS: if true, the network components are assumed to have been split across
              multiple devices, hence manual averaging of gradient is not done.
              Instead grads are co-located with ops, and just applied to the
              vars through the optimizer.
  """
  if split_gpus:
    print(colorize('training SPLIT across multiple GPUs','red',bold=True))
    return train_split_gpus(opts, graph, optim, inputs, training_pl,
            model_factory, global_step, clip_value)
  else:
    num_gpus = len(opts['gpu_ids'])
    if num_gpus == 0:
      print(colorize('training on CPU','red',bold=True))
      return train_single_cpu(opts,graph,optim, inputs, training_pl,
                              model_factory, global_step,clip_value)
    elif num_gpus == 1:
      print(colorize('training on SINGLE gpu: %d'%opts['gpu_ids'][0],'red',bold=True))
      return train_single(opts,graph,optim, inputs, training_pl,
                              model_factory, global_step,clip_value)
    elif num_gpus > 1:
      print(colorize('training on MULTIPLE gpus','red',bold=True))
      return train_multi(opts,graph,optim, inputs, training_pl,
                              model_factory, global_step,clip_value)


def train_loop(opts, graph, loss, train_dataset, training_pl, handle_pl, train_op,
               train_summary_op, test_summary_op,
               num_steps,
               global_step, checkpoint_fname,
               test_dataset=None,
               ignore_missing_vars=False,
               reset_global_step=False, vars_to_restore=None,
               exclude_vars=None, fwd_only=False, allow_growth=False):
  """
  training loop without a supervisor:
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  with graph.as_default(), tf.device('/cpu:0'):
    # define iterators
    train_iterator = train_dataset.make_initializable_iterator()
    if test_dataset:
      test_iterator = test_dataset.make_initializable_iterator()

    session_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    session_config.gpu_options.allow_growth = allow_growth
    session = tf.Session(config=session_config)

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    session.run([global_init,local_init])

    # set up iterators
    train_handle = session.run(train_iterator.string_handle())
    session.run(train_iterator.initializer)
    if test_dataset:
      test_handle = session.run(test_iterator.string_handle())

    # check if we need to restore the model:
    if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname+'.index'):
      print(colorize('RESTORING MODEL from: '+checkpoint_fname, 'blue', bold=True))
      if not isinstance(vars_to_restore,list):
        if vars_to_restore == 'all':
          vars_to_restore = tf.global_variables()
        elif vars_to_restore == 'model':
          vars_to_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
      if reset_global_step >= 0:
        print(colorize('Setting global-step to %d.'%reset_global_step,'red',bold=True))
        var_names = [v.name for v in vars_to_restore]
        reset_vid = [i for i in xrange(len(var_names)) if 'global_step' in var_names[i]]
        if reset_vid:
          vars_to_restore.pop(reset_vid[0])
      print(colorize('vars-to-be-restored:','green',bold=True))
      print(colorize(', '.join([v.name for v in vars_to_restore]),'green'))
      if ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_fname)
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_ignored = [v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
        print(colorize('vars-IGNORED (not restoring):','blue',bold=True))
        print(colorize(', '.join(vars_ignored),'blue'))
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
      if exclude_vars:
        for exclude_var_name in exclude_vars:
          var_names = [v.name for v in vars_to_restore]
          reset_vid = [i for i in xrange(len(var_names)) if exclude_var_name in var_names[i]]
          if reset_vid:
            vars_to_restore.pop(reset_vid[0])
      restorer = tf.train.Saver(var_list=vars_to_restore)
      restorer.restore(session,checkpoint_fname)

    # create a summary writer:
    summary_writer = tf.summary.FileWriter(opts['log_dir'], graph=session.graph)
    # create a check-pointer:
    #  --> keep ALL the checkpoint files:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # get the value of the global-step:
    start_step = session.run(global_step)
    # run the training loop:
    begin_time = time.time()
    for step in xrange(start_step, num_steps):
      start_time = time.time()
      if fwd_only:  # useful for timing..
        feed_dict = {handle_pl: train_handle, training_pl: False}
        loss_value = session.run(loss, feed_dict=feed_dict)
      else:
        feed_dict = {handle_pl: train_handle, training_pl: True}
        if step % opts['n_summary'] == 0:
          loss_value, _, summary_str = session.run([loss, train_op,
                                                    train_summary_op],
                                                   feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush() # write to disk now
        else:
          loss_value, _ = session.run([loss, train_op], feed_dict=feed_dict)
      duration = time.time() - start_time

      # make sure that we have non NaNs:
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # print stats for this batch:
      examples_per_sec = opts['batch_size'] / float(duration)
      format_str = '%s: step %d, loss = %.4f (%.1f examples/sec) %.3f sec/batch'
      tf.logging.info(format_str % (datetime.now(), step, loss_value,
                      examples_per_sec, duration))

      # periodically test on test set
      if not fwd_only and test_dataset and step % opts['n_test'] == 0:
        feed_dict = {handle_pl: test_handle, training_pl: False}
        metrics_reset_ops = tf.get_collection('metrics_reset')
        metrics_update_ops = tf.get_collection('metrics_update')
        session.run(metrics_reset_ops)
        session.run(test_iterator.initializer)
        test_iter = 0
        while True:
          try:
            start_time = time.time()
            if test_iter == 0:
              loss_value, summary_str, _ = session.run(
                [loss, test_summary_op, metrics_update_ops],
                feed_dict=feed_dict)
              summary_writer.add_summary(summary_str, step)
            else:
              loss_value, _ = session.run(
                [loss, metrics_update_ops], feed_dict=feed_dict)
            duration = time.time() - start_time

            examples_per_sec = opts['batch_size'] / float(duration)
            format_str = 'test: %s: step %d, loss = %.4f (%.1f examples/sec) %.3f sec/batch'
            tf.logging.info(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, duration))
          except tf.errors.OutOfRangeError:
            print('iteration through test set finished')
            break
          test_iter += 1

        metrics_summaries_ops = tf.get_collection('metrics_summaries')
        if metrics_summaries_ops:
          summary_str = session.run(tf.summary.merge(metrics_summaries_ops))
          summary_writer.add_summary(summary_str, step)

        summary_writer.flush() # write to disk now

      # periodically write the summary (after every N_SUMMARY steps):
      if not fwd_only:
        # periodically checkpoint:
        if step % opts['n_checkpoint'] == 0:
          checkpoint_path = osp.join(opts['log_dir'],'model.ckpt')
          saver.save(session, checkpoint_path, global_step=step)
    total_time = time.time()-begin_time
    samples_per_sec = opts['batch_size'] * num_steps / float(total_time)
    print('Avg. samples per second %.3f'%samples_per_sec)
