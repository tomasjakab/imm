# ==========================================================
# Author: Ankush Gupta, Tomas Jakab
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import


from tensorflow.contrib.framework.python.ops import variables
import tensorflow as tf
import os.path as osp

# network definition:
from imm.models.imm_model import IMMModel
from imm.utils.box import Box

import imm.train.cnn_train_multi as tru
from imm.utils.colorize import colorize

import metayaml
from imm.utils.dataset_import import import_dataset
"""
So the main steps are:
  1. create the dataset object
  2. get a model factory
  3. build the training/summary ops
  4. run the training loop.
"""


class model_factory():
  """
  Factory which can be used to
  instantiate models.
  """
  def __init__(self, network, **kwargs):
    self.network = network
    self.net_args = kwargs

  def create(self):
    return self.network(**self.net_args)


def load_configs(file_names):
  """
  Loads the yaml config files.
  """
  # with open(file_name, 'r') as f:
  #   config_str = f.read()
  # config = Box.from_yaml(config_str)
  config = Box(metayaml.read(file_names))
  return config


def main(args):
  config = load_configs(args.configs)
  train_config = config.training
  gpus = range(args.ngpus)

  # get the data and logging (checkpointing) directories:
  data_dir = train_config.datadir
  log_dir = train_config.logdir

  SUBSET = 'train'
  NUM_STEPS = 30000000
  # value at which the gradients are clipped
  GRAD_CLIP = train_config.gradclip

  if args.checkpoint is not None:
    checkpoint_fname = args.checkpoint
  else:
    print(colorize('No checkpoint file specified. Initializing randomly.','red',bold=True))
    checkpoint_fname = osp.join(log_dir,'INVALID')

  opts = {}
  opts['gpu_ids'] = gpus
  opts['log_dir'] = log_dir
  opts['n_summary'] = 10 # number of iterations after which to run the summary-op
  if hasattr(train_config,'n_test'):
    opts['n_test'] = train_config.n_test
  else:
    opts['n_test'] = 500
  opts['n_checkpoint'] = train_config.ncheckpoint # number of iteration after which to save the model

  batch_size = train_config.batch
  graph = tf.Graph()
  with graph.as_default():
    global_step = variables.model_variable('global_step',shape=[],
                                            initializer=tf.constant_initializer(args.reset_global_step),
                                            trainable=False)

    # common model / optimizer parameters:
    lr = args.lr_multiple * tf.train.exponential_decay(train_config.lr.start_val,
                                global_step,
                                train_config.lr.step,
                                train_config.lr.decay,
                                staircase=True)
    if train_config.optim.lower() == 'adam':
      optim = tf.train.AdamOptimizer(lr, name='Adam')
    elif train_config.optim.lower() == 'adadelta':
      optim = tf.train.AdadeltaOptimizer(lr, rho=0.95,epsilon=1e-06,use_locking=False,name='Adadelta')
    elif train_config.optim.lower() == 'adagrad':
      optim = tf.train.AdagradOptimizer(lr, use_locking=False,name='AdaGrad')
    else:
      raise ValueError('Optimizer = %s not suppoerted'%train_config.optim)

    factory = model_factory(IMMModel,
                            config=config.model,
                            global_step=global_step)

    opts['batch_size'] = batch_size
    tf.summary.scalar('lr', lr) # add a summary
    print(colorize('log_dir: ' + log_dir,'green',bold=True))
    print(colorize('BATCH-SIZE: %d'%batch_size,'red',bold=True))

    # dynamic import of a dataset class
    dset_class = import_dataset(train_config.dset)

    # default datasets parameters
    train_dset_params = {}
    test_dset_params = {}

    train_subset = 'train'
    test_subset = 'test'
    if hasattr(train_config, 'train_dset_params'):
      train_dset_params.update(train_config.train_dset_params)
      if 'subset' in train_dset_params:
        train_subset = train_dset_params['subset']
        # delete because not positional kwarg
        del train_dset_params['subset']
    if hasattr(train_config, 'test_dset_params'):
      test_dset_params.update(train_config.test_dset_params)
      if 'subset' in test_dset_params:
        test_subset = test_dset_params['subset']
        # delete because not positional kwarg
        del test_dset_params['subset']

    train_dset = dset_class(train_config.datadir, subset=train_subset,
                            **train_dset_params)
    train_dset = train_dset.get_dataset(batch_size, repeat=True, shuffle=False,
                                        num_preprocess_threads=12)

    if hasattr(train_config, 'max_test_samples'):
      raise ValueError('max_test_samples attribute deprecated')
    test_dset = dset_class(train_config.datadir, subset=test_subset,
                           **test_dset_params)
    test_dset = test_dset.get_dataset(batch_size, repeat=False, shuffle=False,
                                      num_preprocess_threads=12)

    # set up inputs
    training_pl = tf.placeholder(tf.bool)
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(
      handle_pl, train_dset.output_types, train_dset.output_shapes)
    inputs = base_iterator.get_next()

    split_gpus = False
    if hasattr(config.model, 'split_gpus'):
      split_gpus = config.model.split_gpus

    # create the network distributed over multi-GPUs:
    loss, train_op, train_summary_op, test_summary_op, _ = tru.setup_training(
      opts, graph, optim, inputs, training_pl, factory, global_step,
      clip_value=GRAD_CLIP, split_gpus=split_gpus)

    # run the training loop:
    if args.restore_optim:
      restore_vars = 'all'
    else:
      restore_vars = 'model'

    tru.train_loop(opts, graph, loss, train_dset, training_pl, handle_pl,
                   train_op, train_summary_op, test_summary_op, NUM_STEPS,
                   global_step, checkpoint_fname,
                   test_dataset=test_dset,
                   ignore_missing_vars=args.ignore_missing_vars,
                   reset_global_step=args.reset_global_step,
                   vars_to_restore=restore_vars,
                   exclude_vars=[],
                   allow_growth=train_config.allow_growth)



if  __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Train Unsupervised Sequence Model')
  parser.add_argument('--configs', nargs='+', default=[], help='Paths to the config files.')
  parser.add_argument('--ngpus',type=int,default=1,required=False,help='Number of GPUs to use for training.')
  parser.add_argument('--lr-multiple',type=float,default=1,help='multiplier on the learning rate.')
  parser.add_argument('--checkpoint',type=str,default=None,
                      help='checkpoint file-name of the *FULL* model to restore.')
  parser.add_argument('--restore-optim',action='store_true',help='Restore the optimizer variables.')
  parser.add_argument('--reset-global-step',type=int,default=-1,help='Force the value of global step.')
  parser.add_argument('--ignore-missing-vars',action='store_true',help='Skip re-storing vars not in the checkpoint file.')
  args = parser.parse_args()
  main(args)
