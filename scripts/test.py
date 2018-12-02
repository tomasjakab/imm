# ==========================================================
# Author: Tomas Jakab
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os.path as osp

from imm.eval import eval_imm
from imm.models.imm_model import IMMModel
import sklearn.linear_model

from imm.utils.dataset_import import import_dataset



def evaluate(net, net_file, model_config, training_config, train_dset, test_dset,
             batch_size=100, bias=False):
  # %% ---------------------------------------------------------------------------
  # ------------------------------- Run TensorFlow -------------------------------
  # ------------------------------------------------------------------------------
  def evaluate(dset):
    results = eval_imm.evaluate(
        dset, net, model_config, net_file, training_config, batch_size=batch_size,
        random_seed=0, eval_tensors=['gauss_yx', 'future_landmarks'])
    results = {k: np.concatenate(v) for k, v in results.items()}
    return results

  train_tensors = evaluate(train_dset)
  test_tensors = evaluate(test_dset)

  # %% ---------------------------------------------------------------------------
  # --------------------------- Regress landmarks --------------------------------
  # ------------------------------------------------------------------------------

  def convert_landmarks(tensors, im_size):
    landmarks = tensors['gauss_yx']
    landmarks_gt = tensors['future_landmarks'].astype(np.float32)
    im_size = np.array(im_size)
    landmarks = ((landmarks + 1) / 2.0) * im_size
    n_samples = landmarks.shape[0]
    landmarks = landmarks.reshape((n_samples, -1))
    landmarks_gt = landmarks_gt.reshape((n_samples, -1))
    return landmarks, landmarks_gt

  X_train, y_train = convert_landmarks(train_tensors, train_dset.image_size)
  X_test, y_test = convert_landmarks(test_tensors, train_dset.image_size)

  # regression
  regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=bias)
  _ = regr.fit(X_train, y_train)
  y_predict = regr.predict(X_test)

  landmarks_gt = test_tensors['future_landmarks'].astype(np.float32)
  landmarks_regressed = y_predict.reshape(landmarks_gt.shape)

  # normalized error with respect to intra-occular distance
  eyes = landmarks_gt[:, :2, :]
  occular_distances = np.sqrt(
      np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
  distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
  mean_error = np.mean(distances / occular_distances[:, None])

  return mean_error


def main(args):
  experiment_name = args.experiment_name
  iteration = args.iteration
  im_size = args.im_size
  bias = args.bias
  batch_size = args.batch_size
  n_train_samples = None
  buffer_name = args.buffer_name

  postfix = ''
  if bias:
    postfix += '-bias'
  else:
    postfix += '-no_bias'
  postfix += '-' + args.test_dataset
  postfix += '-' + args.test_split
  if n_train_samples is not None:
    postfix += '%.0fk' % (n_train_samples / 1000.0)

  config = eval_imm.load_configs(
      [args.paths_config,
       osp.join('configs', 'experiments', experiment_name + '.yaml')])

  if args.train_dataset == 'mafl':
    train_dataset_class = import_dataset('celeba')
    train_dset = train_dataset_class(
        config.training.datadir, dataset='mafl', subset='train',
        order_stream=True, max_samples=n_train_samples, tps=False,
        image_size=[im_size, im_size])
  elif args.train_dataset == 'aflw':
    train_dataset_class = import_dataset('aflw')
    train_dset = train_dataset_class(
        config.training.datadir, subset='train',
        order_stream=True, max_samples=n_train_samples, tps=False,
        image_size=[im_size, im_size])
  else:
    raise ValueError('Dataset %s not supported.' % args.train_dataset)

  if args.test_dataset == 'mafl':
    test_dataset_class = import_dataset('celeba')
    test_dset = test_dataset_class(
        config.training.datadir, dataset='mafl', subset=args.test_split,
        order_stream=True, tps=False,
        image_size=[im_size, im_size])
  elif args.test_dataset == 'aflw':
    test_dataset_class = import_dataset('aflw')
    test_dset = test_dataset_class(
        config.training.datadir, subset=args.test_split,
        order_stream=True, tps=False,
        image_size=[im_size, im_size])
  else:
    raise ValueError('Dataset %s not supported.' % args.test_dataset)

  net = IMMModel

  model_config = config.model
  training_config = config.training

  if iteration is not None:
    net_file = 'model.ckpt-' + str(iteration)
  else:
    net_file = 'model.ckpt'
  checkpoint_file = osp.join(config.training.logdir, net_file + '.meta')
  if not osp.isfile(checkpoint_file):
    raise ValueError('Checkpoint file %s not found.' % checkpoint_file)

  mean_error = evaluate(
      net, net_file, model_config, training_config, train_dset, test_dset,
      batch_size=batch_size, bias=bias)

  if hasattr(config.training.train_dset_params, 'dataset'):
    model_dataset = config.training.train_dset_params.dataset
  else:
    model_dataset = config.training.dset

  print('')
  print('========================= RESULTS =========================')
  print('model trained in unsupervised way on %s dataset' % model_dataset)
  print('regressor trained on %s training set' % args.train_dataset)
  print('error on %s datset %s set: %.5f (%.3f percent)' % (
      args.test_dataset, args.test_split,
      mean_error, mean_error * 100.0))
  print('===========================================================')


if  __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Test model on face datasets.')
  parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment to evaluate.')
  parser.add_argument('--train-dataset', type=str, required=True, help='Training dataset for regressor (mafl|aflw).')
  parser.add_argument('--test-dataset', type=str, required=True, help='Testing dataset for regressed landmarks (mafl|aflw).')

  parser.add_argument('--paths-config', type=str, default='configs/paths/default.yaml', required=False, help='Path to the paths config.')
  parser.add_argument('--iteration', type=int, default=None, required=False, help='Checkpoint iteration to evaluate.')
  parser.add_argument('--test-split', type=str, default='test', required=False, help='Test split (val|test).')
  parser.add_argument('--buffer-name', type=str, default=None, required=False, help='Name of the buffer when using matlab data pipeline.')
  parser.add_argument('--im-size', type=int, default=128, required=False, help='Image size.')
  parser.add_argument('--bias', action='store_true', required=False, help='Use bias in the regressor.')
  parser.add_argument('--batch-size', type=int, default=100, required=False, help='batch_size')

  args = parser.parse_args()
  main(args)
