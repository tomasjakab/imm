# ==========================================================
# Author: Tomas Jakab
# ==========================================================
from __future__ import division

import os.path as osp
import os
import tensorflow as tf
from scipy.io import loadmat

from imm.datasets.tps_dataset import TPSDataset



def load_dataset(data_dir, subset):
  load_subset = 'train' if subset in ['train', 'val'] else 'test'
  with open(os.path.join(data_dir, 'aflw_' + load_subset + '_images.txt'), 'r') as f:
    images = f.read().splitlines()
  mat = loadmat(os.path.join(data_dir, 'aflw_' + load_subset + '_keypoints.mat'))
  keypoints = mat['gt'][:, :, [1, 0]]
  sizes = mat['hw']

  if subset in ['train', 'val']:
    # put the last 10 percent of the training aside for validation
    n_validation = int(round(0.1 * len(images)))
    if subset == 'train':
      images = images[:-n_validation]
      keypoints = keypoints[:-n_validation]
      sizes = sizes[:-n_validation]
    elif subset == 'val':
      images = images[-n_validation:]
      keypoints = keypoints[-n_validation:]
      sizes = sizes[-n_validation:]
    else:
      raise ValueError()

  image_dir = os.path.join(data_dir, 'output')
  return image_dir, images, keypoints, sizes



class AFLWDataset(TPSDataset):
  LANDMARK_LABELS = {'left_eye': 0, 'right_eye': 1}
  N_LANDMARKS = 5


  def __init__(self, data_dir, subset, max_samples=None,
               image_size=[128, 128], order_stream=False, landmarks=False,
               tps=True, vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='CelebADataset'):

    super(AFLWDataset, self).__init__(
        data_dir, subset, max_samples=max_samples,
        image_size=image_size, order_stream=order_stream, landmarks=landmarks,
        tps=tps, vertical_points=vertical_points,
        horizontal_points=horizontal_points, rotsd=rotsd, scalesd=scalesd,
        transsd=transsd, warpsd=warpsd, name=name)

    self._image_dir, self._images, self._keypoints, self._sizes = load_dataset(
        self._data_dir, self._subset)


  def _get_sample_dtype(self):
    d =  {'image': tf.string,
          'landmarks': tf.float32,
          'size': tf.int32}
    d.update({k: tf.int32 for k in self.LANDMARK_LABELS.keys()})
    return d


  def _get_sample_shape(self):
    d = {'image': None,
         'landmarks': [self.N_LANDMARKS, 2],
         'size': 2}
    d.update({k: [] for k in self.LANDMARK_LABELS.keys()})
    return d


  def _proc_im_pair(self, inputs):
    with tf.name_scope('proc_im_pair'):
      height, width = self._image_size[:2]

      # read in the images:
      image = self._read_image_tensor_or_string(inputs['image'])

      if 'landmarks' in inputs:
        landmarks = inputs['landmarks']
      else:
        landmarks = None

      assert self._image_size[0] == self._image_size[1]
      final_size = self._image_size[0]

      if landmarks is not None:
        original_sz = inputs['size']
        landmarks = self._resize_points(
            landmarks, original_sz, [final_size, final_size])

      image = tf.image.resize_images(
          image, [final_size, final_size], tf.image.ResizeMethod.BILINEAR,
          align_corners=True)

      mask = self._get_smooth_mask(height, width, 10, 20)[:, :, None]

      future_landmarks = landmarks
      future_image = image

      inputs = {k: inputs[k] for k in self._get_sample_dtype().keys()}
      inputs.update({'image': image, 'future_image': future_image,
                     'mask': mask, 'landmarks': landmarks,
                     'future_landmarks': future_landmarks})
    return inputs

  def _get_image(self, idx):
    image = osp.join(self._image_dir, self._images[idx])
    landmarks = self._keypoints[idx][:, [1, 0]]
    size = self._sizes[idx]

    inputs = {'image': image, 'landmarks': landmarks, 'size': size}
    inputs.update({k: v for k, v in self.LANDMARK_LABELS.items()})
    return inputs
