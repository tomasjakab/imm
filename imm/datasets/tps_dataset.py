# ==========================================================
# Author: Tomas Jakab, Ankush Gupta
# ==========================================================
from __future__ import division

import numpy as np
import os.path as osp
import tensorflow as tf

from imm.datasets.impair_dataset import ImagePairDataset
from imm.utils.tps_sampler import TPSRandomSampler



class TPSDataset(ImagePairDataset):

  def __init__(self, data_dir, subset, max_samples=None,
               image_size=[128, 128], order_stream=False, landmarks=False,
               tps=True, vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='TPSDataset'):

    super(TPSDataset, self).__init__(
        data_dir, subset, image_size=image_size, jittering=False, name=name)

    if landmarks and tps:
      raise ValueError('Outputing landmarks is not supported with TPS transform.')

    self._max_samples = max_samples
    self._order_stream = order_stream

    self._tps = tps
    if tps:
      self._target_sampler = TPSRandomSampler(
        image_size[1], image_size[0], rotsd=rotsd[0], scalesd=scalesd[0],
        transsd=transsd[0], warpsd=warpsd[:2], pad=False)
      self._source_sampler = TPSRandomSampler(
        image_size[1], image_size[0], rotsd=rotsd[1], scalesd=scalesd[1],
        transsd=transsd[1], warpsd=warpsd[2:], pad=False)


  def num_samples(self):
    raise NotImplementedError()


  def _get_smooth_step(self, n, b):
    x = tf.linspace(tf.cast(-1, tf.float32), 1, n)
    y = 0.5 + 0.5 * tf.tanh(x / b)
    return y


  def _get_smooth_mask(self, h, w, margin, step):
    b = 0.4
    step_up = self._get_smooth_step(step, b)
    step_down = self._get_smooth_step(step, -b)
    def create_strip(size):
      return tf.concat(
          [tf.zeros(margin, dtype=tf.float32),
           step_up,
           tf.ones(size - 2 * margin - 2 * step, dtype=tf.float32),
           step_down,
           tf.zeros(margin, dtype=tf.float32)], axis=0)
    mask_x = create_strip(w)
    mask_y = create_strip(h)
    mask2d = mask_y[:, None] * mask_x[None]
    return mask2d


  def _apply_tps(self, inputs):
    image = inputs['image']
    mask = inputs['mask']

    def target_warp(images):
      return self._target_sampler.forward_py(images)
    def source_warp(images):
      return self._source_sampler.forward_py(images)

    image = tf.concat([mask, image], axis=3)
    shape = image.shape

    future_image = tf.py_func(target_warp, [image], tf.float32)
    image = tf.py_func(source_warp, [future_image], tf.float32)

    image.set_shape(shape)
    future_image.set_shape(shape)

    future_mask = future_image[..., 0:1]
    future_image = future_image[..., 1:]
    mask = image[..., 0:1]
    image = image[..., 1:]

    inputs['image'] = image
    inputs['future_image'] = future_image
    inputs['mask'] = future_mask
    return inputs


  def _get_image(self, idx):
    image = osp.join(self._image_dir, self._images[idx])
    landmarks = self._keypoints[idx][:, [1, 0]]

    inputs = {'image': image, 'landmarks': landmarks}
    inputs.update({k: v for k, v in self.LANDMARK_LABELS.items()})
    return inputs


  def _get_random_image(self):
    idx = np.random.randint(len(self._images))
    return self._get_image(idx)


  def _get_ordered_stream(self):
    for i in range(len(self._images)):
      yield self._get_image(i)


  def sample_image_pair(self):
    f_sample = self._get_random_image
    if self._order_stream:
      g = self._get_ordered_stream()
      f_sample = lambda: next(g)
    max_samples = float('inf')
    if self._max_samples is not None:
      max_samples = self._max_samples
    i_samp = 0
    while i_samp < max_samples:
      yield f_sample()
      if self._max_samples is not None:
          i_samp += 1


  def get_dataset(self, batch_size, repeat=False, shuffle=False,
                  num_preprocess_threads=12, keep_aspect=True, prefetch=True):
    """
    Returns a tf.Dataset object which iterates over samples.
    """
    def sample_generator():
      return self.sample_image_pair()

    sample_dtype = self._get_sample_dtype()
    sample_shape = self._get_sample_shape()
    dataset = tf.data.Dataset.from_generator(
        sample_generator, sample_dtype, sample_shape)
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(2000)

    dataset = dataset.map(self._proc_im_pair,
                          num_parallel_calls=num_preprocess_threads)

    dataset = dataset.batch(batch_size)
    if self._tps:
      dataset = dataset.map(self._apply_tps, num_parallel_calls=1)
    if prefetch:
      dataset = dataset.prefetch(1)
    return dataset
