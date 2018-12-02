# ==========================================================
# Author: Tomas Jakab, Ankush Gupta
# ==========================================================
"""
Interface for dataaset returning image pairs.
"""
import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod

from ..data_utils import image_utils as imu


class ImagePairDataset(object):
  """Abstract class for sampling image pairs."""

  __metaclass__ = ABCMeta

  def __init__( self, data_dir, subset,
                image_size=[128, 128], bbox_padding=[10, 10],
                crop_to_bbox=False, jittering=None,
                augmentations=['flip', 'swap'], name='PairDataset'):
    """
    JITTERING : True/ False / None. If None => True if subset=='train', else false.
    """

    self._data_dir = data_dir
    self._subset = subset
    self._image_size = image_size
    self.image_size = image_size
    self._bbox_padding = bbox_padding
    self._crop_to_bbox = crop_to_bbox
    self._jittering = jittering
    self._augmentations = augmentations
    self._name = name


  def _read_image_tensor_or_string(self, image, channels=3, format='jpeg'):
    """
    Reads image from file if string, and reshapes, and casts to float.
    """
    dtype = image.dtype
    height, width = self._image_size[:2]
    if dtype == tf.string:
      image =  tf.read_file(image)
      image = imu.decode_image_buffer(
          image, format, cast_float=False, channels=channels)
    image.set_shape([None, None, channels])
    image = tf.to_float(image)
    return image


  def _find_common_box(self, box1, box2):
    """
    Finds the union of two boxes, represented as [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('common_box'):
      box = tf.concat([tf.minimum(box1[:2], box2[:2]),
                        tf.maximum(box1[2:], box2[2:])], axis=0)
    return box


  def _fit_bbox(self, box, image_sz):
    """
    Ajusts box size to have the same aspect ratio as the target image
    while preserving the centre.
    """
    with tf.name_scope('fit_box'):
      box = tf.to_float(box)
      im_h, im_w = tf.to_float(image_sz[0]), tf.to_float(image_sz[1])
      h, w = box[2] - box[0], box[3] - box[1]

      # r_im - image aspect ratio, r - box aspect ratio
      r_im = im_w / im_h
      r = w / h

      centre = [box[0] + h / 2, box[1] + w / 2]

      # if r < r_im
      def r_lt_r_im():
        return h, r_im * h
      # if r >= r_im
      def r_gte_r_im():
        return (1 / r_im) *  w, w
      h, w = tf.cond(r < r_im, r_lt_r_im, r_gte_r_im)

      box = [centre[0] - h / 2, centre[1] - w / 2,
             centre[0] + h / 2, centre[1] + w / 2]

      box = tf.cast(tf.stack(box), tf.int32)
    return box


  def _crop_to_box(self, image, bbox, pad=True):
    with tf.name_scope('crop_to_box'):
      bbox = tf.unstack(bbox)
      if pad:
        sz = tf.shape(image)[:2]
        pad_top    = -tf.minimum(0, bbox[0])
        pad_left   = -tf.minimum(0, bbox[1])
        pad_bottom = -tf.minimum(0, sz[0] - bbox[2])
        pad_right  = -tf.minimum(0, sz[1] - bbox[3])
        c = image.shape.as_list()[2]
        image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        # NOTE: workaround as tf.pad does not infer number channels
        image.set_shape([None, None, c])
        bbox[0], bbox[2] = bbox[0] + pad_top,  bbox[2] + pad_top
        bbox[1], bbox[3] = bbox[1] + pad_left, bbox[3] + pad_left
      image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return image


  def _resize_points(self, points, size, new_size):
    with tf.name_scope('resize_landmarks'):
      size = tf.convert_to_tensor(size)
      new_size = tf.convert_to_tensor(new_size)
      dtype = points.dtype
      ratio = tf.to_float(new_size) / tf.to_float(size)
      points = tf.cast(tf.to_float(points) * ratio[None], dtype)
    return points


  def _apply_rand_augment(self, fn, im0, im1, probability):
    with tf.name_scope(None, default_name='rand_augment'):
      im0, im1 = tf.cond(tf.random_uniform([]) < probability,
                         lambda: fn(im0, im1),
                         lambda: (im0, im1))
    return im0, im1


  def _jitter_im(self, im0, im1, flip=True, swap=True):
    """
    Jitters the image pair.
    """
    with tf.name_scope('image_jitter'):
      # random horizontal flips:
      if flip:
        im0, im1 = tf.cond(tf.random_uniform([]) < 0.5,
                            lambda: (im0, im1),
                            lambda: (im0[:,::-1,:], im1[:,::-1,:]))
      if swap:
        im0, im1 = tf.cond(tf.random_uniform([]) < 0.5,
                           lambda: (im0, im1),
                           lambda: (im1, im0))
    return im0, im1


  def _jitter_im_and_points(self, im0, im1, p0, p1, flip=True, swap=True):
    """
    Jitters the image pair.
    """
    with tf.name_scope('image_jitter'):
      # random horizontal flips:
      def do_flip(im0, im1, p0, p1):
        im0 = im0[:, ::-1, :]
        im1 = im1[:, ::-1, :]
        max_x = tf.to_float(tf.shape(im0)[1] - 1)
        p0 = tf.stack([p0[:, 0], max_x - p0[:, 1]], axis=1)
        p1 = tf.stack([p1[:, 0], max_x - p1[:, 1]], axis=1)
        return im0, im1, p0, p1

      if flip:
        im0, im1, p0, p1 = tf.cond(tf.random_uniform([]) < 0.5,
                                   lambda: (im0, im1, p0, p1),
                                   lambda: do_flip(im0, im1, p0, p1))
      if swap:
        im0, im1, p0, p1 = tf.cond(tf.random_uniform([]) < 0.5,
                                   lambda: (im0, im1, p0, p1),
                                   lambda: (im1, im0, p1, p0))
    return im0, im1, p0, p1


  def _proc_im_pair(self, inputs, keep_aspect=True):
    with tf.name_scope('proc_im_pair'):
      height, width = self._image_size[:2]

      # read in the images:
      image = self._read_image_tensor_or_string(inputs['image'])
      future_image = self._read_image_tensor_or_string(inputs['future_image'])

      if 'landmarks' in inputs:
        landmarks = inputs['landmarks']
        future_landmarks = inputs['future_landmarks']
      else:
        landmarks = None
        future_landmarks = None

      sample_dtype = self._get_sample_dtype()

      # crop to bbox
      if self._crop_to_bbox:
        bbox = inputs['bbox']
        future_bbox = inputs['future_bbox']
        bbox_union = self._find_common_box(bbox, future_bbox)
        if keep_aspect:
          bbox_union = self._fit_bbox(bbox_union, [height, width])
        image        = self._crop_to_box(image,        bbox_union)
        future_image = self._crop_to_box(future_image, bbox_union)

        if landmarks is not None:
          landmarks -= bbox_union[:2][None]
          future_landmarks -= bbox_union[:2][None]

      if landmarks is not None:
        sz = tf.shape(image)[:2]
        new_size = tf.constant([height, width])
        landmarks = self._resize_points(landmarks, sz, new_size)
        sz = tf.shape(future_image)[:2]
        future_landmarks = self._resize_points(future_landmarks, sz, new_size)

      image        = tf.image.resize_images(image,        [height, width])
      future_image = tf.image.resize_images(future_image, [height, width])

      should_jitter = ((self._jittering is not None and self._jittering)
                       or (self._jittering is None and self._subset=='train'))

      if should_jitter:
        flip = 'flip' in self._augmentations
        swap = 'swap' in self._augmentations
        if landmarks is not None:
          image, future_image, landmarks, future_landmarks = self._jitter_im_and_points(
            image, future_image, landmarks, future_landmarks, flip=flip,
            swap=swap)
        else:
          image, future_image = self._jitter_im(
            image, future_image, flip=flip, swap=swap)

      inputs = {k: inputs[k] for k in self._get_sample_dtype().keys()}
      inputs.update({'image': image, 'future_image': future_image})
      if landmarks is not None:
        inputs.update({'landmarks': landmarks, 'future_landmarks': future_landmarks})
    return inputs


  def get_dataset(self, batch_size, repeat=False, shuffle=False,
                  num_preprocess_threads=12, keep_aspect=True):
    """
    Returns a tf.Dataset object which iterates over samples.
    """
    def sample_generator():
      return self.sample_image_pair()

    sample_dtype = self._get_sample_dtype()
    sample_shape = self._get_sample_shape()
    dataset = tf.data.Dataset.from_generator(
      sample_generator, sample_dtype, sample_shape)
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(2000)
    dataset = dataset.map(self._proc_im_pair, num_parallel_calls=num_preprocess_threads)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


  def _get_sample_shape(self):
    return {k: None for k in self._get_sample_dtype().keys()}


  @abstractmethod
  def _get_sample_dtype(self):
    """
    Return a dict with the same keys as from ``sample_image_pair``,
    with their tensorflow-datatypes specified.

    'image', 'future_image': can be tf.uint8 (image-tensors)
                                or, tf.string (file-names)
    """
    pass


  @abstractmethod
  def sample_image_pair(self):
    """
    Generator. Returns a dictionary with sampled image and bbox pairs.

    with keys:
      'image', 'future_image', 'bbox', 'future_bbox'.
    """
    pass

  @abstractmethod
  def num_samples(self):
    """
    returns the number of samples per self.SUBSET.
    """
    pass
