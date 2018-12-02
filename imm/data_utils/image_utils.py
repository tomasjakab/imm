# ==========================================================
# Author: Ankush Gupta
# Date: 23 Aug 2016
# ==========================================================
import tensorflow as tf
import random


def decode_image_buffer(image_buffer, image_format, cast_float=True,
                        channels=3, scope=None):
  """
  Decodes PNG/JPEG images, based on IMAGE_FORMAT.
  """
  # select the decoding function:
  image_format = image_format.lower()
  if 'png' in image_format:
    f_decode = tf.image.decode_png
  elif ('jpg' in image_format) or ('jpeg' in image_format):
    f_decode = tf.image.decode_jpeg
  else:
    raise Exception('Unknown image format: '+image_format)

  # decode:
  with tf.op_scope([image_buffer], scope, 'decode_image_buffer'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = f_decode(image_buffer, channels=channels)
    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    if cast_float:
      image = tf.cast(image,dtype=tf.float32)
  return image


def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  with tf.op_scope([image], scope, 'distort_color'):
    color_ordering = thread_id % 2
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def distort_image(image, im_hw, thread_id=0, scope=None):
  """Distort one image for training a network.
  Distort images for data-augmentation.
  Here image-resizing and color distortion is implemented.

  Args:
    image: 3-D float Tensor of image
    im_hw: Tensor of [HEIGHT,WIDTH] int32
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.op_scope([image, im_hw], scope, 'distort_image'):
    # This resizing operation may distort the images because the aspect
    # ratio is not respected. Note that ResizeMethod contains 4 enumerated resizing methods.
    distorted_image = tf.image.resize_images(image, im_hw)
    # Randomly distort the colors.
    # distorted_image = distort_color(distorted_image, thread_id)
    return distorted_image

def resize_image(image, im_hw, scope=None):
  """Prepare one image for evaluation.
  Args:
    image: 3-D float Tensor
    im_hw: tf.int32 2-length tensor of (height,width)
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.op_scope([image, im_hw], scope, 'resize_image'):
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0) # as we need a 4D tensor for the following op
    image = tf.image.resize_bilinear(image, im_hw, align_corners=False)
    image = tf.squeeze(image, [0])
    return image
