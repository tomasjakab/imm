# Common ops for tensorflow
#  Author: Ankush Gupta
#  Date: 27 Jun, 2017
from __future__ import division
import tensorflow as tf


def gradient_scale_op(x,grad_scale):
  """
  Scales the gradient (during the backrward pass) of X
  by GRAD_SCALE.
  Returns:
    A tensor, which is identical to X in the forward pass,
    but scales down the gradients during the backward pass.
  """
  scaled_x = grad_scale * x
  x_hat = scaled_x + tf.stop_gradient(x - scaled_x)
  return x_hat


def safe_div(num, denom, name=None):
  """
  Computes a safe divide which returns 0 if the denominator is zero.

  Args:
    num: An arbitrary `Tensor`.
    denom: A `Tensor` whose shape matches `num`.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  with tf.name_scope(name,"safe_div",[num,denom]) as scope:
    d_is_zero = tf.equal(denom, 0)
    d_or_1 = tf.where(d_is_zero,tf.ones_like(denom), denom)
    return tf.where(d_is_zero, tf.zeros_like(num), tf.div(num, d_or_1))


def safe_log(x, name=None):
  """
  Returns the log of 'X' if positive, else 0 (if x <= 0).

  Args:
    X: An arbitrary `Tensor`.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  with tf.name_scope(name,"safe_log",[x]) as scope:
    x_is_pos = tf.greater(x, 0)
    x_or_1 = tf.where(x_is_pos,x,tf.ones_like(x))
    return tf.log(x_or_1)


def rand_select(x,f_x,p,name=None):
  """
  Returns F_X (a function) with probabaility P, else returns X itself.
  """
  with tf.name_scope(name,"rand_select",[x,p]) as scope:
    r = tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32)
    is_f = tf.less(r,p)
    return tf.cond(is_f,lambda: f_x(x),lambda: tf.identity(x))


def dev_wrap(fn, dev=None):
  if dev:
    with tf.device(dev):
      x = fn()
  else:
    x = fn()
  return x


def summary_wrap(training_pl, summary_fn, name, *args, **kwargs):
  tf.cond(training_pl,
          lambda: summary_fn('train', *args, **kwargs),
          lambda: summary_fn('test', *args, **kwargs),
          name=name)


def create_reset_metric(metric_fn, scope='reset_metric', reset_collections=None,
                        **metric_kwargs):
  with tf.variable_scope(None, default_name=scope):
    metric_op, update_op = metric_fn(**metric_kwargs)
    variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                  scope=tf.contrib.framework.get_name_scope())
    reset_ops = [v.assign(tf.zeros_like(v)) for v in variables]
    if reset_collections is not None:
      for collection in reset_collections:
        for reset_op in reset_ops:
          tf.add_to_collection(collection, reset_op)
  return metric_op, update_op, reset_op


def check_image(image):
  assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
  with tf.control_dependencies([assertion]):
    image = tf.identity(image)

  if image.get_shape().ndims not in (3, 4):
    raise ValueError("image must be either 3 or 4 dimensions")

  # make the last dimension 3 so that you can unstack the colors
  shape = list(image.get_shape())
  shape[-1] = 3
  image.set_shape(shape)
  return image


def rgb_to_lab(srgb):
  """
  It assumes that the RGB uint8 image has been converted to "float" using:

    tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

  which rescales the values to [0,1] for the float datatype.

  ref: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
  """
  with tf.name_scope("rgb_to_lab"):
    srgb = check_image(srgb)
    srgb_pixels = tf.reshape(srgb, [-1, 3])

    with tf.name_scope("srgb_to_xyz"):
      linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
      exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
      rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
      rgb_to_xyz = tf.constant([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334], # R
        [0.357580, 0.715160, 0.119193], # G
        [0.180423, 0.072169, 0.950227], # B
      ])
      xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    with tf.name_scope("xyz_to_cielab"):
      # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

      # normalize for D65 white point
      xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

      epsilon = 6/29.0
      linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
      exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
      fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

      # convert to lab
      fxfyfz_to_lab = tf.constant([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
      ])
      lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

    return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
  """
  ref: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
  """
  with tf.name_scope("lab_to_rgb"):
    lab = check_image(lab)
    lab_pixels = tf.reshape(lab, [-1, 3])

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    with tf.name_scope("cielab_to_xyz"):
      # convert to fxfyfz
      lab_to_fxfyfz = tf.constant([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0], # l
        [1/500.0,     0.0,      0.0], # a
        [    0.0,     0.0, -1/200.0], # b
      ])
      fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

      # convert to xyz
      epsilon = 6/29.0
      linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
      exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
      xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

      # denormalize for D65 white point
      xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    with tf.name_scope("xyz_to_srgb"):
      xyz_to_rgb = tf.constant([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434], # x
        [-1.5371385,  1.8760108, -0.2040259], # y
        [-0.4985314,  0.0415560,  1.0572252], # z
      ])
      rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
      # avoid a slightly negative number messing up the conversion
      rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
      linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
      exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
      srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    return tf.reshape(srgb_pixels, tf.shape(lab))
