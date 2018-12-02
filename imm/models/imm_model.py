# ==========================================================
# Author:  Ankush Gupta, Tomas Jakab
# ==========================================================
"""
Class for IMM models.
"""

from __future__ import division

import tensorflow as tf
import numpy as np
from collections import defaultdict

from ..models.base_model import BaseModel
from ..models.selfsup.build_vgg16 import build_vgg16
from ..utils import utils as utils
from ..tf_utils.op_utils import dev_wrap
from ..tf_utils import op_utils


def image_summary(name, tensor, train_outputs=1, test_outputs=2):
  tf.summary.image(name, tensor, max_outputs=train_outputs, family='train')
  tf.summary.image(name, tensor, max_outputs=test_outputs, family='test',
                   collections=['test_summaries'])


def metrics_summary(name, metric_fn, **metric_kwargs):
  metric, _, _ = op_utils.create_reset_metric(
    metric_fn, updates_collections=['metrics_update'],
    reset_collections=['metrics_reset'], **metric_kwargs)
  tf.summary.scalar(name, metric, collections=['metrics_summaries'], family='test')


def get_gaussian_maps(mu, shape_hw, inv_std, mode='ankush'):
  """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """
  with tf.name_scope(None, 'gauss_map', [mu]):
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

    x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))

  if mode in ['rot', 'flat']:
    mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

    y = tf.reshape(y, [1, 1, shape_hw[0], 1])
    x = tf.reshape(x, [1, 1, 1, shape_hw[1]])

    g_y = tf.square(y - mu_y)
    g_x = tf.square(x - mu_x)
    dist = (g_y + g_x) * inv_std**2

    if mode == 'rot':
      g_yx = tf.exp(-dist)
    else:
      g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

  elif mode == 'ankush':
    y = tf.reshape(y, [1, 1, shape_hw[0]])
    x = tf.reshape(x, [1, 1, shape_hw[1]])

    g_y = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_y - y) * inv_std)))
    g_x = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_x - x) * inv_std)))

    g_y = tf.expand_dims(g_y, axis=3)
    g_x = tf.expand_dims(g_x, axis=2)
    g_yx = tf.matmul(g_y, g_x)  # [B, NMAPS, H, W]

  else:
    raise ValueError('Unknown mode: ' + str(mode))

  g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
  return g_yx


def colorize_landmark_maps(maps):
  """
  Given BxHxWxN maps of landmarks, returns an aggregated landmark map
  in which each landmark is colored randomly. BxHxWxN
  """
  n_maps = maps.shape.as_list()[-1]
  # get n colors:
  colors = utils.get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(colors[i], [1, 1, 1, 3])
           for i in xrange(n_maps)]
  return tf.reduce_max(hmaps, axis=0)



class IMMModel(BaseModel):

  def __init__(self, config, global_step=None, dtype=tf.float32, name='IMMModel'):
    super(IMMModel, self).__init__(dtype, name)
    self._config = config
    self._global_step = global_step


  def conv(self, x, filters, kernel_size, opts, stride=1, batch_norm=True,
           activation=tf.nn.relu, var_device='/cpu:0', name=None):
    x = self.conv_block(opts, x, kernel_size, filters, stride=(1, stride, stride, 1),
                        padding='SAME', batch_norm=batch_norm,
                        activation=activation, var_device=var_device, name=name)
    return x


  def _colorization_reconstruction_loss(
      self, gt_image, pred_image, training_pl, loss_mask=None):
    """
    Returns "perceptual" loss between a ground-truth image, and the
    corresponding generated image.
    Uses pre-trained VGG-16 for cacluating the features.

    *NOTE: Important to note that it assumes that the images are float32 tensors
           with values in [0,255], and 3 channels (RGB).

    Follows "Photographic Image Generation".
    """
    with tf.variable_scope('SelfSupReconstructionLoss'):
      pretrained_file = self._config.perceptual.net_file
      names = self._config.perceptual.comp
      ims = tf.concat([gt_image, pred_image], axis=0)
      feats = build_vgg16(ims, pretrained_file=pretrained_file)
      feats = [feats[k] for k in names]
      feat_gt, feat_pred = zip(*[tf.split(f, 2, axis=0) for f in feats])

      ws = [100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
      f_e = tf.square if self._config.perceptual.l2 else tf.abs

      if loss_mask is None:
        loss_mask = lambda x: x

      losses = []
      n_feats = len(feats)
      # n_feats = 3
      # wl = [self._exp_running_avg(losses[k], training_pl, init_val=ws[k], name=names[k]) for k in range(n_feats)]

      for k in range(n_feats):
        l = f_e(feat_gt[k] - feat_pred[k])
        wl = self._exp_running_avg(tf.reduce_mean(loss_mask(l)), training_pl, init_val=ws[k], name=names[k])
        l /= wl

        l = tf.reduce_mean(loss_mask(l))
        losses.append(l)

      loss = 1000.0*tf.add_n(losses)
    return loss


  def simple_renderer(self, feat_heirarchy, training_pl, n_final_out=3, final_res=128, var_device='/cpu:0'):
    with tf.variable_scope('renderer'):
      opts = self._get_opts(training_pl)

      filters = self._config.n_filters_render * 8
      batch_norm = True

      x = feat_heirarchy[16]

      size = x.shape.as_list()[1:3]
      conv_id = 1
      while size[0] <= final_res:
        x = self.conv(x, filters, [3, 3], opts, stride=1, batch_norm=batch_norm,
                      var_device=var_device, name='conv_%d'%conv_id)
        if size[0]==final_res:
          x = self.conv(x, n_final_out, [3, 3], opts, stride=1, batch_norm=False,
                        var_device=var_device, activation=None, name='conv_%d'%(conv_id+1))
          break
        else:
          x = self.conv(x, filters, [3, 3], opts, stride=1, batch_norm=batch_norm,
                        var_device=var_device, name='conv_%d'%(conv_id+1))
          x = tf.image.resize_images(x, [2 * s for s in size])
        size = x.shape.as_list()[1:3]
        conv_id += 2
        if filters >= 8: filters /= 2
    return x


  def encoder(self, x, training_pl, var_device='/cpu:0'):
    with tf.variable_scope('encoder'):
      batch_norm = True
      filters = self._config.n_filters

      block_features = []

      opts = self._get_opts(training_pl)
      x = self.conv(x, filters, [7, 7], opts, stride=1, batch_norm=batch_norm,
                    var_device=var_device, name='conv_1')
      x = self.conv(x, filters, [3, 3], opts, stride=1, batch_norm=batch_norm,
                    var_device=var_device, name='conv_2')
      block_features.append(x)

      filters *= 2
      x = self.conv(x, filters, [3, 3], opts, stride=2, batch_norm=batch_norm,
                    var_device=var_device, name='conv_3')
      x = self.conv(x, filters, [3, 3], opts, stride=1, batch_norm=batch_norm,
                    var_device=var_device, name='conv_4')
      block_features.append(x)

      filters *= 2
      x = self.conv(x, filters, [3, 3], opts, stride=2, batch_norm=batch_norm,
                    var_device=var_device, name='conv_5')
      x = self.conv(x, filters, [3, 3], opts, stride=1, batch_norm=batch_norm,
                    var_device=var_device, name='conv_6')
      block_features.append(x)

      filters *= 2
      x = self.conv(x, filters, [3, 3], opts, stride=2, batch_norm=batch_norm,
                    var_device=var_device, name='conv_7')
      x = self.conv(x, filters, [3, 3], opts, stride=1, batch_norm=batch_norm,
                    var_device=var_device, name='conv_8')
      block_features.append(x)

      return block_features


  def image_encoder(self, x, training_pl, filters=64,
                    var_device='/cpu:0'):
    """
    Image encoder
    """
    with tf.variable_scope('image_encoder'):
      opts = self._get_opts(training_pl)
      block_features = self.encoder(x, training_pl, var_device=var_device)
      # add input image to supply max resulution features
      block_features = [x] + block_features
      return block_features


  def pose_encoder(self, x, training_pl, n_maps=1, filters=32,
                   gauss_mode='ankush', map_sizes=None,
                   reuse=False, var_device='/cpu:0'):
    """
    Regresses a N_MAPSx2 (2 = (row, col)) tensor of gaussian means.
    These means are then used to generate 2D "heat-maps".
    Standard deviation is assumed to be fixed.
    """
    with tf.variable_scope('pose_encoder', reuse=reuse):
      opts = self._get_opts(training_pl)
      block_features = self.encoder(x, training_pl, var_device=var_device)
      x = block_features[-1]

      xshape = x.shape.as_list()
      x = self.conv(x, n_maps, [1, 1], opts, stride=1, batch_norm=False,
                     var_device=var_device, activation=None, name='conv_1')

      tf.add_to_collection('tensors', ('heatmaps', x))

      def get_coord(other_axis, axis_size):
        # get "x-y" coordinates:
        g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
        g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
        coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
        coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
        g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
        return g_c, g_c_prob

      xshape = x.shape.as_list()
      gauss_y, gauss_y_prob = get_coord(2, xshape[1])  # B,NMAP
      gauss_x, gauss_x_prob = get_coord(1, xshape[2])  # B,NMAP
      gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)

      tf.add_to_collection('tensors', ('gauss_y_prob', gauss_y_prob))
      tf.add_to_collection('tensors', ('gauss_x_prob', gauss_x_prob))

      gauss_xy = []
      for map_size in map_sizes:
        gauss_xy_ = get_gaussian_maps(gauss_mu, [map_size, map_size],
                                      1.0 / self._config.gauss_std,
                                      mode=gauss_mode)
        gauss_xy.append(gauss_xy_)

      return gauss_mu, gauss_xy


  def model(self, im, future_im, image_encoder, pose_encoder, renderer):
    """
    Inputs IM, FUTURE_IM are shaped: [N x H x W x C]
    """
    with tf.variable_scope('model'):
      im_dev, pose_dev, render_dev = None, None, None
      if hasattr(self._config, 'split_gpus'):
        if self._config.split_gpus:
          im_dev = self._config.devices.image_encoder
          pose_dev = self._config.devices.pose_encoder
          render_dev = self._config.devices.renderer

      max_size = future_im.shape.as_list()[1:3]
      assert max_size[0] == max_size[1]
      max_size = max_size[0]

      # determine the sizes for the renderer
      render_sizes = []
      size = max_size
      stride = self._config.renderer_stride
      while True:
        render_sizes.append(size)
        if size <= self._config.min_res:
          break
        size = size // stride
      # assert render_sizes[-1] == 4

      embeddings = dev_wrap(lambda: image_encoder(im), im_dev)
      gauss_pt, pose_embeddings = dev_wrap(
        lambda: pose_encoder(future_im, map_sizes=render_sizes, reuse=False), pose_dev)

      # create joint embeddings corresponding to renderer sizes
      def group_by_size(embeddings):
        # process image embeddings
        grouped_embeddings = defaultdict(list)
        for embedding in embeddings:
          size = embedding.shape.as_list()[1:3]
          assert size[0] == size[1]
          size = int(size[0])
          grouped_embeddings[size].append(embedding)
        return grouped_embeddings

      grouped_embeddings = group_by_size(embeddings)

      # downsample
      for render_size in render_sizes:
        if render_size not in grouped_embeddings:
          # find closest larger size and resize
          embedding_size = None
          embedding_sizes = sorted(list(grouped_embeddings.keys()))
          for embedding_size in embedding_sizes:
            if embedding_size >= render_size:
              break
          resized_embeddings = []
          for embedding in grouped_embeddings[embedding_size]:
            resized_embeddings.append(tf.image.resize_bilinear(embedding, [render_size, render_size], align_corners=True))
          grouped_embeddings[render_size] += resized_embeddings

      # process pose embeddings
      grouped_pose_embeddings = group_by_size(pose_embeddings)

      # concatenate embeddings
      joint_embeddings = {}
      for rs in render_sizes:
        joint_embeddings[rs] = tf.concat(
          grouped_embeddings[rs] + grouped_pose_embeddings[rs], axis=-1)

      future_im_pred = dev_wrap(lambda: renderer(joint_embeddings), render_dev)

      workaround_channels = 0
      if hasattr(self._config, 'channels_bug_fix'):
        if self._config.channels_bug_fix:
          workaround_channels = len(self._config.perceptual.comp)

      color_channels = future_im_pred.shape.as_list()[3] - workaround_channels
      future_im_pred_mu, _ = tf.split(
          future_im_pred, [color_channels, workaround_channels], axis=3)

      return future_im_pred_mu, gauss_pt, pose_embeddings


  def loss(self, future_im_pred, future_im,
           future_yx, future_yx_gmaps,
           costs_collection, training_pl, loss_mask=None):
    loss_dev = None

    if self._config.loss_mask:
      if loss_mask is not None:
        loss_mask = loss_mask
      else:
        raise RuntimeError('No loss mask recieved but is required.')
    else:
      loss_mask = None

    if loss_mask is None:
      loss_mask = lambda x: x

    w_reconstruct = 1.0/(255.0)# ** 2)
    if self._config.reconstruction_loss == 'perceptual':
      if hasattr(self._config, 'split_gpus'):
        if self._config.split_gpus:
          loss_dev = self._config.devices.loss
      w_reconstruct = 1.0
      reconstruction_loss = dev_wrap(
          lambda: self._colorization_reconstruction_loss(future_im, future_im_pred, training_pl, loss_mask=loss_mask), loss_dev)

    elif self._config.reconstruction_loss == 'l2':
      l = tf.square(future_im_pred - future_im)
      reconstruction_loss = 1000*tf.reduce_mean(loss_mask(l))
    else:
      raise ValueError('Reconsutruction loss-type: '+self._config.reconstruction_loss + ' not understood')
    self._add_cost_summary(reconstruction_loss, 'reconstruction_loss')

    metrics_summary('reconstruction_metric', tf.metrics.mean,
                    values=reconstruction_loss)

    weights_loss = self._decay()
    self._add_cost_summary(weights_loss, 'weights_loss')

    # sum up the losses:
    loss = w_reconstruct * reconstruction_loss
    loss += weights_loss

    self._add_cost_summary(loss,'loss_total')
    tf.add_to_collection(costs_collection, loss)

    return loss


  def _loss_mask(self, map, mask):
    mask = tf.image.resize_images(mask, map.shape.as_list()[1:3])
    return map * mask


  def build(self, inputs, training_pl,
            costs_collection='costs', scope=None,
            var_device='/cpu:0', output_tensors=False, build_loss=True):
    """
    Note the ground truth labels are not used for supervision, but only for monitoring
    the accuracy during training.
    """
    im, future_im = inputs['image'], inputs['future_image']

    if 'mask' in inputs:
      loss_mask = lambda x: self._loss_mask(x, inputs['mask'])
    else:
      loss_mask = None

    n_maps = self._config.n_maps
    gauss_mode = self._config.gauss_mode
    filters = self._config.n_filters

    future_im_size = future_im.shape.as_list()[1:3]
    assert future_im_size[0] == future_im_size[1]
    future_im_size = future_im_size[0]

    image_encoder = lambda x: self.image_encoder(
      x, training_pl, filters=filters)

    pose_encoder = lambda x, map_sizes, reuse: self.pose_encoder(
        x, training_pl, filters=filters, n_maps=n_maps,
        gauss_mode=gauss_mode, map_sizes=map_sizes, reuse=reuse)

    # get the number of output channels based on the loss:
    n_renderer_channels = 3

    workaround_channels = 0
    if hasattr(self._config, 'channels_bug_fix'):
      if self._config.channels_bug_fix:
        workaround_channels = len(self._config.perceptual.comp)

    renderer = lambda x: self.simple_renderer(
      x, training_pl,
      n_final_out=n_renderer_channels + workaround_channels,
      final_res=future_im_size)

    # visualize the inputs:
    image_summary('future_im', future_im)
    image_summary('im', im)

    # build the model:
    future_im_pred, gauss_yx, pose_embeddings = self.model(
      im, future_im, image_encoder, pose_encoder, renderer)

    # visualize the predicted landmarks:
    pose_embed_agg = colorize_landmark_maps(pose_embeddings[0])
    image_summary('pose_embedding', pose_embed_agg)

    future_im_pred_clip = tf.clip_by_value(future_im_pred, 0, 255)
    image_summary('future_im_pred', future_im_pred_clip)

    loss = None
    if build_loss:
      if loss_mask:
        image_summary('mask', inputs['mask'])

      # compute the losses:
      loss = self.loss(future_im_pred, future_im,
                      gauss_yx, pose_embeddings,
                      costs_collection, training_pl, loss_mask=loss_mask)

    tensors = {}
    tensors.update(inputs)
    tensors.update({'future_im': future_im, 'im': im,
                    'pose_embedding': pose_embed_agg,
                    'future_im_pred': future_im_pred,
                    'gauss_yx': gauss_yx})

    if output_tensors:
      return None, loss, self._avg_ops, tensors
    else:
      return None, loss, self._avg_ops
