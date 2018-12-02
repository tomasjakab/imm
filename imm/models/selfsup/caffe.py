"""
Code for colorization network adapted from
Colorization as a Proxy Task for Visual Understanding, Larsson, Maire, Shakhnarovich, CVPR 2017
https://github.com/gustavla/self-supervision
"""

from .util import DummyDict
from .util import tprint
import deepdish as dd
import numpy as np

# CAFFE WEIGHTS: O x I x H x W
# TFLOW WEIGHTS: H x W x I x O

def to_caffe(tfW, name=None, shape=None, color_layer='', conv_fc_transitionals=None, info=DummyDict()):
    assert conv_fc_transitionals is None or name is not None
    if tfW.ndim == 4:
        if (name == 'conv1_1' or name == 'conv1' or name == color_layer) and tfW.shape[2] == 3:
            tfW = tfW[:, :, ::-1]
            info[name] = 'flipped'
        cfW = tfW.transpose(3, 2, 0, 1)
        return cfW
    else:
        if conv_fc_transitionals is not None and name in conv_fc_transitionals:
            cf_shape = conv_fc_transitionals[name]
            tf_shape = (cf_shape[2], cf_shape[3], cf_shape[1], cf_shape[0])
            cfW = tfW.reshape(tf_shape).transpose(3, 2, 0, 1).reshape(cf_shape[0], -1)
            info[name] = 'fc->c transitioned with caffe shape {}'.format(cf_shape)
            return cfW
        else:
            return tfW.T


def from_caffe(cfW, name=None, color_layer='', conv_fc_transitionals=None, info=DummyDict()):
    assert conv_fc_transitionals is None or name is not None
    if cfW.ndim == 4:
        tfW = cfW.transpose(2, 3, 1, 0)
        assert conv_fc_transitionals is None or name is not None
        if (name == 'conv1_1' or name == 'conv1' or name == color_layer) and tfW.shape[2] == 3:
            tfW = tfW[:, :, ::-1]
            info[name] = 'flipped'
        return tfW
    else:
        if conv_fc_transitionals is not None and name in conv_fc_transitionals:
            cf_shape = conv_fc_transitionals[name]
            tfW = cfW.reshape(cf_shape).transpose(2, 3, 1, 0).reshape(-1, cf_shape[0])
            info[name] = 'c->fc transitioned with caffe shape {}'.format(cf_shape)
            return tfW
        else:
            return cfW.T


def load_caffemodel(path, session, prefix='', ignore=set(),
                    conv_fc_transitionals=None, renamed_layers=DummyDict(),
                    color_layer='', verbose=False, pre_adjust_batch_norm=False):
    import tensorflow as tf
    def find_weights(name, which='weights'):
        for tw in tf.trainable_variables():
            if tw.name.split(':')[0] == name + '/' + which:
                return tw
        return None

    """
    def find_batch_norm(name, which='mean'):
        for tw in tf.all_variables():
            if tw.name.endswith(name + '/bn_' + which + ':0'):
                return tw
        return None
    """

    data = dd.io.load(path, '/data')

    assigns = []
    loaded = []
    info = {}
    for key in data:
        local_key = prefix + renamed_layers.get(key, key)
        if key not in ignore:
            bn_name = 'batch_' + key
            if '0' in data[key]:
                weights = find_weights(local_key, 'weights')

                if weights is not None:
                    W = from_caffe(data[key]['0'], name=key, info=info,
                                   conv_fc_transitionals=conv_fc_transitionals,
                                   color_layer=color_layer)
                    if W.ndim != weights.get_shape().as_list():
                        W = W.reshape(weights.get_shape().as_list())

                    init_str = ''
                    if pre_adjust_batch_norm and bn_name in data:
                        bn_data = data[bn_name]
                        sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
                        W /= sigma
                        init_str += ' batch-adjusted'

                    assigns.append(weights.assign(W))
                    loaded.append('{}:0 -> {}:weights{} {}'.format(key, local_key, init_str, info.get(key, '')))

            if '1' in data[key]:
                biases = find_weights(local_key, 'biases')
                if biases is not None:
                    bias = data[key]['1']

                    init_str = ''
                    if pre_adjust_batch_norm and bn_name in data:
                        bn_data = data[bn_name]
                        sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
                        mu = bn_data['0'] / bn_data['2']
                        bias = (bias - mu) / sigma
                        init_str += ' batch-adjusted'

                    assigns.append(biases.assign(bias))
                    loaded.append('{}:1 -> {}:biases{}'.format(key, local_key, init_str))

            # Check batch norm and load them (unless they have been folded into)
            #if not pre_adjust_batch_norm:

    session.run(assigns)
    if verbose:
        tprint('Loaded model from', path)
        for l in loaded:
            tprint('-', l)
    return loaded


def save_caffemodel(path, session, layers, prefix='',
                    conv_fc_transitionals=None, color_layer='', verbose=False,
                    save_batch_norm=False, lax_naming=False):
    import tensorflow as tf
    def find_weights(name, which='weights'):
        for tw in tf.trainable_variables():
            if lax_naming:
                ok = tw.name.split(':')[0].endswith(name + '/' + which)
            else:
                ok = tw.name.split(':')[0] == name + '/' + which
            if ok:
                return tw
        return None

    def find_batch_norm(name, which='mean'):
        for tw in tf.all_variables():
            #if name + '_moments' in tw.name and tw.name.endswith(which + '/batch_norm:0'):
            if tw.name.endswith(name + '/bn_' + which + ':0'):
                return tw
        return None

    data = {}
    saved = []
    info = {}
    for lay in layers:
        if isinstance(lay, tuple):
            lay, p_lay = lay
        else:
            p_lay = lay

        weights = find_weights(prefix + p_lay, 'weights')
        d = {}
        if weights is not None:
            tfW = session.run(weights)
            cfW = to_caffe(tfW, name=lay,
                           conv_fc_transitionals=conv_fc_transitionals,
                           info=info, color_layer=color_layer)
            d['0'] = cfW
            saved.append('{}:weights -> {}:0  {}'.format(prefix + p_lay, lay, info.get(lay, '')))

        biases = find_weights(prefix + p_lay, 'biases')
        if biases is not None:
            b = session.run(biases)
            d['1'] = b
            saved.append('{}:biases -> {}:1'.format(prefix + p_lay, lay))

        if d:
            data[lay] = d

        if save_batch_norm:
            mean = find_batch_norm(lay, which='mean')
            variance = find_batch_norm(lay, which='var')

            if mean is not None and variance is not None:
                d = {}
                d['0'] = np.squeeze(session.run(mean))
                d['1'] = np.squeeze(session.run(variance))
                d['2'] = np.array([1.0], dtype=np.float32)

                data['batch_' + lay] = d

                saved.append('batch_norm({}) saved'.format(lay))

    dd.io.save(path, dict(data=data), compression=None)
    if verbose:
        tprint('Saved model to', path)
        for l in saved:
            tprint('-', l)
    return saved
