"""
Code for colorization network adapted from
Colorization as a Proxy Task for Visual Understanding, Larsson, Maire, Shakhnarovich, CVPR 2017
https://github.com/gustavla/self-supervision
"""

from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf


_tlog_path = None


class DummyDict(object):
    def __init__(self):
        pass
    def __getitem__(self, item):
        return DummyDict()
    def __setitem__(self, item, value):
        return DummyDict()
    def get(self, item, default=None):
        if default is None:
            return DummyDict()
        else:
            return default


def config():
    NUM_THREADS = os.environ.get('OMP_NUM_THREADS')

    config = tf.ConfigProto(
            allow_soft_placement=True,
            )
    config.gpu_options.allow_growth=True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    if NUM_THREADS is not None:
        config.intra_op_parallelism_threads = int(NUM_THREADS)
    return config


def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    return parser


def tlog(path):
    global _tlog_path
    _tlog_path = path


def tprint(*args, **kwargs):
    global _tlog_path
    import datetime
    GRAY = '\033[1;30m'
    RESET = '\033[0m'
    time_str = GRAY+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+RESET
    print(*((time_str,) + args), **kwargs)

    if _tlog_path is not None:
        with open(_tlog_path, 'a') as f:
            nocol_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(*((nocol_time_str,) + args), file=f, **kwargs)


def mkdirs(args):
    for arg in args:
        try:
            os.mkdir(arg)
        except:
            pass
