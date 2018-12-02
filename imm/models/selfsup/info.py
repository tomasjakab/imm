"""
Code for colorization network adapted from
Colorization as a Proxy Task for Visual Understanding, Larsson, Maire, Shakhnarovich, CVPR 2017
https://github.com/gustavla/self-supervision
"""

from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import sys
from . import printing


def create(scale_summary=False):
    info = {
        'activations': OrderedDict(),
        'init': OrderedDict(),
        'config': dict(return_weights=False),
        'weights': OrderedDict(),
        'vars': OrderedDict(),
    }
    if scale_summary:
        info['scale_summary'] = True
    return info


def print_init(info):
    for k, v in info['init'].items():
        if v.startswith('file'):
            v = printing.paint(v, 'green')
        else:
            v = printing.paint(v, 'red')
        print('{:20s}{}'.format(k, v))
