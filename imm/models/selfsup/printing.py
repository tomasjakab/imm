"""
Code for colorization network adapted from
Colorization as a Proxy Task for Visual Understanding, Larsson, Maire, Shakhnarovich, CVPR 2017
https://github.com/gustavla/self-supervision
"""

from __future__ import division, print_function, absolute_import
import sys
import numpy as np


COLORS = dict(
    black='0;30',
    darkgray='1;30',
    red='1;31',
    green='1;32',
    brown='0;33',
    yellow='1;33',
    blue='1;34',
    purple='1;35',
    cyan='1;36',
    white='1;37',
    reset='0'
)

COLORIZE = sys.stdout.isatty()


def paint(s, color, colorize=COLORIZE):
    if colorize:
        if color in COLORS:
            return '\033[{}m{}\033[0m'.format(COLORS[color], s)
        else:
            raise ValueError('Invalid color')
    else:
        return s


def print_init(info, file=sys.stdout, colorize=COLORIZE):
    print('Initialization overview')
    for k, v in info['init'].items():
        if v == 'file':
            color = 'green'
        elif v == 'init':
            color = 'red'
        else:
            color = 'white'
        print('%30s %s' % (k, paint(v, color=color, colorize=colorize)), file=file)


def histogram(x, bins='auto', columns=40):
    if np.isnan(x).any():
        print("Error: Can't produce histogram when there are NaNs")
        return
    total_count = len(x)
    counts, bins = np.histogram(x, bins=bins, normed=True)
    for i, c in enumerate(counts):
        frac = c
        cols = int(frac * columns)
        bar = '#' * min(60, cols) + ('>' if cols > 60 else '')
        print('[{:6.2f}, {:6.2f}): {}'.format(bins[i], bins[i+1], bar))
