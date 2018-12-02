"""A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""
import numpy as np
import matplotlib.colors as colors


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight = False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    # Import six here so that `utils` has no import-time dependencies.
    # We want this since we use `utils` during our import-time sanity checks
    # that verify that our dependencies (including six) are actually present.
    import six

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(six.u(str(num)))
    if bold: attr.append(six.u('1'))
    attrs = six.u(';').join(attr)
    return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)

def green(s):
  return colorize(s,'green',bold=True)

def blue(s):
  return colorize(s,'blue',bold=True)

def red(s):
  return colorize(s,'red',bold=True)

def magenta(s):
  return colorize(s,'magenta',bold=True)

def colorize_mat(mat,hsv):
    """
    Colorizes the values in a 2D matrix MAT
    to the color as defined by the color HSV.
    The values in the matrix modulate the 'V' (or value) channel.
    H,S (hue and saturation) are held fixed.

    HSV values are assumed to be in range [0,1].

    Returns an uint8 'RGB' image.
    """
    mat = mat.astype(np.float32)
    m,M = np.min(mat), np.max(mat)
    v = (mat - m) / (M-m)
    h,s = hsv[0] * np.ones_like(v), hsv[1]*np.ones_like(v)
    hsv = np.dstack([h,s,v])
    rgb = (255 * colors.hsv_to_rgb(hsv)).astype(np.uint8)
    return rgb


