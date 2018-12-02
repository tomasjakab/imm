# ==========================================================
# Author: Tomas Jakab
# ==========================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def get_marker_style(i, cmap='Dark2'):
  cmap = plt.get_cmap(cmap)
  colors = [cmap(c) for c in np.linspace(0., 1., 8)]
  markers = ['v', 'o', 's', 'd', '^', 'x', '+']
  max_i = len(colors) * len(markers) - 1
  if i > max_i:
    raise ValueError('Exceeded maximum (' + str(max_i) + ') index for styles.')
  c = i % len(colors)
  m = int(i / len(colors))
  return colors[c], markers[m]


def single_marker_style(color, marker):
  return lambda _: (color, marker)


def plot_landmark(ax, landmark, k, size=1.5, zorder=2, cmap='Dark2',
                  style_fn=None):
  if style_fn is None:
    c, m = get_marker_style(k, cmap=cmap)
  else:
    c, m = style_fn(k)
  ax.scatter(landmark[1], landmark[0], c=c, marker=m,
             s=(size * mpl.rcParams['lines.markersize']) ** 2,
             zorder=zorder)


def plot_landmarks(ax, landmarks, size=1.5, zorder=2, cmap='Dark2', style_fn=None):
  for k, landmark in enumerate(landmarks):
    plot_landmark(ax, landmark, k, size=size, zorder=zorder,
                  cmap=cmap, style_fn=style_fn)
