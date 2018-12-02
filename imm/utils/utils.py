"""
Utility functions.

Author: Ankush Gupta
Date: 29 Jan, 2017
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import itertools
import random

def softmax(x,temp=1.0,axis=-1):
  """
  Softmax of x in python.
  """
  xt = x / temp
  e_x = np.exp(xt - np.max(xt,axis=axis,keepdims=True))
  d = np.sum(e_x,axis=axis,keepdims=True)
  return e_x / d

def sigmoid(x):
  """
  Element wise sigmoid.
  """
  return 1.0 / (1.0 + np.exp(-x))

def one_hot(sym,d_embed,dtype=np.float32):
  """
  Takes a D-dimensional tensor SYM
  and returns a one hot encoded D+1 dimensional
  tensor, with the (D+1)^th dimension equal to D_EMBED.

  Classic:
  http://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
  """
  idx = np.arange(d_embed)
  return (idx == sym[...,None]).astype(dtype)


def center_im_1HW1(image,tol=1e-8):
  """
  Center a tensor: subtract mean, divide by std.
  """
  mu,v = tf.nn.moments(tf.reshape(image,[-1]),[0])
  v = tf.rsqrt(tf.abs(tf.add(v,tol)))
  return tf.mul(tf.sub(image,mu),v)


def get_numeric_shape(t):
  """
  Get the HW of the tensor by adding
  ones of size IM (useful to find shapes
  of tensors with unknown shapes at
  graph construction time).
  """
  o = tf.ones_like(t, dtype=tf.int32)
  ds = [tf.unique(tf.reshape(tf.reduce_sum(o,reduction_indices=[i]),[-1]))[0][0] for i in range(o.get_shape().ndims)]
  return ds

def get_algebra_size(t):
  return tf.stop_gradient(tf.reduce_sum(tf.maximum(tf.abs(tf.sign(t)),1)))

def get_coordinates_padding(hw,dtype=tf.float32):
  """
  Returns extra x,y channels in the shape of the feature F (size = [B,H,W,C]).
  """
  f_h, f_w = hw
  # x-coordinates:
  x_c = tf.reshape(tf.cast(tf.linspace(-1.0,1.0,f_w),dtype),[1,1,f_w,1])
  x_c = tf.tile(x_c,[1,f_h,1,1])
  # y-coodinates:
  y_c = tf.reshape(tf.cast(tf.linspace(-1.0,1.0,f_h),dtype),[1,f_h,1,1])
  y_c = tf.tile(y_c,[1,1,f_w,1])
  # concate with images:
  xy = tf.concat([x_c,y_c], axis=3)
  return xy

def same_words(s1,s2):
  """
  Checks if strings S1 and S2 have the same "words"
  i.e.: Ignores the spaces in matching the two strings.
  """
  s1,s2 = s1.strip(), s2.strip()
  return ' '.join(s1.split()) == ' '.join(s2.split())

def dedup(t,v):
  """this works"""
  t_dtype = t.dtype
  t,v = tf.cast(t,tf.float32), tf.cast(v,tf.float32)
  init_seq = tf.constant([],dtype=tf.float32)

  def collapse(seq,i_s):
    i,s = i_s[0], i_s[1]
    v1 = tf.concat(0,[seq,[s]])
    is_dup = tf.logical_and(tf.reduce_all(tf.equal(seq[-1:],s)),tf.equal(s,v))
    dedup_val = tf.cond(is_dup, lambda: seq, lambda: v1)
    res = tf.cond(tf.reduce_all(tf.equal(i,0)),
                  lambda: v1, lambda: dedup_val)
    return res

  # get the index + values:
  t = tf.reshape(t,[-1,1])
  iter = tf.reshape(tf.cast(tf.range(tf.size(t)),tf.float32),[-1,1])
  elems = tf.concat(1,[iter,t])

  out = tf.foldl(collapse,elems=elems,initializer=init_seq,back_prop=False)
  out = tf.cast(out,t_dtype)

  return out


def split_tensors(ts, num_splits, axis=0):
  """
  Splits a nested structure of tensors TS, into
  NUM_SPLITS along the AXIS dimension.
  """
  ts_flat = nest.flatten(ts)
  splits = [tf.split(t,num_splits,axis=axis) for t in ts_flat]
  splits = [nest.pack_sequence_as(ts,[s[i] for s in splits]) for i in range(num_splits)]
  return splits

def merge_tensors(ts_split, axis=0):
  """
  Merge a nested structure of tensors TS, along the dimension DIM.
  """
  ts_flat = [nest.flatten(si) for si in ts_split]
  ts_merged = [tf.concat([s[i] for i in xrange(len(ts_flat))], axis=dim) for s in ts_flat]
  return nest.pack_sequence_as(ts_split[0], ts_merged)

# def dedup(t,v):
#   """
#   Removes repeated occurences of values v in t (one-dimensional / flattened).
#   """
#   with tf.variable_scope('dedup'):
#     t_dtype = t.dtype
#     t,v = tf.cast(t,tf.float32), tf.cast(v,tf.float32)
#     v_id = tf.reshape(tf.concat(0,[[1.0],tf.cast(tf.equal(t,v),tf.float32),[1.0]]),[1,-1,1])
#     # edge-detection, for finding the extents of the substrings :
#     start_id = tf.where(tf.equal(tf.reshape(tf.nn.conv1d(v_id,tf.reshape([1.,-1.],[2,1,1]),1,'VALID'),[-1]),1))
#     end_id = tf.where(tf.equal(tf.reshape(tf.nn.conv1d(v_id,tf.reshape([-1.,1.],[2,1,1]),1,'VALID'),[-1]),1))
#     # now join back the contiguous sub-arrays:
#     init_seq = tf.constant([],dtype=tf.float32)
#     iter = tf.cast(tf.reshape(tf.range(tf.size(start_id)),[-1,1]),tf.int64)
#     elems = tf.concat(1,[iter,start_id,end_id-start_id])
#     def concat(seq,i_s_e):
#       i,s,e = i_s_e[0],i_s_e[1],i_s_e[2]
#       subseq = tf.slice(t,[s],[e])
#       joined_subseq = tf.concat(0,[seq,[v],subseq])
#       out = tf.cond(tf.equal(i,0),lambda:subseq,lambda:joined_subseq)
#       return out
#     out_t = tf.foldl(concat,elems,initializer=init_seq,back_prop=False)
#     out_t = tf.cast(out_t,t_dtype)
#   return out_t


def meshgrid(*args, **kwargs):
  """Broadcasts parameters for evaluation on an N-D grid.
  Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
  of N-D coordinate arrays for evaluating expressions on an N-D grid.
  Notes:
  `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
  When the `indexing` argument is set to 'xy' (the default), the broadcasting
  instructions for the first two dimensions are swapped.
  Examples:
  Calling `X, Y = meshgrid(x, y)` with the tensors
  ```prettyprint
    x = [1, 2, 3]
    y = [4, 5, 6]
  ```
  results in
  ```prettyprint
    X = [[1, 1, 1],
         [2, 2, 2],
         [3, 3, 3]]
    Y = [[4, 5, 6],
         [4, 5, 6],
         [4, 5, 6]]
  ```
  Args:
    *args: `Tensor`s with rank 1
    indexing: Either 'xy' or 'ij' (optional, default: 'xy')
    name: A name for the operation (optional).
  Returns:
    outputs: A list of N `Tensor`s with rank N
  """
  indexing = kwargs.pop("indexing", "xy")
  name = kwargs.pop("name", "meshgrid")
  if kwargs:
    key = list(kwargs.keys())[0]
    raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))

  if indexing not in ("xy", "ij"):
    raise ValueError("indexing parameter must be either 'xy' or 'ij'")

  with tf.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
      output.append(tf.reshape(tf.expand_dims(x,0), (s0[:i] + (-1,) + s0[i + 1::])) )
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]

    output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype

    if indexing == "xy" and ndim > 1:
      output[0] = tf.reshape(output[0], (1, -1) + (1,)*(ndim - 2))
      output[1] = tf.reshape(output[1], (-1, 1) + (1,)*(ndim - 2))
      shapes[0], shapes[1] = shapes[1], shapes[0]

    mult_fact = tf.ones(shapes, output_dtype)
    return [x * mult_fact for x in output]


def split_indices(s, c=' '):
  """
  Splits the string S at character C,
  and returns the indices of the contiguous
  sub-strings.
  """
  p = 0
  inds = []
  for k, g in itertools.groupby(s, lambda x:x==c):
    q = p + sum(1 for i in g)
    if not k:
      inds.append((p, q))
    p = q
  return inds


# get "maximally" different random colors:
#  ref: https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor = 0.5):
  return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]


def color_distance(c1,c2):
  return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])


def generate_new_color(existing_colors,pastel_factor = 0.5):
  max_distance = None
  best_color = None
  for i in range(0,100):
    color = get_random_color(pastel_factor = pastel_factor)
    if not existing_colors:
      return color
    best_distance = min([color_distance(color,c) for c in existing_colors])
    if not max_distance or best_distance > max_distance:
      max_distance = best_distance
      best_color = color
  return best_color


def get_n_colors(n, pastel_factor=0.9):
  colors = []
  for i in xrange(n):
    colors.append(generate_new_color(colors,pastel_factor = 0.9))
  return colors


def get_grid(x_range, y_range, nmajor=5, nminor=20):
  """
  Returns 2 lists, corresponding to horizontal and vertical lines,
  each containing NMAJOR elements corresponding NMAJOR lines.
  Each line is represented as a [NMINOR,2] tensor (for x,y-coordinates).
  """
  h_lines = [np.concatenate(np.meshgrid(np.linspace(x_range[0], x_range[1], nminor), y),
                            axis=0).T for y in np.linspace(y_range[0], y_range[1], nmajor)]
  v_lines = [np.concatenate(np.meshgrid(x, np.linspace(y_range[0], y_range[1], nminor)),
                            axis=1) for x in np.linspace(x_range[0], x_range[1], nmajor)]
  return h_lines, v_lines