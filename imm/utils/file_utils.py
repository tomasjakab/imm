import os.path as osp
import os
import glob, re
import fnmatch
import numpy as np
import multiprocessing as mp
import subprocess as sp
import json
import errno
import string
import random

from ..utils.colorize import *


def makedirs(path, exist_ok=False):
  try:
    os.makedirs(path)
  except OSError as e:
    if not exist_ok or e.errno != errno.EEXIST:
      raise e


def get_subdirs(dir):
  """
  Returns all the subdirs in DIR.
  """
  files = os.listdir(dir)
  subdirs = [f for f in files if osp.isdir(f)]
  return subdirs


def get_files(dir):
  """
  Returns all the files in DIR (no subdirs).
  """
  files = os.listdir(dir)
  files = [f for f in files if osp.isfile(f)]
  return files


def recursive_glob(rootdir, pattern='*', match='files'):
  """Search recursively for files matching a specified pattern.
  Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python

  MATCH: in {'files', 'dir'} : matches files or directories respectively
  """
  matches = []
  for root, dirnames, filenames in os.walk(rootdir):
    if match=='files':
      to_match = filenames
    else:
      to_match = dirnames
    for m in fnmatch.filter(to_match, pattern):
      matches.append(os.path.join(root, m))
  return matches


def syscall(cmd, verbose=True):
  if verbose: print(green('sys-cmd: '+cmd))
  os.system(cmd)


def parallel_syscalls(cmds, npool=4):
  """
  CMDS: list of system calls to make.
  NPOOL: size of the multi-processing pool.

  Makes the syscalls in CMDS using NPOOL processes.
  """
  pool = mp.Pool(npool)
  pool.map(syscall, cmds)


def get_video_info(video_file):
  """
  Extracts video information.
  Assumes 'ffprobe' is in PATH.
  """
  if not osp.exists(video_file):
    raise ValueError('File does not exist: '+video_file)
  cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams %s'
  cmd = cmd % video_file.replace(' ', '\ ')
  try:
    vinfo = sp.check_output(cmd, shell=True)
    vinfo = json.loads(vinfo)
  except:
    raise Exception('Error extracting video information for: '+video_file)
  return vinfo


def split_video_into_frames(video_fname, save_dir, file_format='%05d.jpg',
                            quality=5, bbox=None, frame_hw=None, duration=None):
  """
  VIDEO_FNAME: video filename.
  SAVE_DIR: directory to save the frames in.
  FILE_FORMAT: file_format for frames-name.
  QUALITY: a value from 1 to 31 (for jpeg image quality).
  FRAME_HW: frame output size
  BBOX: [ymin, xmin, ymax, xmax]
  """
  out_path = osp.join(save_dir, file_format)
  resize = ''
  crop = ''
  seek = ''
  if bbox:
    out_w, out_h = bbox[3] - bbox[1], bbox[2] - bbox[0]
    x, y = bbox[1], bbox[0]
    crop = ' -filter:v "crop=%d:%d:%d:%d"' % (out_w, out_h, x, y)
  if frame_hw:
    resize = ' -s %dx%d' % (frame_hw[1], frame_hw[0])
  if duration:
    seek = ' -ss 0 -to %d' % duration

  cmd = 'ffmpeg -hide_banner -loglevel panic -i %s' + seek + ' -q:v %d -start_number 0' + crop + resize + ' %s'
  cmd = cmd%(video_fname.replace(' ','\ '), quality, out_path.replace(' ','\ '))
  syscall(cmd)


def get_num_frames(video_file):
  """
  Extracts the number of frames in a video.
  """
  vinfo = get_video_info(video_file)
  return int(vinfo['streams'][0]['nb_frames'])


def extract_frames_from_video(vid_fname, frame_ids=[], frame_hw=None):
  """
  Extract frames (RGB) from videos (using FFMPEG.

  VID_FNAME: path to the video file.
  FRAME_IDS: list of frame-ids to extract (assumed to be) valid (in range).
  FRAME_HW: height, width of the frames in the video.
            If None, H,W are retrieved using ffprobe.

  Returns a 4D numpy uint8 tensor [B,H,W,3], where B == len(FRAME_IDS).
  """
  if frame_hw is None:
    vid_info = get_video_info(vid_fname)
    vid_info = vid_info['streams'][0]
    frame_hw = ( int(vid_info['height']), int(vid_info['width']) )

  cmd = ("ffmpeg -loglevel panic -hide_banner -i %s -f image2pipe -vsync"
         + " vfr -vf select='%s' -pix_fmt rgb24 -vcodec rawvideo -")
  select_frames = '+'.join(['eq(n\,%d)'%fid for fid in frame_ids])
  cmd = cmd % (vid_fname.replace(' ', '\ '), select_frames)

  pipe = sp.Popen(cmd, shell=True, stdout=sp.PIPE, bufsize=10**8)
  n_frames = len(frame_ids)
  frames = np.zeros((n_frames, frame_hw[0], frame_hw[1], 3), dtype=np.uint8)
  for i in xrange(len(frame_ids)):
    raw_image = pipe.stdout.read(frame_hw[0]*frame_hw[1]*3)
    im = np.fromstring(raw_image, dtype='uint8')
    frames[i,...] = im.reshape((frame_hw[0], frame_hw[1], 3))
  pipe.stdout.flush()
  return frames

def get_random_name(len=32):
  return ''.join([random.choice(string.ascii_letters + string.digits) for _ in xrange(len)])

# removes restrictions on subprocessing.map:
# ref: https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
def func_wrap(f, q_in, q_out):
  while True:
    i, x = q_in.get()
    if i is None:
      break
    q_out.put((i, f(x)))

def parmap(f, iterates, nprocs=mp.cpu_count()//2):
  q_in = mp.Queue(1)
  q_out = mp.Queue()
  proc = [mp.Process(target=func_wrap, args=(f, q_in, q_out)) for _ in range(nprocs)]
  for p in proc:
    p.daemon = True
    p.start()
  sent = [q_in.put((i, x)) for i, x in enumerate(iterates)]
  [q_in.put((None, None)) for _ in range(nprocs)]
  res = [q_out.get() for _ in range(len(sent))]
  [p.join() for p in proc]
  return [x for i, x in sorted(res)]
