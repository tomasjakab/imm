# ==========================================================
# Author: Ankush Gupta, Tomas Jakab
# ==========================================================
import scipy.spatial.distance as ssd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class TPSRandomSampler(nn.Module):

    def __init__(self, height, width, vertical_points=10, horizontal_points=10,
                 rotsd=0.0, scalesd=0.0, transsd=0.1, warpsd=(0.001, 0.005),
                 cache_size=1000, cache_evict_prob=0.01, pad=True, device=None):
        super(TPSRandomSampler, self).__init__()

        self.input_height = height
        self.input_width = width

        self.h_pad = 0
        self.w_pad = 0
        if pad:
          self.h_pad = self.input_height // 2
          self.w_pad = self.input_width // 2

        self.height = self.input_height + self.h_pad
        self.width = self.input_width + self.w_pad

        self.vertical_points = vertical_points
        self.horizontal_points = horizontal_points

        self.rotsd = rotsd
        self.scalesd = scalesd
        self.transsd = transsd
        self.warpsd = warpsd
        self.cache_size = cache_size
        self.cache_evict_prob = cache_evict_prob

        self.tps = TPSGridGen(
            self.height, self.width, vertical_points, horizontal_points)

        self.cache = [None] * self.cache_size

        self.pad = pad

        self.device = device


    def _sample_grid(self):
        W = sample_tps_w(
            self.vertical_points, self.horizontal_points, self.warpsd,
            self.rotsd, self.scalesd, self.transsd)
        W = torch.from_numpy(W.astype(np.float32))
        # generate grid
        grid = self.tps(W[None])
        return grid


    def _get_grids(self, batch_size):
        grids = []
        for i in range(batch_size):
            entry = random.randint(0, self.cache_size - 1)
            if self.cache[entry] is None or random.random() < self.cache_evict_prob:
                grid = self._sample_grid()
                if self.device is not None:
                    grid = grid.to(self.device)
                self.cache[entry] = grid
            else:
                grid = self.cache[entry]
            grids.append(grid)
        grids = torch.cat(grids)
        return grids


    def forward(self, input):
        if self.device is not None:
            input_device = input.device
            input = input.to(self.device)

        # get TPS grids
        batch_size = input.size(0)
        grids = self._get_grids(batch_size)

        if self.device is None:
            grids = grids.to(input.device)

        input = F.pad(input, (self.h_pad, self.h_pad, self.w_pad,
                               self.w_pad), mode='replicate')
        input = F.grid_sample(input, grids)
        input = F.pad(input, (-self.h_pad, -self.h_pad, -self.w_pad, -self.w_pad))

        if self.device is not None:
            input = input.to(input_device)

        return input


    def forward_py(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input)
            input = input.permute([0, 3, 1, 2])
            input = self.forward(input)
            input = input.permute([0, 2, 3, 1])
            input = input.numpy()
            return input



class TPSGridGen(nn.Module):

  def __init__(self, Ho, Wo, Hc, Wc):
    """
    Ho,Wo: height/width of the output tensor (grid dimensions).
    Hc,Wc: height/width of the control-point grid.

    Assumes for simplicity that the control points lie on a regular grid.
    Can be made more general.
    """
    super(TPSGridGen, self).__init__()

    self._grid_hw = (Ho, Wo)
    self._cp_hw = (Hc, Wc)

    # initialize the grid:
    xx, yy = np.meshgrid(np.linspace(-1, 1, Wo), np.linspace(-1, 1, Ho))
    self._grid = np.c_[xx.flatten(), yy.flatten()].astype(np.float32)  # Nx2
    self._n_grid = self._grid.shape[0]

    # initialize the control points:
    xx, yy = np.meshgrid(np.linspace(-1, 1, Wc), np.linspace(-1, 1, Hc))
    self._control_pts = np.c_[
        xx.flatten(), yy.flatten()].astype(np.float32)  # Mx2
    self._n_cp = self._control_pts.shape[0]

    # compute the pair-wise distances b/w control-points and grid-points:
    Dx = ssd.cdist(self._grid, self._control_pts, metric='sqeuclidean')  # NxM

    # create the tps kernel:
    # real_min = 100 * np.finfo(np.float32).min
    real_min = 1e-8
    Dx = np.clip(Dx, real_min, None)  # avoid log(0)
    Kp = np.log(Dx) * Dx
    Os = np.ones((self._grid.shape[0]))
    L = np.c_[Kp, np.ones((self._n_grid, 1), dtype=np.float32),
              self._grid]  # Nx(M+3)
    self._L = torch.from_numpy(L.astype(np.float32))  # Nx(M+3)


  def forward(self, w_tps):
    """
    W_TPS: Bx(M+3)x2 sized tensor of tps-transformation params.
            here `M` is the number of control-points.
                `B` is the batch-size.

    Returns an BxHoxWox2 tensor of grid coordinates.
    """
    assert w_tps.shape[1] - 3 == self._n_cp
    batch_size = w_tps.shape[0]
    tfm_grid = torch.matmul(self._L, w_tps)
    tfm_grid = tfm_grid.reshape(
        (batch_size, self._grid_hw[0], self._grid_hw[1], 2))
    return tfm_grid



def sample_tps_w(Hc, Wc, warpsd, rotsd, scalesd, transsd):
  """
  Returns randomly sampled TPS-grid params of size (Hc*Wc+3)x2.

  Params:
    WARPSD: 2-tuple
    {ROT/SCALE/TRANS}-SD: 1-tuple of standard devs.
  """
  Nc = Hc * Wc  # no of control-pots
  # non-linear component:
  mask = (np.random.rand(Nc, 2) > 0.5).astype(np.float32)
  W = warpsd[0] * np.random.randn(Nc, 2) + \
      warpsd[1] * (mask * np.random.randn(Nc, 2))
  # affine component:
  rnd = np.random.randn
  rot = np.deg2rad(rnd() * rotsd)
  sc = 1.0 + rnd() * scalesd
  aff = [[transsd*rnd(),      transsd*rnd()],
         [sc * np.cos(rot),   sc * -np.sin(rot)],
         [sc * np.sin(rot),   sc * np.cos(rot)]]
  W = np.r_[W, aff]
  return W
