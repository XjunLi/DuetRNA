"""Time sampling and geodesic integration shared by DuetRNA's two frames."""

import torch
from scipy.spatial.transform import Rotation

from duetrna.geometry import so3 as so3_utils

def _centered_gaussian(num_batch, num_res, device):
    # returns randomly-sampled zero-centered Gaussian noise for translations
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    # returns uniformly-sampled rotations (from U(SO(3)))
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(), # random rotation
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

class Interpolant:
    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._sample_cfg = cfg.sampling

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """Sample training time inside the configured endpoint margin."""
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        """
        Perform Euler ODESolve step for translations
        """
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        """
        Perform Euler ODESolve step for rotations based on prescribed sampling schedule
        """
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f'Unknown sample schedule {self._rots_cfg.sample_schedule}')

        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)
