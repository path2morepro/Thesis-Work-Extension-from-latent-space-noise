import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffusion_networks import SongUNet


class CondSampler:
    """
    Conditional flow-matching sampler for p(x_{t+1} | x_t).

    Operates on 2-level SQG fields: all inputs/outputs are (B, 2, H, W).
    The model receives z_s and x_t concatenated channel-wise → (B, 4, H, W).

    Interpolant: z_s = (1-s)*z0 + s*x_{t+1},  s in [0, 1]
    Velocity:    b(z_s, s, x_t) ≈ x_{t+1} - z0
    """

    def __init__(
        self,
        model_path,
        device,
        steps=100,
        invert_steps=500,
        eps=None,
        invert_eps=None,
        debug=False,
    ):
        """
        Args:
            model_path   : path to saved state dict (.pth)
            device       : torch device
            steps        : Euler steps for sampling (forward ODE)
            invert_steps : Euler steps for inversion (backward ODE)
            eps          : callable t -> float, diffusion coefficient for stochastic
                           sampling; None or lambda t: 0 gives deterministic ODE
            invert_eps   : same but for inversion; None gives deterministic inversion
            debug        : if True, sample() returns full trajectory tensor
        """
        self.model = SongUNet(
            img_resolution=64,
            in_channels=4,       # 2 ch (z_s) + 2 ch (x_t conditioning)
            out_channels=2,
            embedding_type='fourier',
            encoder_type='residual',
            decoder_type='standard',
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=32,
            channel_mult=[2, 2, 2],
            attn_resolutions=[32],
            label_dropout=0.1,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.steps = steps
        self.invert_steps = invert_steps
        self.eps = eps if eps is not None else lambda t: 0.0
        self.invert_eps = invert_eps if invert_eps is not None else lambda t: 0.0
        self.debug = debug

    def _score(self, t_val, b, zt):
        """Score estimate s(z_s, t) = (t*b - z_t) / (1 - t) from the velocity field."""
        beta = 1.0 - t_val
        if beta > 1e-8:
            return (t_val * b - zt) / beta
        return torch.zeros_like(b)

    def sample(self, z0, x_t):
        """
        Forward ODE: latent noise z0 → physical forecast x_{t+1}, conditioned on x_t.

        Args:
            z0  : initial Gaussian noise, shape (B, 2, H, W)
            x_t : conditioning previous state, shape (B, 2, H, W)

        Returns:
            predicted x_{t+1} on CPU, shape (B, 2, H, W).
            If debug=True, also returns trajectory tensor (B, steps+1, 2, H, W).
        """
        B = z0.shape[0]
        dt = 1.0 / self.steps
        ts = torch.linspace(0, 1, self.steps + 1, device=self.device)[:-1]

        zt = z0.to(self.device)
        x_t = x_t.to(self.device)

        trajectory = [zt.clone().cpu()] if self.debug else None

        with torch.no_grad():
            enum = tqdm(ts, desc='sample') if self.debug else ts
            for t in enum:
                t_val = t.item()
                eps_t = float(self.eps(t_val))

                s_vec = torch.full((B,), t_val, device=self.device)
                b = self.model(zt, s_vec, class_labels=x_t)

                score = self._score(t_val, b, zt) if eps_t > 0 else 0.0

                dz = (b + score * eps_t) * dt
                dW = (torch.randn_like(zt) * math.sqrt(2.0 * dt * eps_t)
                      if eps_t > 0 else 0.0)
                zt = zt + dz + dW

                if self.debug:
                    trajectory.append(zt.clone().cpu())

        result = zt.cpu()
        if self.debug:
            return result, torch.stack(trajectory, dim=1)
        return result

    def invert(self, x_t1, x_t):
        """
        Reverse ODE: physical field x_{t+1} → latent noise z0, conditioned on x_t.

        Args:
            x_t1 : next state to invert, shape (B, 2, H, W)
            x_t  : conditioning previous state, shape (B, 2, H, W)

        Returns:
            latent noise z0 on CPU, shape (B, 2, H, W)
        """
        B = x_t1.shape[0]
        dt = -1.0 / self.invert_steps
        ts = torch.linspace(1, 0, self.invert_steps + 1, device=self.device)[:-1]

        zt = x_t1.to(self.device)
        x_t = x_t.to(self.device)

        with torch.no_grad():
            enum = tqdm(ts, desc='invert') if self.debug else ts
            for t in enum:
                t_val = t.item()
                eps_t = float(self.invert_eps(t_val))

                s_vec = torch.full((B,), t_val, device=self.device)
                b = self.model(zt, s_vec, class_labels=x_t)

                score = self._score(t_val, b, zt) if eps_t > 0 else 0.0

                dz = (b - score * eps_t) * dt
                dW = (torch.randn_like(zt) * math.sqrt(2.0 * abs(dt) * eps_t)
                      if eps_t > 0 else 0.0)
                zt = zt + dz + dW

        return zt.cpu()
