import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from diffusion_networks import SongUNet


class CondSampler:
    """
    Conditional flow-matching sampler for p(x_{t+lead_time} | x_t).

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
        invert_steps=100,
        debug=False
    ):
        """
        Args:
            model_path   : path to saved state dict (.pth)
            device       : torch device
            steps        : Euler steps for sampling (forward ODE)
            invert_steps : Euler steps for inversion (backward ODE)
            debug        : if True, sample() returns full trajectory tensor
        """
        self.model = SongUNet(
            img_resolution=64,
            in_channels=4,       # 2 ch (z_s) + 2 ch (x_t conditioning)
            out_channels=2,
            embedding_type='positional',
            encoder_type='residual',
            decoder_type='standard',
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=32,
            channel_mult=[2, 2, 2],
            attn_resolutions=[32],
            label_dropout=0.1,
            time_emb=1,          # enable map_time so the lead-time condition is used
                                 # (must match training; checkpoint has map_time.freqs)
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
        self.debug = debug

    def _score(self, t_val, b, zt):
        """Score estimate s(z_s, t) = (t*b - z_t) / (1 - t) from the velocity field."""
        beta = 1.0 - t_val
        if beta > 1e-8:
            return (t_val * b - zt) / beta
        return torch.zeros_like(b)

    def sample(self, z0, x_t, lead_times=None):
        """
        input: 
        z0: noise
        xt: initial state
        lead_times: lead_times
        """
        B = z0.shape[0]
        dt = 1.0 / self.steps
        ts = torch.linspace(0, 1, self.steps + 1, device=self.device)[:-1]

        zt = z0.to(self.device)
        x_t = x_t.to(self.device)
        if lead_times is not None:
            lead_times = lead_times.to(self.device)

        trajectory = [zt.clone().cpu()] if self.debug else None

        with torch.no_grad():
            enum = tqdm(ts, desc='sample') if self.debug else ts
            for t in enum:
                t_val = t.item()

                s_vec = torch.full((B,), t_val, device=self.device)
                # lead_times is the (B,) lead-time condition; it is fixed for the
                # whole ODE integration (only the interpolant time s_vec changes).
                b = self.model(zt, s_vec, class_labels=x_t, time_labels=lead_times)

                dz = b * dt
                zt = zt + dz

                if self.debug:
                    trajectory.append(zt.clone().cpu())

        result = zt.cpu()
        if self.debug:
            return result, torch.stack(trajectory, dim=1)
        return result



