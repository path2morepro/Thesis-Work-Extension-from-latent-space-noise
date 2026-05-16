import torch
from functools import partial
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffusion_networks import SongUNet

class Sampler():
    def __init__(self, device, members, eps, steps, 
                 invert_eps, invert_steps, model_path, debug=False, grad=False, deterministic=False):
        self.model = SongUNet(img_resolution=64, in_channels=2, out_channels=2,
                              embedding_type='fourier', encoder_type='residual', decoder_type='standard',
                              channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=32, channel_mult=[2, 2, 2],
                              attn_resolutions=[32,]
                              )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.model.to(device)
        self.model.eval()
        # and what's this?
        self.model.load_state_dict(torch.load(
            model_path, map_location=device, weights_only=True))

        self.device = device
        self.members = members

        self.alpha = lambda t: t
        self.beta = lambda t: 1 - t  # Adjusted beta to account for prior noise
        self.alpha_dot = lambda t: 1  # Derivative of alpha with respect to t
        self.beta_dot = lambda t: - 1  # Derivative of beta with respect to t
        self.gamma = lambda t: self.beta(t) * self.alpha_dot(t) - self.alpha(t) * self.beta_dot(t)
        
        self.invert_eps = invert_eps
        self.invert_steps = invert_steps
        self.eps = eps
        self.steps = steps
        self.grad = grad
        self.deterministic = deterministic
        self.debug = debug

    # Euler-Maruyama sampling with learned b
    def sample(self, z0):
        eps = self.eps
        steps = self.steps

        status = torch.enable_grad() if self.grad else torch.no_grad()

        with status:
            tmin, tmax = 0, 1
            zt = z0
            ts = torch.linspace(tmin, tmax, steps, device=self.device)[:-1]
            dt = (tmax - tmin) / steps

            if self.debug:
                enum = tqdm(ts)
            else:
                enum = ts

            zs = [zt.clone().cpu()] if self.debug else None

            for t in enum:
                alpha_t, beta_t = self.alpha(t), self.beta(t)
                # alpha(t) = t
                # beta(t) = 1-t
                alpha_dot_t, gamma_t = self.alpha_dot(t), self.gamma(t)
                eps_t = eps(t)  

                t_tensor = torch.ones((self.members,), device=self.device) * t
                # which structure it is?
                b = self.model(zt, t_tensor)

                if t != 1:
                    s = (alpha_t * b - alpha_dot_t * zt) / (beta_t * gamma_t)  # s = (t * b - zt) / (1 - t)
                else:
                    s, eps_t = 0, 0

                dz = (b + s * eps_t) * dt
                
                dW = 0.0 if self.deterministic else torch.randn_like(zt) * math.sqrt(2*math.fabs(dt) * eps_t)

                zt = zt + dz + dW

                if self.debug:
                    zs.append(zt.clone().cpu())

            if self.debug:
                return zt, torch.stack(zs, dim=1)
            else:
                return zt

    # Inversion
    def invert(self, z1):
        eps = self.invert_eps
        steps = self.invert_steps

        # I remember it is used to turn off gradient tracking
        # honestly I don't know how does this work
        with torch.no_grad():
            tmin, tmax = 1, 0
            zt = z1

            ts = torch.linspace(tmin, tmax, steps, device=self.device)[:-1]
            dt = (tmax - tmin) / steps

            if self.debug:
                enum = tqdm(ts)
            else:
                enum = ts

            for t in enum:
                # what do alpha, beta, gamma present? the noise intensity
                # there is statements before, dot means derivative
                alpha_t, beta_t = self.alpha(t), self.beta(t)
                alpha_dot_t, gamma_t = self.alpha_dot(t), self.gamma(t)
                
                eps_t = eps(t)

                t_tensor = torch.ones((self.members,), device=self.device) * t
                
                b = self.model(zt, t_tensor)
                # SongUNet, check it out later
                # what the output would be like?
                # the output is the predicted velocity field, namely z1-z0
                # z1 is data, z0 is white noise

                # it's actually ODE so the definition of s would not be used anyway
                if t != 1:
                    s = (alpha_t * b - alpha_dot_t * zt) / (beta_t * gamma_t)  # s = (t * b - zt) / (1 - t)
                else:
                    s, eps_t = 0, 0
                # dz/dt = b
                dz = (b - eps_t * s) * dt
                # it is forward progress, it's impossible to remove noise
                dW = torch.randn_like(zt) * math.sqrt(2*math.fabs(dt) * eps_t)

                zt = zt + dz + dW
            return zt
