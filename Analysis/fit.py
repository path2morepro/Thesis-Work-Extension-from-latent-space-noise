from utils import *
import math
import numpy as np
import torch
from torch.nn.functional import silu

invert_euler960 = ROOT.parent / 'data' / 'invert' / 'from_val'
invert_rk4150 = ROOT.parent / 'data' / 'invert' / 'from_val_rk4'
model_save_path = ROOT / "Analysis" / "Models"

class TrajDataset():
    """Point-wise loss, trajectory-wise batches, plus the initial state x0 of every window.

    The (lead, ch, x, y) grid is the same for every trajectory, 
    so it is built once
    (flattened to P = (L-1)*C*H*W points). 
    z_prev / z_next are (n_traj, P) 
    and the point-wise loss broadcasts over the trajectory dim. 
    OPTIONAL x0 is (n_traj, 2, 64, 64): the physical field at the anchor time, standardize in get_windows()
    """
    def __init__(self, path, split, device, traj_per_batch=8, physical_dir=ROOT.parent / 'data' / 'val', condition=False):
        Z, traj_ids, anchors = load_invert(path)
        meta = torch.tensor(np.stack([traj_ids, anchors], 1))
        N, L, C, H, W = Z.shape
        N = int(N*0.8)
        Z = Z[:N] if split == 'train' else Z[N:]                        # split Z exactly once
        if condition:
            _, X0 = get_window(meta, physical_dir)                          # (N, 2, 64, 64)
            X0 = X0[:N] if split == 'train' else X0[N:]
            self.X0 = X0.float().to(device)
        else:
            self.X0 = None                                              # x0-free: batches carry x0 = None

        self.Z = torch.from_numpy(Z).to(device)                         # (n, L, C, H, W)
        self.n, self.L, self.C, self.H, self.W = self.Z.shape
        self.P = (L - 1) * C * H * W
        self.size = self.n * self.P                                     # number of loss terms per epoch
        self.traj_per_batch = traj_per_batch                            # batch size

        t, ch, x, y = torch.meshgrid(torch.arange(L - 1), torch.arange(C), torch.arange(H), torch.arange(W), indexing='ij')
        self.lead = (t.reshape(-1) + 2).float().to(device)              # Z index t+1 is lead t+2
        self.ch = ch.reshape(-1).to(device)
        self.x = x.reshape(-1).float().to(device)
        self.y = y.reshape(-1).float().to(device)

    def iterate_batches(self):
        # one call=sample once with replacement
        perm = torch.randperm(self.n, device=self.Z.device)
        for start in range(0, self.n, self.traj_per_batch):
            idx = perm[start:start + self.traj_per_batch]
            z = self.Z[idx]                                             # (b, L, C, H, W)
            yield (self.lead, self.ch, self.x, self.y,
                   z[:, :-1].reshape(len(idx), -1), z[:, 1:].reshape(len(idx), -1),
                   None if self.X0 is None else self.X0[idx])


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=24, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        #FROMTRACE:
        # num_channels decide how long the output will be
        # max_position decides the slowest wave
        
        # Define the frequencies domain from [0, num_channels//2]
        # Normalise the frequencies domain to [0, 1]
        # For each f: x = x * max_positions^(-f)
        freqs = torch.arange(start=0, end=self.num_channels //
                             2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class FourierEmbedding(torch.nn.Module):
    # num_channels = 0 gives an empty embedding (coordinate ignored)
    def __init__(self, num_channels, period=64):
        super().__init__()
        n_freq = num_channels // 2                    # F = num_channels / 2
        freqs = torch.arange(1, n_freq + 1, dtype=torch.float32) / period
        self.register_buffer('freqs', freqs)   

    def forward(self, x):
        # Outer product
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# without x0 condition
class TimeVaryingOU(torch.nn.Module):
    # x0-free baseline: sigma, c2 as a function of (lead, ch, y) only
    # embedding t with positional embedding
    # embedding x / y with fourier embedding (x_dim / y_dim channels; 0 = ignore that coordinate)
    # embedding ch with nn.Embedding(2, d)
    # n_hidden MLP layers with SiLU, softplus heads
    # tuned: 1 hidden layer, no x, one y mode (val nll 0.1157 vs 0.1160 for the 3-layer xy_dim=8 model)
    def __init__(self, emb_dim=64, hidden_dim=256, n_hidden=1, x_dim=0, y_dim=2, out_dim=2):
        super().__init__()
        self.t_emb = PositionalEmbedding(emb_dim)
        self.x_emb = FourierEmbedding(x_dim)
        self.y_emb = FourierEmbedding(y_dim)
        self.ch_emb = torch.nn.Embedding(2, emb_dim)

        in_dim = emb_dim + x_dim + y_dim + emb_dim 
        dims = [in_dim] + [hidden_dim] * n_hidden
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])
        )
        self.out = torch.nn.Linear(hidden_dim, out_dim)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        self.out.bias.data[:2] = torch.tensor([0.541, 0.07])   # sigma = 1, alpha ≈ 0.7

    def features(self, t, x, y, ch):
        h = torch.cat([self.t_emb(t), self.x_emb(x), self.y_emb(y), self.ch_emb(ch)], dim=1)
        for layer in self.layers:
            h = silu(layer(h))
        return self.out(h)                                          # (P, out_dim)

    def forward(self, t, x, y, ch, x0=None):                        # x0 is accepted and ignored
        h = self.features(t, x, y, ch)
        sigma = torch.nn.functional.softplus(h[:, 0]) + 1e-6
        c2 = torch.nn.functional.softplus(h[:, 1]) + 1e-6
        return sigma, c2

# x0 encoding
class HandFeatures(torch.nn.Module):
    """Fixed per-pixel descriptors of x0: value, gradient magnitude and Laplacian of each
    channel (periodic central finite differences), standardised with train-set statistics.
    Output (b, 6, 64, 64)."""
    def __init__(self):
        super().__init__()
        self.r = 6
        self.register_buffer('mu', torch.zeros(self.r))
        self.register_buffer('sd', torch.ones(self.r))

    def raw(self, x0):
        dx = (torch.roll(x0, -1, 2) - torch.roll(x0, 1, 2)) / 2
        dy = (torch.roll(x0, -1, 3) - torch.roll(x0, 1, 3)) / 2
        lap = torch.roll(x0, -1, 2) + torch.roll(x0, 1, 2) + torch.roll(x0, -1, 3) + torch.roll(x0, 1, 3) - 4 * x0
        return torch.cat([x0, (dx**2 + dy**2).sqrt(), lap], 1)

    def fit(self, X0):
        with torch.no_grad():
            f = self.raw(X0)
            self.mu.copy_(f.mean((0, 2, 3)))
            self.sd.copy_(f.std((0, 2, 3)) + 1e-6)

    def forward(self, x0):
        return (self.raw(x0) - self.mu[None, :, None, None]) / self.sd[None, :, None, None]


class CNNFeatures(torch.nn.Module):
    """Fully convolutional net on [x0, HandFeatures(x0)] -> (b, r, 64, 64) learned per-pixel features.
    3x3 convolutions with circular padding (periodic box) and increasing dilation, so the
    receptive field grows to ~63 px (the whole box) with dilations [1, 2, 4, 8, 16, 1]. Translation-equivariant:
    it reacts to the flow pattern around a pixel, never to the pixel's absolute position.
    Tuned: whole-box context, dropout 0.1, with hand features -> val nll 0.066 (harness 0.0669, notebook run 0.0658)."""
    def __init__(self, r=32, width=32, dilations=(1, 2, 4, 8, 16, 1), hand=True, dropout=0.1):
        super().__init__()
        self.hand = HandFeatures() if hand else None
        c = 2 + (self.hand.r if hand else 0)
        layers = []
        for d in dilations:
            layers += [torch.nn.Conv2d(c, width, 3, padding=d, dilation=d, padding_mode='circular'), torch.nn.SiLU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout2d(dropout))     # channel dropout: the wide net over-fits the 120 train x0 fields without it
            c = width
        layers += [torch.nn.Conv2d(c, r, 1)]
        self.net = torch.nn.Sequential(*layers)
        self.r = r

    def forward(self, x0):
        inp = x0 if self.hand is None else torch.cat([x0, self.hand(x0)], 1)
        return self.net(inp)


class X0ConditionedOU(torch.nn.Module):
    """sigma, c2 depend on (lead, ch, y) AND on the initial state x0.

    Bilinear readout (a FiLM / hypernetwork-style conditional linear layer):
        pre-softplus h[b, p, o] = base[p, o] + sum_r A[p, o, r] * F[b, r, pix(p)]
    where base and A come from the (lead, ch, y) MLP (TimeVaryingOU with 2 + 2r outputs) and
    F = CNNFeatures(x0). A is zero-initialised, so training starts exactly at the x0-free model.
    Returns sigma, c2 of shape (b, P) -> the per-pixel loss broadcasts over trajectories."""
    def __init__(self, feat=None, emb_dim=64, hidden_dim=256, n_hidden=1, x_dim=0, y_dim=2):
        super().__init__()
        self.feat = feat if feat is not None else CNNFeatures()
        self.r = self.feat.r
        self.point = TimeVaryingOU(emb_dim, hidden_dim, n_hidden, x_dim, y_dim, out_dim=2 + 2 * self.r)

    def forward(self, t, x, y, ch, x0):
        o = self.point.features(t, x, y, ch)                         # (P, 2 + 2r)
        base, A = o[:, :2], o[:, 2:].view(-1, 2, self.r)
        Fm = self.feat(x0).flatten(2)                                # (b, r, 64*64)
        Fp = Fm[:, :, x.long() * 64 + y.long()]                      # (b, r, P)  gather the pixel of every point
        h = base[None] + torch.einsum('por,brp->bpo', A, Fp)        # (b, P, 2)
        sigma = torch.nn.functional.softplus(h[..., 0]) + 1e-6
        c2 = torch.nn.functional.softplus(h[..., 1]) + 1e-6
        return sigma, c2

# with learning both sigma and c2
def sigma_c2_mle(batch, model, device):
    *tensors, x0 = batch
    lead, ch, x, y, z_prev, z_next = (b.to(device) for b in tensors)
    x0 = None if x0 is None else x0.to(device)      # None for the x0-free dataset
    sigma_next, c2 = model(lead, x, y, ch, x0)      # sigma at lead k, c^2 on interval [k-1, k]
    sigma_prev, _ = model(lead - 1, x, y, ch, x0)   # sigma at lead k-1
    alpha = torch.exp(-0.5 * c2)                    # dt = 1 h
    one_m_a2 = -torch.expm1(-c2)                    # 1 - alpha^2, stable as alpha -> 1
    xi_prev, xi_next = z_prev / sigma_prev, z_next / sigma_next   # (n_traj, P)
    loss = 0.5 * (torch.log(sigma_next**2) + torch.log(one_m_a2) + (xi_next - alpha * xi_prev)**2 / one_m_a2)
    first = lead == 2                               # transition 1 -> 2 also carries the lead-1 marginal
    loss = loss + first * 0.5 * (torch.log(sigma_prev**2) + xi_prev**2)
    return loss.mean(), loss.numel()

# only c2



def sigma_c2_train(model, train_ds, val_ds, device, epochs=150, lr=3e-4, weight_decay=1e-4, cosine=True, print_every=10):
    # dt is the gap between consecutive leads in hours, so c is in physical units
    # one epoch = n_traj / traj_per_batch optimizer steps (15 for the train split), hence the many epochs
    model = model.to(device)
    if getattr(getattr(model, 'feat', None), 'hand', None) is not None:
        model.feat.hand.fit(train_ds.X0)                         # standardise hand features with train stats
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(train_ds.n / train_ds.traj_per_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch) if cosine else None

    history = {'train': [], 'val': []}
    best = (float('inf'), None)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_ds.iterate_batches():
            loss, n = sigma_c2_mle(batch, model, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * n
        history['train'].append(total / train_ds.size)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_ds.iterate_batches():
                loss, n = sigma_c2_mle(batch, model, device)
                total += loss.item() * n
        history['val'].append(total / val_ds.size)
        if history['val'][-1] < best[0]:
            best = (history['val'][-1], {k: v.detach().clone() for k, v in model.state_dict().items()})

        if (epoch + 1) % print_every == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}/{epochs}  train nll {history['train'][-1]:.4f}  val nll {history['val'][-1]:.4f}")
    print(f"best val nll {best[0]:.5f} at epoch {int(np.argmin(history['val'])) + 1}, last val nll {history['val'][-1]:.5f}")
    history['best_state'] = best[1]
    return history
