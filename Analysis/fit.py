
from utils import *
import math
import numpy as np
import torch
from torch.nn.functional import silu, softplus

invert_euler960 = ROOT.parent / 'data' / 'invert' / 'from_val'
invert_rk4150 = ROOT.parent / 'data' / 'invert' / 'from_val_rk4'
model_save_path = ROOT / 'Models'                     # ROOT is already the Analysis folder
result_path = ROOT / 'Result'


# ----------------------------------------------------------------------------- data
class TrajDataset():
    """Trajectory-wise batches on the factored (lead, ch, x, y) grid.

    Z: (n, L, C, H, W) latents of the split, kept as is (no flattening, no index arrays).
    lead: (K,) = leads 2..L of the transitions; Z index k is lead k+1, so z_next = Z[:, 1:] is at `lead`.
    Optional X0 (condition=True): (n, 2, 64, 64) physical state at the anchor, standardised in get_window().
    A batch is (lead, z_prev, z_next, x0) with z_prev / z_next of shape (b, K, C, H, W) and x0 None
    when condition=False."""
    def __init__(self, path, split, device, traj_per_batch=8, physical_dir=ROOT.parent / 'data' / 'val',
                 condition=False, train_frac=0.8):
        Z, traj_ids, anchors = load_invert(path)
        n_train = int(len(Z) * train_frac)
        part = slice(None, n_train) if split == 'train' else slice(n_train, None)   # split exactly once

        self.Z = torch.from_numpy(Z[part]).to(device)                   # (n, L, C, H, W)
        if condition:
            meta = torch.tensor(np.stack([traj_ids[part], anchors[part]], 1))
            _, X0 = get_window(meta, physical_dir)                      # only this split's windows are read
            self.X0 = X0.float().to(device)                             # (n, 2, 64, 64)
        else:
            self.X0 = None

        self.n, self.L, self.C, self.H, self.W = self.Z.shape
        self.lead = torch.arange(2, self.L + 1, dtype=torch.float32, device=device)   # (K,), K = L-1
        self.size = self.n * (self.L - 1) * self.C * self.H * self.W   # number of loss terms per epoch
        self.traj_per_batch = traj_per_batch                            # batch size in trajectories

    def iterate_batches(self):
        # one pass over the split in random order, without replacement
        perm = torch.randperm(self.n, device=self.Z.device)
        for start in range(0, self.n, self.traj_per_batch):
            idx = perm[start:start + self.traj_per_batch]
            z = self.Z[idx]                                             # (b, L, C, H, W)
            yield self.lead, z[:, :-1], z[:, 1:], None if self.X0 is None else self.X0[idx]


# ----------------------------------------------------------------------------- embeddings
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


# ----------------------------------------------------------------------------- models
class TimeVaryingOU(torch.nn.Module):
    """x0-free model: sigma, c2 as a function of (lead, ch, y) only.

    lead -> positional embedding,
    ch -> nn.Embedding(2, d), 
    x / y -> Fourier embedding
    n_hidden MLP layers with SiLU, softplus heads.

    The MLP is evaluated on the factored grid: K leads x C channels x (H if x_dim else 1) x W.
    fit_sigma=True  -> forward returns (sigma, c2), each (1, K, C, H|1, W)
    fit_sigma=False -> forward returns c2 only
    out_dim overrides the head size when the MLP is used as a feature network (X0ConditionedOU)."""
    def __init__(self, emb_dim=64, hidden_dim=256, n_hidden=1, x_dim=0, y_dim=2, fit_sigma=True,
                 out_dim=None, C=2, H=64, W=64):
        super().__init__()
        self.fit_sigma = fit_sigma
        self.x_dim = x_dim
        self.t_emb = PositionalEmbedding(emb_dim)
        self.x_emb = FourierEmbedding(x_dim)
        self.y_emb = FourierEmbedding(y_dim)
        self.ch_emb = torch.nn.Embedding(C, emb_dim)
        # coordinate axes of the box (not saved in the state dict)
        self.register_buffer('chs', torch.arange(C, dtype=torch.float32), persistent=False)   # float: meshgrid needs one dtype
        self.register_buffer('xs', torch.arange(H, dtype=torch.float32), persistent=False)
        self.register_buffer('ys', torch.arange(W, dtype=torch.float32), persistent=False)

        in_dim = emb_dim + x_dim + y_dim + emb_dim
        dims = [in_dim] + [hidden_dim] * n_hidden
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])
        )
        n_heads = 2 if fit_sigma else 1
        self.out = torch.nn.Linear(hidden_dim, out_dim or n_heads)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        # start at sigma = 1 (softplus(0.541) = 1), c2 = softplus(0.07) ~ 0.73 -> alpha ~ 0.7
        self.out.bias.data[:n_heads] = torch.tensor([0.541, 0.07] if fit_sigma else [0.07])

    def features(self, lead):
        """lead: (K,) -> MLP output (K, C, H|1, W, out_dim). Each (lead, ch, x, y) is evaluated once."""
        xs = self.xs if self.x_dim else self.xs[:1]                     # x ignored -> a single x
        t, c, x, y = torch.meshgrid(lead, self.chs, xs, self.ys, indexing='ij')
        h = torch.cat([self.t_emb(t.reshape(-1)), self.x_emb(x.reshape(-1)),
                       self.y_emb(y.reshape(-1)), self.ch_emb(c.reshape(-1).long())], dim=1)
        for layer in self.layers:
            h = silu(layer(h))
        return self.out(h).view(*t.shape, -1)

    def forward(self, lead, x0=None):                                   # x0 is accepted and ignored
        h = softplus(self.features(lead)[None]) + 1e-6                  # (1, K, C, H|1, W, n_heads)
        return (h[..., 0], h[..., 1]) if self.fit_sigma else h[..., 0]


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
    """Fully convolutional net on x0 (optionally plus HandFeatures(x0)) -> (b, r, 64, 64) learned per-pixel features.
    3x3 convolutions with circular padding (periodic box) and increasing dilation, so the
    receptive field grows to ~63 px (the whole box) with dilations [1, 2, 4, 8, 16, 1]. Translation-equivariant:
    it reacts to the flow pattern around a pixel, never to the pixel's absolute position.
    Tuned (sigma + c2): whole-box context, dropout 0.1 -> val nll 0.066 with hand features, 0.067 without."""
    def __init__(self, r=32, width=32, dilations=(1, 2, 4, 8, 16, 1), hand=False, dropout=0.1):
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

    def __init__(self, feat=None, emb_dim=64, hidden_dim=256, n_hidden=1, x_dim=0, y_dim=2, fit_sigma=True):
        super().__init__()
        self.feat = feat if feat is not None else CNNFeatures()
        self.r = self.feat.r
        self.fit_sigma = fit_sigma
        self.n_heads = 2 if fit_sigma else 1
        self.point = TimeVaryingOU(emb_dim, hidden_dim, n_hidden, x_dim, y_dim, fit_sigma=fit_sigma,
                                   out_dim=self.n_heads * (1 + self.r))

    def forward(self, lead, x0):
        o = self.point.features(lead)                                   # (K, C, H|1, W, n + n*r)
        n = self.n_heads
        base, A = o[..., :n], o[..., n:].unflatten(-1, (n, self.r))     # (K,C,X,W,n), (K,C,X,W,n,r)
        F = self.feat(x0)                                               # (b, r, H, W)
        if A.shape[2] == 1:                                             # x ignored: A is the same for every x
            h = base[None] + torch.einsum('kcwor,brhw->bkchwo', A[:, :, 0], F)
        else:
            h = base[None] + torch.einsum('kchwor,brhw->bkchwo', A, F)
        h = softplus(h) + 1e-6                                          # (b, K, C, H, W, n)
        return (h[..., 0], h[..., 1]) if self.fit_sigma else h[..., 0]


# ----------------------------------------------------------------------------- losses
def sigma_c2_mle(batch, model):
    """Learn sigma and c2. The model is called once on leads 1..L (one CNN pass, one dropout mask),
    then sliced: sigma at lead k-1 and k, c2 on the interval [k-1, k]."""
    lead, z_prev, z_next, x0 = batch                                    # z: (b, K, C, H, W), leads 2..L
    sigma, c2 = model(torch.cat([lead[:1] - 1, lead]), x0)              # leads 1..L
    sigma_prev, sigma_next, c2 = sigma[:, :-1], sigma[:, 1:], c2[:, 1:]
    alpha = torch.exp(-0.5 * c2)                    # dt = 1 h
    one_m_a2 = -torch.expm1(-c2)                    # 1 - alpha^2, stable as alpha -> 1
    xi_prev, xi_next = z_prev / sigma_prev, z_next / sigma_next
    loss = 0.5 * (torch.log(sigma_next**2) + torch.log(one_m_a2) + (xi_next - alpha * xi_prev)**2 / one_m_a2)
    first = (lead == 2).float().view(1, -1, 1, 1, 1)   # transition 1 -> 2 also carries the lead-1 marginal
    loss = loss + first * 0.5 * (torch.log(sigma_prev**2) + xi_prev**2)
    return loss.mean(), loss.numel()


def c2_mle(batch, model):
    """Learn c2 only: z is treated as unit variance, so only alpha = exp(-c2/2) is fitted."""
    lead, z_prev, z_next, x0 = batch
    c2 = model(lead, x0)                            # c^2 on interval [k-1, k], broadcasts to (b, K, C, H, W)
    alpha = torch.exp(-0.5 * c2)                    # dt = 1 h
    one_m_a2 = -torch.expm1(-c2)                    # 1 - alpha^2
    loss = 0.5 * (torch.log(one_m_a2) + (z_next - alpha * z_prev)**2 / one_m_a2)
    return loss.mean(), loss.numel()

def c2_mle_without_x0(batch, model):
    """Learn c2 only: z is treated as unit variance, so only alpha = exp(-c2/2) is fitted."""
    lead, z_prev, z_next, _ = batch
    c2 = model(lead)                            # c^2 on interval [k-1, k], broadcasts to (b, K, C, H, W)
    alpha = torch.exp(-0.5 * c2)                    # dt = 1 h
    one_m_a2 = -torch.expm1(-c2)                    # 1 - alpha^2
    loss = 0.5 * (torch.log(one_m_a2) + (z_next - alpha * z_prev)**2 / one_m_a2)
    return loss.mean(), loss.numel()


# ----------------------------------------------------------------------------- training
def train(model, train_ds, val_ds, loss_fn, epochs=150, lr=1e-3, weight_decay=1e-4, cosine=True,
          print_every=10, patience=None, min_delta=1e-5):
    """AdamW + cosine learning-rate schedule stepped every batch, as tuned in time-varying.ipynb.
    One epoch = n_traj / traj_per_batch optimizer steps (15 for 120 train trajectories), hence many epochs.

    The model is left at its LAST epoch. history['best_state'] holds a copy of the best-val state.
    Optional early stop: patience = epochs without val improving by more than min_delta (None = off).
    Note that val nll moves ~5e-4 between epochs and the cosine schedule plans for all `epochs`,
    so early stopping with a small patience cuts the low-lr phase."""
    device = train_ds.Z.device
    model = model.to(device)
    if getattr(getattr(model, 'feat', None), 'hand', None) is not None:
        model.feat.hand.fit(train_ds.X0)                         # standardise hand features with train stats
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(train_ds.n / train_ds.traj_per_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch) if cosine else None

    history = {'train': [], 'val': [], 'lr': []}
    best = (float('inf'), None, -1)                            # (val, state, epoch index) under the min_delta rule
    wait = 0                                                     # epochs since the last val improvement
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_ds.iterate_batches():
            loss, n = loss_fn(batch, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * n
        history['train'].append(total / train_ds.size)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_ds.iterate_batches():
                loss, n = loss_fn(batch, model)
                total += loss.item() * n
        history['val'].append(total / val_ds.size)
        if history['val'][-1] < best[0] - min_delta:
            best = (history['val'][-1], {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % print_every == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}/{epochs}  train nll {history['train'][-1]:.4f}  val nll {history['val'][-1]:.4f}")
        if patience is not None and wait >= patience:
            print(f"early stop at epoch {epoch + 1}: no val improvement for {patience} epochs")
            break
    print(f"best val nll {best[0]:.5f} at epoch {best[2] + 1}, last val nll {history['val'][-1]:.5f}")
    history['best_state'] = best[1]
    history['best_epoch'] = best[2]                              # 0-based; the epoch best_state comes from
    return history


def plot_history(history, path, title=''):
    """Train / val nll per epoch (top) and learning rate (bottom), saved to `path`."""
    fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    epochs = np.arange(1, len(history['train']) + 1)
    ax.plot(epochs, history['train'], label='train')
    ax.plot(epochs, history['val'], label='val')
    best = history.get('best_epoch', int(np.argmin(history['val'])))   # argmin only for old histories
    ax.scatter(epochs[best], history['val'][best], color='tab:red', zorder=3,
               label=f"best val {history['val'][best]:.4f} (epoch {best + 1})")
    ax.set_ylabel('NLL per loss term')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    ax_lr.plot(epochs, history['lr'], color='gray')
    ax_lr.set_xlabel('epoch')
    ax_lr.set_ylabel('lr')
    ax_lr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history, path):
    """Train / val nll per epoch, saved to `path`."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(history['train']) + 1)
    ax.plot(epochs, history['train'], label='train')
    ax.plot(epochs, history['val'], label='val')
    best = history.get('best_epoch', int(np.argmin(history['val'])))   # argmin only for old histories
    ax.scatter(epochs[best], history['val'][best], color='tab:red', zorder=3,
               label=f"best val {history['val'][best]:.4f} (epoch {best + 1})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

