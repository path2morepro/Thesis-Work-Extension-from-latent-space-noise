# Thesis Session Recap

## What happened across these sessions

Two things: (1) reading and understanding the theoretical background papers, and (2) making the first concrete implementation step from Martin's project plan.

---

## 1. Theory read — what you understood

### Martin's paper (Continuous Ensemble Weather Forecasting with Diffusion Models)

The model is a **score-based diffusion model** (not flow-matching) trained to denoise: it maps random Gaussian noise → real SQG/ERA5 weather states conditioned on an initial condition $x_0$ and lead time $t$. Specifically it learns $p(x_t | x_0, t)$ in a single forward pass — no autoregressive stepping.

**Training:** standard denoising objective from Karras et al. (EDM 2022). The score network $S_\theta$ is trained to match the score function via denoising regression. Conditioning on $x_0$ and $t$ is injected via `class_labels` and `time_labels` in `SongUNet`.

**Forecasting — two sampling modes:**
- **Algorithm 1 (fixed noise):** a single $Z \sim \mathcal{N}(0,I)$ is drawn and reused for all lead times. Makes trajectories conditionally deterministic — knowing one timestep determines all others.
- **Algorithm 2 (OU process):** the driving noise $Z(t)$ evolves as an Ornstein–Uhlenbeck process across lead times:
  $$z_i = e^{-\rho \Delta t} z_{i-1} + \sqrt{1 - e^{-2\rho \Delta t}}\, \nu_i, \quad \nu_i \sim \mathcal{N}(0,I)$$
  Keeps correct marginals $Z(t) \sim \mathcal{N}(0,I)$ at every lead time while adding temporal stochasticity. **No retraining needed** — it only changes inference.

---

### Flow-matching / Stochastic Interpolants (Albergo, Boffi, Vanden-Eijnden)

The interpolant framework gives a different way to build generative models. Key objects:

**Interpolant:**
$$x_t = \alpha(t)\, x_0 + \beta(t)\, x_1 + \gamma(t)\, z, \quad z \sim \mathcal{N}(0,I)$$
where $x_0 \sim \rho_0$ (base, e.g. Gaussian), $x_1 \sim \rho_1$ (target, e.g. real data).

**What you train — the velocity field $b(t,x)$:**
$$b(t, x) = \mathbb{E}[\dot{x}_t \mid x_t = x]$$
This is the conditional expectation of the instantaneous velocity, conditioned on position $x$ at time $t$ — **not** conditioned on $x_1$ alone. Training minimises MSE between a neural network $\hat{b}$ and the analytic velocity $\dot{x}_t = \dot{\alpha}(t)x_0 + \dot{\beta}(t)x_1 + \dot{\gamma}(t)z$.

**Sampling (generation):** integrate the ODE forward $t: 0 \to 1$
$$\frac{d}{dt}X_t = b(t, X_t), \quad X_0 \sim \rho_0$$
This maps noise → real data directly. No separate latent encoding step.

**Encoding (inversion):** run the same ODE **backward** $t: 1 \to 0$. Maps real data → latent noise. This gives a bijection (when $\gamma(t) = 0$) and is used in DAISI for data assimilation.

**Repo uses flow-matching, not score-based diffusion.** Martin told you to use the project-course repo which is a flow-matching model trained unconditionally on SQG (learns $p(x_1)$). The `class_labels` / `time_labels` slots exist in the architecture but are unused.

---

### SongUNet forward pass — the important parts

The network takes `(x, noise_labels, class_labels, time_labels)`.

**Embedding block:** `noise_labels` (σ or interpolant time $s$) → `FourierEmbedding` → summed with projected `class_labels` + `time_labels` → 2-layer MLP → `emb` vector `[B, emb_channels]`. This global vector gets injected into every UNetBlock via adaptive normalisation.

**Spatial conditioning:** `class_labels` is *also* concatenated channel-wise onto `x` before the encoder: `cat(x, class_labels, dim=1)`. This is the full spatial conditioning. The two uses of `class_labels` (embedding path = global summary, concatenation path = full spatial) are independent and complementary.

**Encoder → Bottleneck → Decoder:** standard U-Net with skip connections. Skip tensors saved during encoder, popped during decoder. Output `F_x` is shape `[B, out_channels, H, W]` — the predicted velocity (in flow-matching) or denoised image (in EDM).

---

## 2. Your thesis direction (clarified)

**You do not need to retrain from scratch.** The task is:

1. **Step 2** — train a conditional flow-matching model $p(x_{t+1} | x_t)$ (autoregressive, 1-step). This requires modifying the existing repo.
2. **Step 3** — extend to $p(x_t | x_0, t)$ continuous forecasting, adding `time_labels`.
3. **Steps 4–7** — the actual thesis contribution: invert real trajectories through the reverse ODE to get latent noise sequences, characterise their structure, design or learn a stochastic process $Z(t)$ that drives better ensemble trajectories than fixed noise.

The thesis question in one sentence: **can we find a better noise process $Z(t)$ than fixed noise or OU, by analysing what noise sequences real SQG trajectories correspond to in latent space?**

---

## 3. Implementation done — `trainDiffusion.ipynb` modified

**What changed from the unconditional notebook:**

| Part | Before | After |
|---|---|---|
| Dataset | `SQGPairDataset` — single `.npy` file, time-split | `SQGPairDataset` — multi-file, trajectory-split |
| `in_channels` | `2` | `4` (2 for $z_s$ + 2 for $x_t$ concatenated) |
| `loss_fn` signature | `(model, batch, target_fn)` | `(model, x_t, x_t1)` |
| `class_labels` | `None` | `x_t` — previous SQG state |
| `label_dropout` | not set | `0.1` — enables CFG at inference |
| Training loop | unpacks single `image` | unpacks `(x_t, x_t1)` pairs |

**Loss (unchanged formula, new meaning):**
$$\mathcal{L} = \mathbb{E}\left[\|\hat{b}(z_s, s, x_t) - (x_{t+1} - z_0)\|^2\right]$$
- Interpolant: $z_s = (1-s)z_0 + s \cdot x_{t+1}$
- Target velocity: $x_{t+1} - z_0$
- Conditioning: $x_t$ concatenated spatially inside the network

**Dataset structure:** the `data/` folder has ~1000+ individual files `sqg_N64_3hrly_{version}.npy`, each shape `(100, 2, 64, 64)`. The new dataset class loads all matching files, splits 80/20 by trajectory index (not time), and builds a flat index of `(trajectory_i, timestep_t)` pairs.

**Sanity check cell** at end: Euler ODE integration from $s=0$ (noise) to $s=1$ (forecast), plots conditioning state / prediction / truth side-by-side for 4 validation samples.

---

## 4. Data generation (`sqg_nature_run.py`)

Changes needed for 100 trajectories × 100 steps × 1-hourly: (not sure whether I should do that, I work on 3 hourly data firstly)

```python
versions = 100          # was 1
hrs = 1                 # was 3
tmax = tmin + 100 * 3600.   # was 312.5*86400
```

Run from project root: `python SQG/sqg_nature_run.py`

Output: `data/sqg_N64_1hrly_0.npy` … `data/sqg_N64_1hrly_99.npy`, each `(100, 2, 64, 64)`.

**Note:** decided to use existing 3-hourly dataset (`sqg_N64_3hrly_*.npy`) for now because data generation is CPU-only, slow, and the 3-hourly data is already available in sufficient quantity.

---

## Key things to remember

- The repo's unconditional model is **flow-matching**, not score-based diffusion — this matters because the velocity target and ODE are different from Martin's paper.
- `class_labels` does double duty: spatial concatenation (full field) AND projected into the embedding (global summary). Both happen automatically when you pass it.
- The thesis novelty is **not** training a better model architecture — it's finding a better $Z(t)$ process at inference time by analysing the latent dynamics of real trajectories.
- `label_dropout=0.1` is already in the model — to use CFG at inference, run the model twice per step (once with `class_labels=x_t`, once with `class_labels=None`) and interpolate: $\hat{v} = v_\text{uncond} + w(v_\text{cond} - v_\text{uncond})$.
