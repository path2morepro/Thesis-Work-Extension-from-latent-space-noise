
import json
import sys
import matplotlib
matplotlib.use('Agg')                                   # no display needed
from fit import *

CONFIG = dict(
    # data
    data=str(invert_euler960),          # Euler latents (data/invert/from_val)
    train_frac=0.8,                     # first 80% of windows train, last 20% val
    traj_per_batch=8,                   # trajectories per optimizer step
    # model: X0ConditionedOU, c2 only
    emb_dim=64, hidden_dim=256, n_hidden=1,   # (lead, ch, y) MLP: 1 hidden layer
    x_dim=0, y_dim=2,                   # x ignored, one Fourier mode in y
    r=32, width=32,                     # CNN readout features / channels
    dilations=(1, 2, 4, 8, 16, 1),      # whole-box receptive field
    hand=False,                         # no hand features
    dropout=0.1,                        # Dropout2d after every conv layer
    # optimisation (as tuned in time-varying.ipynb for the x0-conditioned model)
    epochs=150, lr=1e-3, weight_decay=1e-4,
    cosine=True,                        # CosineAnnealingLR stepped every batch, T_max = all steps
    seed=0,
)
EXPERIMENTS = {                         # command-line key -> (NAME, same_x0)
    'own': ('c2_euler', False),
    'same': ('c2_euler_same_x0', True),
}


def run(name, same_x0, device):
    """Train one experiment and save its model, loss plot and history."""
    torch.manual_seed(CONFIG['seed'])                    # same seed -> same initial weights and batch order

    kw = dict(device=device, traj_per_batch=CONFIG['traj_per_batch'], condition=True, train_frac=CONFIG['train_frac'])
    train_ds = TrajDataset(CONFIG['data'], split='train', **kw)
    val_ds = TrajDataset(CONFIG['data'], split='val', **kw)
    if same_x0:
        fixed = train_ds.X0[:1]                          # x0 of the first training window, (1, 2, 64, 64)
        train_ds.X0 = fixed.expand(train_ds.n, -1, -1, -1)
        val_ds.X0 = fixed.expand(val_ds.n, -1, -1, -1)
    print(f'[{name}] {train_ds.n} train / {val_ds.n} val trajectories, {train_ds.L} leads, '
          f"x0: {'same for all windows' if same_x0 else 'own per window'}, device {device}")

    feat = CNNFeatures(r=CONFIG['r'], width=CONFIG['width'], dilations=CONFIG['dilations'],
                       hand=CONFIG['hand'], dropout=CONFIG['dropout'])
    model = X0ConditionedOU(feat, emb_dim=CONFIG['emb_dim'], hidden_dim=CONFIG['hidden_dim'],
                            n_hidden=CONFIG['n_hidden'], x_dim=CONFIG['x_dim'], y_dim=CONFIG['y_dim'],
                            fit_sigma=False)
    print(f'[{name}]', sum(p.numel() for p in model.parameters()), 'parameters')

    history = train(model, train_ds, val_ds, c2_mle, epochs=CONFIG['epochs'], lr=CONFIG['lr'],
                    weight_decay=CONFIG['weight_decay'], cosine=CONFIG['cosine'], patience=None)

    torch.save(model.state_dict(), model_save_path / f'{name}.pt')     # last epoch
    plot_history(history, result_path / f'{name}_loss.png',
                 title=f"{name}: c2-only OU, x0 {'same for all' if same_x0 else 'own'}, lr {CONFIG['lr']} cosine")
    with open(result_path / f'{name}_history.json', 'w') as f:
        json.dump({'config': CONFIG, 'same_x0': same_x0,
                   'train': history['train'], 'val': history['val'], 'lr': history['lr']}, f, indent=1)
    print(f'[{name}] saved', model_save_path / f'{name}.pt', 'and', result_path / f'{name}_loss.png')


def plot_x0_effect():
    """Both runs on one axis: solid = val, dashed = train. Uses the saved histories."""
    files = {key: result_path / f'{name}_history.json' for key, (name, _) in EXPERIMENTS.items()}
    if not all(f.exists() for f in files.values()):
        print('comparison plot skipped: run both experiments first')
        return
    hist = {key: json.load(open(f)) for key, f in files.items()}
    labels = {'own': 'own x0 per window', 'same': 'same x0 for all windows'}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, color in [('own', 'tab:blue'), ('same', 'tab:orange')]:
        h = hist[key]
        epochs = np.arange(1, len(h['val']) + 1)
        ax.plot(epochs, h['val'], color=color, label=f"{labels[key]}: val (last {h['val'][-1]:.4f})")
        ax.plot(epochs, h['train'], color=color, ls='--', alpha=0.6, label=f'{labels[key]}: train')
    gap = hist['same']['val'][-1] - hist['own']['val'][-1]
    ax.set_title(f'x0 effect on val nll at the last epoch: {gap:+.4f}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('nll per loss term')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_path / 'c2_euler_x0_effect.png', dpi=150)
    plt.close(fig)
    print('x0 effect (val nll, last epoch): same x0 - own x0 =', f'{gap:+.5f}')


def main(keys):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    for key in keys:
        run(*EXPERIMENTS[key], device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    plot_x0_effect()



if __name__ == '__main__':
    keys = sys.argv[1:] or list(EXPERIMENTS)
    assert all(k in EXPERIMENTS for k in keys), f'choose from {list(EXPERIMENTS)}'
    main(keys)
