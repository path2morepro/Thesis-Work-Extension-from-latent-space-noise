from fit import *         
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Trajectory batches on the factored (lead, ch, x, y) grid: z_prev / z_next are (b, K, C, H, W), lead is (K,) = leads 2..24.
# x0-free: condition=False, so x0 in each batch is None. 80/20 train/val split, 8 trajectories per batch.
train_ds = TrajDataset(invert_euler960, split='train', device=device, traj_per_batch=8, condition=False)

# test code
# lead, z_prev, z_next, _ = next(train_ds.iterate_batches())
# print(train_ds.n, 'trajectories,', train_ds.size, 'loss terms')
# print(lead.shape, z_prev.shape, z_next.shape)
# print(f"lead max: {torch.max(lead)}, lead min {torch.min(lead)}")

val_ds = TrajDataset(invert_euler960, split='val', device=device, traj_per_batch=8, condition=False)   # same data as train
# print(val_ds.n, 'trajectories,', val_ds.size, 'loss terms')


torch.manual_seed(0)
model = TimeVaryingOU(emb_dim=64, hidden_dim=256, n_hidden=1, x_dim=0, y_dim=4, fit_sigma=False)
# print(sum(p.numel() for p in model.parameters()), 'parameters')
history = train(model, train_ds, val_ds, c2_mle_without_x0, epochs=50, lr=1e-3, weight_decay=1e-4,
                cosine=False, patience=20)
plot_loss(history, result_path / 'c2_euler_withoutx0.png')
torch.save(history['best_state'], model_save_path / "c2_euler_withoutx0.pt")      # best-val state; use model.state_dict() for the last epoch
print('saved', save_path)