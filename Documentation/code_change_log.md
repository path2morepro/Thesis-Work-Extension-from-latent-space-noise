date: 31 maj
aim to hyperparameter optimization to get better performance:


trainCondDiffusion.ipynb corrections:
- Imports (cell 2): Added natsorted, sample_one_step, autoregressive_ensemble_rollout, compute_rmse, compute_ssr from autoregressive_forecasting.py
- Config (cell 4): NUM_EPOCHS 50 → 20; added ROLLOUT_STEPS=10, N_ENSEMBLE=10, ODE_STEPS=100, N_INIT=20, TRAIN_FRAC=0.8
- Sanity check (cell 18): Removed duplicate sample_conditional function — now calls sample_one_step from autoregressive_forecasting.py
- New cells 19–22: Load val trajectories, run 10-step rollout, compute + print RMSE & SSR for all 10 lead times, plot both metrics

Hyperparameter_optimization.ipynb (written from scratch, 15 cells):
- Sweeps label_dropout ∈ [0.5, 0.4, 0.3, 0.2] — skips 0.1 (already trained)
- Dataset (SQGPairDataset) and val trajectories loaded once, reused across all runs
- train_one_run(ld): trains fresh model for 20 epochs, saves best checkpoint to ../models/best_model_ld{ld:.2f}.pth, then evaluates autoregressive rollout
- Computes RMSE and SSR only at all 10 rollout steps, averaged over 20 fixed init points (seed=42)
- Full per-lead table + lead-1/lead-5 summary + side-by-side RMSE/SSR comparison plots