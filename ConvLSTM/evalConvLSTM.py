import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SQG_ROOT = PROJECT_ROOT / "SQG"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SQG_ROOT) not in sys.path:
    sys.path.insert(0, str(SQG_ROOT))

from sampler import Sampler

def map_latent_to_physical(
    latent,
    sampler,
    device=None,
    data_std=1,
    return_torch=False
):
    """
    Map latent field(s) -> physical field(s) using sampler.sample.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    x = latent
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError("latent must be a numpy array or a torch tensor.")

    x = x.to(torch.float32)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
        T = 1
    elif x.ndim == 3:
        T = x.shape[0]
        x = x.unsqueeze(1).repeat(1, 2, 1, 1)
    elif x.ndim == 4:
        T, L, _, _ = x.shape
        if L == 1:
            x = x.repeat(1, 2, 1, 1)
        elif L != 2:
            raise ValueError(f"Expected level dimension L in {{1,2}}, got L={L}.")
    else:
        raise ValueError(f"Unsupported latent ndim={x.ndim}. Expected 2/3/4 dims.")

    if sampler.members != T:
        raise ValueError(f"sampler.members ({sampler.members}) must equal number of frames T ({T}).")

    x = x.to(device)
    y = sampler.sample(x)
    y0 = y[:, 0, ...] * float(data_std)
    y0_cpu = y0.detach().cpu()
    return y0_cpu if return_torch else y0_cpu.numpy().astype(np.float32)


def visualize_and_rmse(
    phys_pred,
    phys_true,
    suptitle="Latent to Physical comparison",
    cmap="viridis"
):
    import matplotlib.pyplot as plt

    if not isinstance(phys_pred, np.ndarray) or not isinstance(phys_true, np.ndarray):
        raise TypeError("phys_pred and phys_true must be numpy arrays.")

    pred = phys_pred.astype(np.float32)
    true = phys_true.astype(np.float32)

    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected shape (T, W, H), got {pred.shape}")

    diff = pred - true
    per_frame_rmse = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    overall_rmse = float(np.sqrt(np.mean(diff ** 2)))

    print(f"[RMSE] overall = {overall_rmse:.6f}")

    T = pred.shape[0]
    fig, axes = plt.subplots(T, 3, figsize=(13, 3.8 * T))
    if T == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(T):
        vmin = min(true[idx].min(), pred[idx].min())
        vmax = max(true[idx].max(), pred[idx].max())

        im0 = axes[idx, 0].imshow(true[idx], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axes[idx, 0].set_title(f"Truth (example={idx})")
        plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[idx, 1].imshow(pred[idx], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axes[idx, 1].set_title(f"Prediction (example={idx})")
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        im2 = axes[idx, 2].imshow(np.abs(diff[idx]), cmap="bwr", origin="lower")
        axes[idx, 2].set_title(f"Abs diff | RMSE={per_frame_rmse[idx]:.4f}")
        plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)

        for c in range(3):
            axes[idx, c].set_xticks([])
            axes[idx, c].set_yticks([])

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

    return overall_rmse, per_frame_rmse



def test(
    model,
    test_loader,
    device,
    show_progress=True,
):
    criterion = nn.MSELoss()
    model = model.to(device)
    model.eval()

    test_loss_sum = 0.0
    test_count = 0

    test_bar = tqdm(test_loader, desc="Test", leave=False, disable=not show_progress)
    with torch.no_grad():
        for x, y in test_bar:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            batch_size = x.shape[0]
            test_loss_sum += loss.item() * batch_size
            test_count += batch_size
            test_bar.set_postfix(batch_loss=f"{loss.item():.6f}")

    avg_test_loss = test_loss_sum / max(test_count, 1)
    print(f"Test latent loss: {avg_test_loss:.6f}")
    return {"test_loss": avg_test_loss}


def compare_physical_predictions(
    model,
    dataset,
    device,
    sampler_model_path=SQG_ROOT / "best_model.pth",
    truth_folder=DATA_ROOT / "sqg_subset",
    num_examples=3,
    sample_steps=100,
    data_std=2660.0,
):
    model = model.to(device)
    model.eval()

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    chosen = np.linspace(0, len(dataset) - 1, num=min(num_examples, len(dataset)), dtype=int)
    latent_preds = []
    physical_targets = []
    metadata = []

    for idx in chosen:
        x, _ = dataset[idx]
        info = dataset.get_sample_info(idx)

        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu()[0]

        pred = pred * dataset.std + dataset.mean
        latent_preds.append(pred)

        source_file_name = info.get("source_file_name", info["noise_file_name"])
        truth_name = source_file_name.replace("inverted_", "", 1)
        truth_path = Path(truth_folder) / truth_name
        if not truth_path.exists():
            raise FileNotFoundError(f"Missing aligned truth file: {truth_path}")

        truth_arr = np.load(truth_path)
        physical_targets.append(truth_arr[info["target_idx"], info["level"]].astype(np.float32))
        metadata.append({
            **info,
            "truth_file": str(truth_path),
        })

    latent_preds = torch.stack(latent_preds, dim=0)
    physical_targets = np.stack(physical_targets, axis=0)

    sampler_device = torch.device(device)
    zero_eps = lambda t: 0.0
    sampler = Sampler(
        sampler_device,
        members=latent_preds.shape[0],
        eps=zero_eps,
        steps=sample_steps,
        invert_eps=zero_eps,
        invert_steps=sample_steps,
        model_path=str(sampler_model_path),
        debug=False,
        deterministic=True,
    )

    physical_preds = map_latent_to_physical(
        latent_preds,
        sampler,
        device=sampler_device,
        data_std=data_std,
        return_torch=False,
    )

    overall_rmse, per_frame_rmse = visualize_and_rmse(
        physical_preds,
        physical_targets,
        suptitle="Test examples mapped back to physical space",
        cmap="RdBu_r",
    )

    for meta, rmse in zip(metadata, per_frame_rmse):
        print(
            f"dataset_idx={meta['dataset_idx']} | traj_id={meta['traj_id']} | "
            f"target_idx={meta['target_idx']} | truth_file={Path(meta['truth_file']).name} | "
            f"physical_RMSE={rmse:.4f}"
        )

    return {
        "overall_physical_rmse": overall_rmse,
        "per_example_physical_rmse": per_frame_rmse,
        "physical_preds": physical_preds,
        "physical_targets": physical_targets,
        "metadata": metadata,
    }


def compare_true_predictions(
    model,
    dataset,
    device,
    num_examples=3,
):
    model = model.to(device)
    model.eval()

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    chosen = np.linspace(0, len(dataset) - 1, num=min(num_examples, len(dataset)), dtype=int)
    physical_preds = []
    physical_targets = []
    metadata = []

    for idx in chosen:
        x, _ = dataset[idx]
        info = dataset.get_sample_info(idx)

        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu()[0]

        pred = (pred * dataset.std + dataset.mean).squeeze(0).numpy().astype(np.float32)

        source_path = Path(info.get("source_file", info["noise_file"]))
        source_arr = np.load(source_path)
        target = source_arr[info["target_idx"], info["level"]].astype(np.float32)

        physical_preds.append(pred)
        physical_targets.append(target)
        metadata.append({
            **info,
            "source_file": str(source_path),
        })

    physical_preds = np.stack(physical_preds, axis=0)
    physical_targets = np.stack(physical_targets, axis=0)

    overall_rmse, per_frame_rmse = visualize_and_rmse(
        physical_preds,
        physical_targets,
        suptitle="Test examples in physical space",
        cmap="RdBu_r",
    )

    for meta, rmse in zip(metadata, per_frame_rmse):
        print(
            f"dataset_idx={meta['dataset_idx']} | traj_id={meta['traj_id']} | "
            f"target_idx={meta['target_idx']} | source_file={Path(meta['source_file']).name} | "
            f"physical_RMSE={rmse:.4f}"
        )

    return {
        "overall_physical_rmse": overall_rmse,
        "per_example_physical_rmse": per_frame_rmse,
        "physical_preds": physical_preds,
        "physical_targets": physical_targets,
        "metadata": metadata,
    }
