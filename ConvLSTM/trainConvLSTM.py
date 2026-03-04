import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from convLSTM import ConvLSTMForecaster



def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=1e-3,
    alpha=0.9,
    patience=10,
    min_delta=0.0,
    save_path=MODELS_DIR / "best_convlstm.pt",
    show_progress=True,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha)

    model = model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": None,
    }

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
            leave=False,
            disable=not show_progress,
        )
        for x, y in train_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size
            train_bar.set_postfix(batch_loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
            leave=False,
            disable=not show_progress,
        )
        with torch.no_grad():
            for x, y in val_bar:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                batch_size = x.shape[0]
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size
                val_bar.set_postfix(batch_loss=f"{loss.item():.6f}")

        avg_val_loss = val_loss_sum / max(val_count, 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"train_loss: {avg_train_loss:.6f} - val_loss: {avg_val_loss:.6f}",
            flush=True,
        )

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss

    return model, history




def plot_training_history(history):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ConvLSTM Training Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, loader, device, num_examples=3):
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

    x = x.cpu()
    y = y.cpu()
    pred = pred.cpu()

    num_examples = min(num_examples, x.shape[0])
    fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3 * num_examples))
    if num_examples == 1:
        axes = axes[None, :]

    for i in range(num_examples):
        last_input = x[i, -1, 0]
        target = y[i, 0]
        prediction = pred[i, 0]
        error = prediction - target

        vmax = max(target.abs().max().item(), prediction.abs().max().item())
        err_max = max(error.abs().max().item(), 1e-8)

        images = [last_input, target, prediction, error]
        titles = ["Last Input Frame", "Target Frame", "Predicted Frame", "Prediction Error"]
        cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "coolwarm"]
        limits = [None, (-vmax, vmax), (-vmax, vmax), (-err_max, err_max)]

        for j, ax in enumerate(axes[i]):
            vlim = limits[j]
            if vlim is None:
                im = ax.imshow(images[j], cmap=cmaps[j])
            else:
                im = ax.imshow(images[j], cmap=cmaps[j], vmin=vlim[0], vmax=vlim[1])
            ax.set_title(titles[j])
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def load_model_checkpoint(
    checkpoint_path=None,
    device=None,
    model_kwargs=None,
):
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / "best_convlstm.pt"
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_kwargs is None:
        model_kwargs = dict(input_channels=1, hidden_channels=(16, 16), input_frames=10)

    model = ConvLSTMForecaster(**model_kwargs).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# from convLSTM import ConvLSTMForecaster
# from dataset import NoiseDataset
# from sampler import Sampler

# train_dataset = NoiseDataset(folder_path="../data/inverted_sqg_subset", split="train")
# val_dataset = NoiseDataset(folder_path="../data/inverted_sqg_subset", split="val")

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ConvLSTMForecaster(input_channels=1, hidden_channels=(16, 16), input_frames=10)
# model, history = train(model, train_loader, val_loader, device, num_epochs=100, patience=10)
# plot_training_history(history)
# visualize_predictions(model, val_loader, device, num_examples=3)
