import xarray as xr
import numpy as np
import os
import json
from tqdm import tqdm
import torch
import pandas as pd
import datetime
from utils import ERA5Dataset
from torch.utils.data import DataLoader


file_directory = "ERA5_DATA_LOCATION"
save_directory = "./data"
os.makedirs(save_directory, exist_ok=True)

field_names = {
    "orography": "orog",
    "lsm": "lsm",
}

var_names = {
    "geopotential_500": ("z500", "z"),
    "temperature_850": ("t850", "t"),
    "2m_temperature": ("t2m", "t2m"),
    "10m_u_component_of_wind": ("u10", "u10"),
    "10m_v_component_of_wind": ("v10", "v10"),
}

chunk_size = 1000

# Constants
var_name = "constants"
file_pattern = f"{file_directory}/{var_name}/{var_name}*.nc"
df = xr.open_mfdataset(file_pattern, combine='by_coords')

lat = df['lat'].values
lon = df['lon'].values
np.savez(f'{save_directory}/latlon_1979-2018_5.625deg.npz', lat=lat, lon=lon)

## Static fields
static_fields = []
save_name = ''
for field_name, var_name in field_names.items():
    data_array = df[field_name].values
    static_fields.append(data_array)
    save_name += var_name + '_'

np.save(f'{save_directory}/{save_name}1979-2018_5.625deg.npy', np.stack(static_fields, axis=0))

# Variables
file_prefix, names = list(var_names.items())[0]
var_name = names[0]
short_name = names[1]
file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
combined_shape = (ds[short_name].shape[0], len(var_names), ds[short_name].shape[1], ds[short_name].shape[2])
print("Shape:", combined_shape)

save_name = '_'.join([var_name[0] for var_name in var_names.values()])
memmap_file_path = f'{save_directory}/{save_name}_1979-2018_5.625deg.npy'
memmap_array = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=combined_shape)

statistics = {}
mean_value = 0
std_value = 0

i = 0
for file_prefix, names in var_names.items():
    var_name = names[0]
    short_name = names[1]

    file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
    print(f"Opening: {file_pattern}")
    
    # Open the dataset with dask for efficient memory handling
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
    array = ds[short_name]
    
    # Loop through the chunks of the array and write each chunk to the memmap file
    # We will iterate through the time dimension (0th axis) chunk by chunk
    for j in tqdm(range(0, array.shape[0], chunk_size)):
        end_idx = min(j + chunk_size, array.shape[0])  # Ensure we don't go out of bounds
        
        # Convert the chunk to a NumPy array and assign it to the corresponding memmap slice
        chunk = array[j:end_idx, :, :].compute()  # Compute the chunk lazily
        memmap_array[j:end_idx, i, :, :] = chunk

        # Calculate the mean and std of the chunk
        mean_value += np.sum(chunk).values
        std_value += np.sum(chunk ** 2).values
    
    # Calculate the mean and std of the variable
    num_elements = array.size
    mean_value /= num_elements
    std_value = np.sqrt(std_value / num_elements - mean_value ** 2)
    statistics[var_name] = {"mean": float(mean_value),"std": float(std_value)}

    i += 1

print(f"Combined data saved as memory-mapped file: {memmap_file_path}")

# Save the statistics to a JSON file
json_file = f'{save_directory}/norm_factors.json'
with open(json_file, 'w') as f:
    json.dump(statistics, f, indent=4)

print(f"Normalization factors saved to {json_file}")

# Print out the mean and std for each variable
for var_name, stats in statistics.items():
    print(f"{var_name}: Mean = {stats['mean']}, Std = {stats['std']}")

## Calculate residual stds

variable_names = [k[0] for k in var_names.values()]

mean_data = torch.tensor([stats["mean"] for (key, stats) in statistics.items() if key in variable_names])
std_data = torch.tensor([stats["std"] for (key, stats) in statistics.items() if key in variable_names])
norm_factors = np.stack([mean_data, std_data], axis=0)

# Get the number of samples, training and validation samples
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_samples, n_train, n_val = len(ti), sum(ti.year <= 2015), sum((ti.year >= 2016) & (ti.year <= 2017))

kwargs = {
            'dataset_path':     f'{save_directory}/{save_name}_1979-2018_5.625deg.npy',
            'sample_counts':    (n_samples, n_train, n_val),
            'dimensions':       (len(var_names), 32, 64),
            'max_horizon':      240, # For scaling the time embedding
            'norm_factors':     norm_factors,
            'device':           'cpu',
            'spacing':          1,
            'dtype':            'float32',
            'conditioning_times':    [0],
            'lead_time_range':  (1, 240, 1),
            'static_data_path': None,
            'random_lead_time': 0,
            }

stds_directory = f"{save_directory}/residual_stds"
os.makedirs(stds_directory, exist_ok=True)

def calculate_residual_mean_std(loader):
    # --- Math symbols used below ---
    # τ        : the forecast lead time (e.g. 6 hours). Fixed for one call; the outer loop varies it.
    # x_t      : the atmospheric state at time t  — shape (C, H, W), C variables, H×W spatial grid.
    # x_{t+τ}  : the atmospheric state τ hours later.
    # r_{t,τ}  : the residual (forecast increment) for one sample, defined as  r_{t,τ} = x_{t+τ} − x_t
    # μ_τ      : the mean of r_{t,τ} over all training times t and all grid points, per variable.
    # σ_τ      : the std  of r_{t,τ} over all training times t and all grid points, per variable.
    #
    # --- What "residual" means here ---
    # The model does NOT predict x_{t+τ} directly.
    # It predicts the *change* from the initial condition: r_{t,τ} = x_{t+τ} − x_t.
    # This difference is called the "residual" because it is what remains after you subtract
    # the part of the future state that is already explained by just standing still (x_t itself).
    # At τ=1h the residual is tiny (weather barely moves in one hour).
    # At τ=240h the residual is large (weather has diverged substantially).
    # σ_τ therefore grows with lead time.
    #
    # --- Why we need σ_τ ---
    # In loss.py the training loss is divided by σ_τ (see residual_scaling).
    # Without this division, a long-lead sample (large σ_τ, large raw loss value) would dominate
    # gradient updates and the model would effectively ignore short-lead accuracy.
    # Nice design! But how you implement that?
    # Dividing by σ_τ puts every lead time on the same scale: a "one-sigma miss" at τ=6h
    # counts the same as a "one-sigma miss" at τ=240h.

    # Question: the sigma_residual is the variance of residual right? there is no business with ensemble right?
    # Question: which dimension would sigma_residual calculate? (bs, frame, level, H, W), which one?
    mean_data_latent, std_data_latent, count = 0.0, 0.0, 0

    with torch.no_grad():
        for current, next, _ in loader:
            # current : x_t,    shape (B, C, H, W)   — B samples in the batch
            # next    : x_{t+τ}, shape (B, C, H, W)  — τ is whatever lead time the dataset is set to
            inputs = next - current
            # inputs is now r_{t,τ} for every sample in the batch, shape (B, C, H, W).

            count += inputs.size(0)
            # count accumulates the number of samples B seen so far (only one batch here, see break below).

            mean_data_latent += torch.sum(inputs, dim=(0,2,3))
            # Sum over the batch axis (0) and both spatial axes (2=H, 3=W), keeping the variable axis (1).
            # Result shape: (C,) — one running sum per variable.
            # Summing space here is valid because we will divide by the total number of elements. ok, ok but...really?
            # (batch × H × W) at the end, giving the spatial average.

            std_data_latent += torch.sum(inputs ** 2, dim=(0,2,3))
            # Same axes. Accumulates Σ r² per variable, needed for Var = E[r²] − (E[r])².

            break # Calculating for a single batch is sufficient
            # One large batch (bs=10000 samples) is used as a Monte-Carlo estimate of μ_τ and σ_τ.
            # This is an approximation, but 10 000 × H × W values is enough for a stable std estimate.

        count = count * inputs[0, 0].cpu().detach().numpy().size
        # inputs[0, 0] is one spatial map (H, W) for one variable of one sample.
        # .size gives H×W — the number of grid points.
        # So count is now  B × H × W: the total number of scalar values summed above per variable.

        mean_data_latent /= count
        # μ_τ[c] = (Σ_{b,i,j} r_{b,c,i,j}) / (B × H × W)   for each variable c.

        std_data_latent = torch.sqrt(std_data_latent / count - mean_data_latent ** 2)
        # Standard identity:  Var(r) = E[r²] − (E[r])²
        # std_data_latent / count  gives  E[r²]  per variable.
        # Subtracting mean_data_latent² gives the variance.
        # sqrt gives σ_τ[c] for each variable c.

    return mean_data_latent, std_data_latent
    # Returns (μ_τ, σ_τ), each shape (C,) — one value per atmospheric variable.
    # The outer loop calls this for every τ in [1, 240] and saves σ_τ to disk.
    # Those files are later loaded in train.py as `residual_stds` and passed to WGCLoss.

lead_time = 1
max_lead_time = 240

bs = 10000
train_dataset = ERA5Dataset(lead_time=lead_time, dataset_mode='train', **kwargs)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

ts = np.arange(lead_time, max_lead_time + 1, 1)

stds_dict = {var_name: [] for var_name in variable_names}

for t in (ts):
    train_dataset.set_lead_time(t)
    
    mean_t, std_t = calculate_residual_mean_std(train_loader)
    print(t, std_t)
    for i, var_name in enumerate(stds_dict):
        stds_dict[var_name].append(std_t[i].item())
    
for var_name, stds in stds_dict.items():
    stds_content = "\n".join([f"{ts[i]} {std}" for i, std in enumerate(stds)])
    
    file_path = f"{stds_directory}/WB_{var_name}.txt"
    with open(file_path, "w") as file:
        file.write(stds_content)

    print(f"Standard deviations for {var_name} saved to {file_path}")
