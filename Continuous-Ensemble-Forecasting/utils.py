import numpy as np
import torch
from torch.utils.data import Sampler

class ERA5Dataset(torch.utils.data.Dataset):
    # --- Overview of how a training pair (x_0, x_τ) is built ---
    #
    # The raw data is one giant time-series array of shape (T, C, H, W):
    #   T = total hourly timesteps (~350 000 for 1979-2018)
    #   C = number of atmospheric variables (e.g. 5)
    #   H, W = spatial grid (lat × lon)
    #
    # Each call to __getitem__(idx) returns ONE pair:
    #   x_0  — the initial condition, one or more snapshots around time t
    #   x_τ  — the target state τ hours later
    #   τ    — the lead time itself (returned as a label for the loss and network)
    #
    # Step 1 — _generate_indices() pre-computes the set of valid "anchor" timestamps.
    #           An anchor is just a raw integer index into the T axis of the memmap.
    #           Only anchors where anchor + max_horizon < T are kept, so x_τ never
    #           reads past the end of the file for ANY possible τ.
    #
    # Step 2 — get_lead_time() samples τ each time a sample is requested.
    #           If random_lead_time=1 it picks τ uniformly at random from the discrete
    #           grid {t_min, t_min+Δt, t_min+2Δt, …, t_max}.
    #           If random_lead_time=0 it returns the fixed self.lead_time (used at eval).
    #
    # Step 3 — __getitem__(idx) uses the anchor and τ to index the memmap:
    #           x_0  ← mmap[anchor + conditioning_times]   (one or more past/current frames)
    #           x_τ  ← mmap[anchor + τ]                    (the future frame)
    #
    # The DynamicKBatchSampler (below) adds a SECOND source of τ variation:
    # it calls set_lead_time(new_τ) before each batch so that within a single epoch
    # every batch can have a different τ, ensuring the model sees the full lead-time
    # range each epoch even when random_lead_time=0.

    def __init__(self,
                 dataset_path,     # str: Path to the dataset files.
                 dataset_mode,     # str: Dataset dataset_mode ('train', 'val', 'test').
                 sample_counts,    # tuple: Total, training, and validation sample counts (total_samples, train_samples, val_samples).
                 dimensions,        # tuple: Dimensions of the dataset (variables, latitude, longitude).
                 lead_time,        # int: Current lead time for forecasting.
                 max_horizon,    # int: Maximum lead time we want to forecast. Used for not going outside dataset
                 norm_factors,     # tuple: Mean and standard deviation for normalization (mean, std_dev).
                 device,           # torch.device: Device on which tensors will be loaded.
                 lead_time_range,  # Range of lead time
                 spinup = 0,       # int: Number of samples to discard at the start for stability.
                 spacing = 1,      # int: Sample selection interval for data reduction.
                 dtype='float32',   # str: Data type of the dataset (default 'float32').
                 conditioning_times=[0,], # list: Times to condition on for forecasting. 
                 static_data_path = None, # str: Path to the static data file.
                 random_lead_time = 0, # bool: Whether to use random lead time

                ):
        """
        Initialize a custom Dataset for lazily loading WB samples from a memory-mapped file,
        which allows for efficient data handling without loading the entire dataset into memory.
        """
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device

        self.dataset_mode = dataset_mode
        self.n_samples, self.n_train, self.n_val = sample_counts
        self.num_variables, self.n_lat, self.n_lon = dimensions
        self.max_horizon = max_horizon
        self.lead_time = lead_time
        self.spinup = spinup + 24 # Change this if we ever look back more than 24h
        self.spacing = spacing
        self.mean, self.std_dev = norm_factors
        self.t_min, self.t_max, self.delta_t = lead_time_range

        self.static_data_path = static_data_path
        self.static_fields = None

        self.static_vars = 0
        if static_data_path != None:
            self.static_fields = self.load_static_data()      
            self.static_vars = self.static_fields.shape[1]
        
        self.conditioning_times = conditioning_times
        self.input_times = self.num_variables * len(self.conditioning_times)
        self.output_times = self.num_variables * (len(self.lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)

        self.index_array = self._generate_indices()

        self.mmap = self.create_mmap()

        self.random_lead_time = random_lead_time


    def create_mmap(self):
        """Creates a memory-mapped array for the dataset to facilitate large data handling."""
        # I don't understand this part, is this useful for me?
        return np.memmap(self.dataset_path, dtype=np.float32, mode='r', shape=(self.n_samples, self.num_variables, self.n_lat, self.n_lon))

    def load_static_data(self):
        """Load and normalize static fields."""
        static_fields = np.load(self.static_data_path)

        min_vals = np.min(static_fields, axis=(1,2))
        max_vals = np.max(static_fields, axis=(1,2))

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Replace zero range with one to avoid division by zero

        # Apply min-max scaling: (x - min) / (max - min)
        scaled_static_fields = (static_fields - min_vals[:, None, None]) / range_vals[:, None, None]
        return scaled_static_fields

    def _generate_indices(self):
        """Generates indices for dataset partitioning according to the specified dataset_mode."""
        # These indices are the valid "anchor" positions in the raw time-series array.
        # An anchor at position i means: x_0 = mmap[i], x_τ = mmap[i + τ].
        #
        # The split boundaries:
        #   train : [spinup,              n_train)
        #   val   : [spinup + n_train,    n_train + n_val)
        #   test  : [spinup + n_train + n_val,  n_samples)
        #
        # spinup (default 24h) discards the first 24 timesteps of each split.
        # This ensures that conditioning_times offsets like -6 or -12 never
        # reach before the very start of the array.
        if self.dataset_mode == 'train':
            start, stop = self.spinup, self.n_train
        elif self.dataset_mode == 'val':
            start, stop = self.spinup + self.n_train, self.n_train + self.n_val
        elif self.dataset_mode == 'test':
            start, stop = self.spinup + self.n_train + self.n_val, self.n_samples

        # stop - self.max_horizon: the upper bound is shrunk by max_horizon.
        # This guarantees that anchor + max_horizon < T for every stored anchor,
        # so that even the longest possible τ = t_max never reads out of bounds.
        # Example: if T=350 000 and max_horizon=240, the last valid anchor is T-241.
        #
        # [::self.spacing] sub-samples anchors every `spacing` steps.
        # spacing=1 uses every hourly timestep; spacing=6 uses every 6th hour, etc.
        # This is purely a data-reduction knob; it does not affect how τ is sampled.
        return np.arange(start, stop - self.max_horizon)[::self.spacing]

    def set_lead_time(self, lead_time):
        """ Updates the lead time lead_time for generating future or past indices."""
        self.lead_time = lead_time

    def set_lead_time_range(self, lead_time_range):
        self.t_min, self.t_max, self.delta_t = lead_time_range

    def get_lead_time(self):
        # This is WHERE τ is chosen for each sample.
        #
        # Case 1 — random_lead_time=1 (used during training):
        #   τ is sampled uniformly at random from the discrete grid:
        #   {t_min, t_min + delta_t, t_min + 2*delta_t, …, t_max}
        #   The number of grid points is  1 + (t_max - t_min) // delta_t.
        #   Example: t_min=1, t_max=240, delta_t=1 → 240 possible lead times,
        #   each chosen with probability 1/240 for every individual sample.
        #   This means within a single epoch each anchor is paired with a
        #   DIFFERENT randomly drawn τ each time it is visited.
        #
        # Case 2 — random_lead_time=0 (used during evaluation / create_dataset.py):
        #   τ is fixed at self.lead_time, which is set externally via set_lead_time().
        #   create_dataset.py loops over all τ in [1, 240] and calls set_lead_time(t)
        #   before each pass to compute residual stds at every specific lead time.
        #   predict.py also uses this path so it can request exactly the lead times
        #   it wants to forecast.
        if self.random_lead_time:
            return self.t_min + self.delta_t * torch.randint(0, 1 + (self.t_max - self.t_min) // self.delta_t, (1,), device=self.device)[0]
        return self.lead_time

    def __len__(self):
        """Returns the number of samples available in the dataset based on the computed indices."""
        return self.index_array.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and its corresponding future or past state from the dataset."""
        # --- Step 1: resolve the anchor timestamp ---
        start_index = self.index_array[idx]
        # start_index is an absolute integer position in the raw memmap array.
        # It represents the "now" moment for this sample — i.e., the time of x_0.
        # It was pre-validated by _generate_indices so that start_index + max_horizon < T.

        # --- Step 2: sample the lead time τ ---
        lead_times = self.get_lead_time()
        # lead_times is τ, an integer number of hours (e.g. 47).
        # It is drawn randomly during training (random_lead_time=1) or fixed during eval.
        # Every call to __getitem__ can return a DIFFERENT τ, so consecutive samples
        # in the same batch (if DynamicKBatchSampler is NOT used) may have different lead times.
        # With DynamicKBatchSampler the whole batch shares one τ set before the batch starts.

        # --- Step 3: build the index arrays for x_0 and x_τ ---
        x_index = start_index + self.conditioning_times
        # conditioning_times is a list of integer offsets relative to start_index.
        # Example: conditioning_times = [0, -6]
        #   → x_index = [start_index, start_index - 6]
        #   → loads the frame AT the IC time AND the frame 6 hours before it.
        # This gives the network a "velocity" hint: it can see how the atmosphere
        # was moving before the IC, which helps medium-range skill.
        # Example: conditioning_times = [0]
        #   → x_index = [start_index]
        #   → loads only the IC frame itself (single-frame conditioning).

        y_index = start_index + lead_times
        # y_index is a single integer: the raw array position of x_τ.
        # x_τ = mmap[start_index + τ]  — the atmospheric state τ hours after the IC.
        # This is the TARGET the model must learn to produce from x_0 and τ.

        # --- Step 4: load from the memory-mapped file ---
        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        # Shape: (len(conditioning_times), C, H, W) — one frame per offset in conditioning_times.
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)
        # Shape: (C, H, W) if lead_time is a scalar, or (n_lead_times, C, H, W) if a list.
        # Reading from memmap is lazy: only the requested rows are paged in from disk.

        # --- Step 5: normalize both frames to zero mean / unit std ---
        X_sample = (X_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]
        Y_sample = (Y_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]
        # self.mean and self.std_dev are the global per-variable statistics computed in
        # create_dataset.py and saved to norm_factors.json.
        # After this, each variable has approximately mean=0, std=1 across the training set.
        # The model works in this normalized space; renormalize() in train/predict undoes it for metrics.

        # --- Step 6: flatten the time and variable axes into a single channel axis ---
        X_sample = torch.tensor(X_sample, dtype=torch.float32).view(self.input_times, self.n_lat, self.n_lon)
        # self.input_times = num_variables * len(conditioning_times).
        # .view() merges the (len(conditioning_times), C) prefix into a flat channel dim.
        # Result shape: (input_times, H, W) — this is what the network sees as "class_labels".
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).view(self.output_times, self.n_lat, self.n_lon)
        # Result shape: (output_times, H, W) — this is the regression target in the loss.

        # --- Step 7: append static fields to the conditioning channels ---
        if self.static_vars != 0:
            X_sample = np.concatenate([X_sample, self.static_fields], axis=0)
        # static_fields (orography, land-sea mask) are appended as extra channels.
        # They never change with time, so they are stored once and tiled into every sample here.
        # Final X_sample shape: (input_times + static_vars, H, W).

        return X_sample, Y_sample, lead_times
        # Returns:
        #   X_sample  — x_0 (+ optional past frames + static fields), the initial condition
        #   Y_sample  — x_τ, the target future state τ hours ahead
        #   lead_times — τ, passed to the network as time_labels and to the loss for residual scaling


def get_uniform_t_dist_fn(t_min, t_max, delta_t):
    """ Create the update function """
    # Returns a callback function that, when called with a dataset, sets the dataset's
    # lead_time to a freshly sampled τ drawn uniformly from {t_min, t_min+Δt, …, t_max}.
    # This callback is handed to DynamicKBatchSampler, which fires it once per batch.
    # The result: every batch in the epoch trains on a DIFFERENT τ, cycling through
    # the full lead-time range even when random_lead_time=0 inside __getitem__.

    def uniform_t_dist(dataset):
        new_lead_time = t_min + delta_t * np.random.randint(0, 1 + (t_max - t_min) // delta_t)
        dataset.set_lead_time(new_lead_time)

    return uniform_t_dist

class DynamicKBatchSampler(Sampler):
    # --- Why this class exists ---
    # A standard DataLoader assigns one fixed τ for the whole epoch (whatever self.lead_time is).
    # DynamicKBatchSampler breaks the epoch into batches and re-samples τ before EACH batch.
    # This is the mechanism that lets the model train on ALL lead times within one pass over
    # the data, without multiplying the dataset size by 240. (why you mention this suddenly, seemingly irrelavent)
    #
    # Execution order for each batch:
    #   1. Collect batch_size indices from the shuffled index list.
    #   2. Call t_update_callback(dataset)  →  dataset.set_lead_time(new_τ).
    #   3. yield the batch indices to the DataLoader.
    #   4. DataLoader calls __getitem__ for each index — they all use the SAME new_τ
    #      because set_lead_time already updated it.
    #
    # So every sample within one batch shares the same τ, but τ varies across batches.
    # This is important: the loss's residual_scaling divides by σ_τ, which is a
    # per-batch scalar, so it only makes sense when all samples in the batch have the same τ.
    # residual_scaling, I mean normalization is really big and important part in this pipeline
    # require to figure it out at first

    def __init__(self, dataset, batch_size, drop_last, t_update_callback, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.t_update_callback = t_update_callback
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        # Shuffle indices at the beginning of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.indices)

        batch = []
        for idx in self.indices:
            if len(batch) == self.batch_size:
                self.t_update_callback(self.dataset)  # Update `lead_time` before yielding the batch
                # τ is set HERE, before the batch is handed to the DataLoader workers.
                # All batch_size calls to __getitem__ that follow will read this new τ.
                yield batch
                batch = []
            batch.append(idx)
        if batch and not self.drop_last:
            self.t_update_callback(self.dataset)  # Update `lead_time` for the last batch if not dropping it
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
