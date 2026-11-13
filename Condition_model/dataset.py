NORM_STD  = 2660

from collections import OrderedDict
from pathlib import Path
import re

import numpy as np
import torch
from numpy.lib import format as npformat
from torch.utils.data import Dataset
from natsort import natsorted


# Open memmaps held per process. Each one costs exactly one file descriptor,
# and a systemd user session starts with a soft limit of 1024 (hard 524288).
# Caching one per trajectory file meant a DataLoader worker accumulated 1200
# of them and epoch 1 died with "[Errno 24] Too many open files" at batch 119.
# Reopening is cheap -- np.load(mmap_mode='r') only reads the ~100-byte header.
MAX_OPEN_MEMMAPS = 128


THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"

# The split IS the directory — there is no in-folder split any more.
# Trajectory ids are global and never overlap between folders:
DATA_TRAIN = DATA_PATH / "train"      # 1200 files, ids    0..1199
DATA_VAL   = DATA_PATH / "val"        #  150 files, ids 1200..1349
DATA_TEST  = DATA_PATH / "test"       #  150 files, ids 1350..1499

DATA_test = DATA_TEST    # old spelling, still imported by forecasting.py etc.


def _npy_n_frames(path):
    """Number of frames T in an .npy file, read from its header alone.

    This touches ~100 bytes and never reads the array, so it is cheap enough to
    do for every file at startup. We need it PER FILE because the trajectories
    are not all the same length: 600 of the training files have 101 frames and
    600 have 100. A single global max_frames would either waste a frame on half
    of them or run off the end of the other half.
    """
    with open(path, 'rb') as fh:
        version = npformat.read_magic(fh)
        if version == (1, 0):
            shape, _, _ = npformat.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, _, _ = npformat.read_array_header_2_0(fh)
        else:
            raise ValueError(f'unsupported .npy version {version} in {path}')
    return shape[0]


class SQGLeadTimeDataset(Dataset):
    """Training pairs for p(x_{a+lead} | x_a, lead). Training only.

    One item is ONE pair, returned as (x0, xt, lead_label) — the order train.py
    unpacks and flow_matching_loss(model, x_t, x_t1, lead_time) expects.

    anchor : frame index inside a trajectory, 0 .. T-1-max_lead
    lead   : how many frames ahead the target is, 1 .. max_lead

    Splits are directories — pass DATA_TRAIN / DATA_VAL / DATA_TEST. Two
    datasets differ only by their path. Everything here is indexed by POSITION
    in the folder; 
    the trajectory id is carried along for traceability and is
    never used as an index, because ids and positions are different now
    (data/val holds ids 1200..1349 at positions 0..149).

    Anchors are redrawn every epoch, so capping them at `anchors_per_file` sets
    the epoch SIZE without discarding data: a 100-frame file has 76 valid
    anchors, and across epochs all of them get used. Call set_epoch(e) at the
    top of each epoch; __init__ calls set_epoch(0) so the dataset works at once..
    """

    def __init__(self, data_dir, anchors_per_file=10, max_lead=24,
                 std=NORM_STD, seed=42):
        """
        Args:
            data_dir         : directory of sqg_N64_*hrly_*.npy files
            anchors_per_file : anchors drawn per trajectory per epoch
                               (epoch size = n_files * anchors_per_file * max_lead)
            max_lead         : largest lead in frames; leads are 1..max_lead
            std              : normalisation std (NORM_STD = 2660)
            seed             : base seed; the anchors depend on (seed, epoch)
        """
        data_dir = Path(data_dir)
        files = natsorted(data_dir.glob('sqg_N64_*hrly_*.npy'))
        assert len(files) > 0, f"No sqg_N64_*hrly_*.npy files in {data_dir}"

        self.paths    = list(files)
        self.traj_ids = [int(re.search(r'_(\d+)\.npy$', f.name).group(1)) for f in files]

        self.n_anchors = []
        for f in files:
            T = _npy_n_frames(f)
            n = T - max_lead          # anchor a is valid iff a + max_lead <= T - 1
            if n <= 0:
                raise ValueError(
                    f'{f.name} has {T} frames, too short for max_lead={max_lead}'
                )
            self.n_anchors.append(n)

        self.std              = float(std)
        self.max_lead         = max_lead
        self.leads            = list(range(1, max_lead + 1))   # 1..24, no lead 0
        self.anchors_per_file = anchors_per_file
        self.seed             = seed

        # LRU, bounded by MAX_OPEN_MEMMAPS; filled lazily, one cache per process.
        self._mm = OrderedDict()     # file position -> open memmap
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Redraw this epoch's anchors and rebuild the index.

        Must be called in the parent process BEFORE the DataLoader starts
        iterating. With the default persistent_workers=False the workers are
        rebuilt each epoch and inherit the index made here; call it too late and
        the epoch silently runs on the previous epoch's anchors.

        The draw is stratified: each file's anchor range is cut into
        `anchors_per_file` contiguous bins and one anchor is taken from each, so
        every epoch covers the whole trajectory. A plain uniform draw could put
        all 10 anchors in the first fifth of the run.
        """
        rng = np.random.default_rng([self.seed, epoch])

        rows = []
        for pos, n in enumerate(self.n_anchors):
            k = min(self.anchors_per_file, n)
            # array_split copes with n/k not being an integer
            # (76 anchors in 10 bins -> six bins of 8 and four of 7).
            for b in np.array_split(np.arange(n), k):
                a = int(rng.choice(b))
                rows.extend((pos, a, lead) for lead in self.leads)

        self._pairs = np.asarray(rows, dtype=np.int32)      # (N, 3)
        self.epoch  = epoch

        # Drop memmaps opened in THIS process. The cache is only ever filled in
        # __getitem__, so inside a worker this is a no-op; it matters when the
        # main process has iterated the dataset itself (debugging), because
        # those memmaps would otherwise be inherited across the fork by the next
        # epoch's workers.
        self._mm.clear()

    def __len__(self):
        return len(self._pairs)

    def _memmap(self, pos):
        """Read-only memmap of one trajectory file, opened once per process.

        mmap_mode='r' is what keeps this cheap: slicing it pages in only the
        frames the slice touches, not the whole 3.3 MB file. Opened here rather
        than in __init__ so every DataLoader worker builds its own instead of
        inheriting one across a fork.
        """
        mm = self._mm.get(pos)
        if mm is not None:
            self._mm.move_to_end(pos)
            return mm

        # Evict the least recently used before opening another descriptor.
        # _frame() copies out of the memmap, so nothing holds a reference to
        # the evicted one and CPython closes it as soon as it is dropped.
        while len(self._mm) >= MAX_OPEN_MEMMAPS:
            self._mm.popitem(last=False)

        mm = np.load(self.paths[pos], mmap_mode='r')
        self._mm[pos] = mm
        return mm

    def _frame(self, mm, i):
        """One normalised frame as a (C, H, W) float32 tensor.

        Slice first, convert after. The files are float64 on disk, so
        converting the file and then indexing would read and upcast ~50x more
        than this needs.
        """
        x = np.asarray(mm[i], dtype=np.float32)
        x /= self.std
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        pos, anchor, lead = (int(v) for v in self._pairs[idx])
        mm = self._memmap(pos)

        x0 = self._frame(mm, anchor)                # (C, H, W) — conditioning anchor
        xt = self._frame(mm, anchor + lead)         # (C, H, W) — generation target
        lead_label = torch.tensor(                  # 0-dim -> collates to (B,)
            lead / self.max_lead, dtype=torch.float32
        )
        # order matches `for initial, target, lead_time in loader` in train.py
        return x0, xt, lead_label

    def __getstate__(self):
        # Never ship open memmaps to a worker process.
        state = self.__dict__.copy()
        state['_mm'] = OrderedDict()
        return state


class SQGDataset(Dataset):
    """Whole trajectories, for forecast evaluation.

    One item is one complete trajectory plus its id:
        traj    : (T, C, H, W) float32, already normalised
        traj_id : int, parsed from the filename

    ARCI rolls out to the end of a trajectory, so it needs all of it — there is
    nothing to sample here, and no memmap: the 150 test trajectories are about
    0.5 GB in float32, which is fine to hold. The id comes along so a forecast
    can always be traced back to the file it came from.

    Normalisation happens ONCE, at load. The previous version divided by std in
    __init__ and again in __getitem__, so every value came out std**2 (a factor
    of 2660) too small.
    """

    def __init__(self, data_path, mean=0, std=NORM_STD, limit=None):
        """
        Args:
            data_path : directory of .npy trajectory files
            mean, std : normalization stats
            limit     : keep only the first `limit` files. Everything here is
                        loaded eagerly -- 150 test trajectories is 491 MB --
                        so a run that only forecasts 50 of them should say so
                        rather than pay for the other 100.
        """
        self.mean = float(mean)
        self.std  = float(std)

        files = natsorted(Path(data_path).glob('*.npy'))
        assert len(files) > 0, f'No .npy files in {data_path}'
        if limit is not None:
            files = files[:limit]
        self.paths = list(files)

        self.traj_ids = []
        for f in files:
            m = re.search(r'_(\d+)\.npy$', f.name)
            assert m is not None, f'cannot read a trajectory id from {f.name}'
            self.traj_ids.append(int(m.group(1)))

        self.data = [
            torch.from_numpy((np.load(f).astype(np.float32) - self.mean) / self.std)
            for f in files
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.traj_ids[idx]
