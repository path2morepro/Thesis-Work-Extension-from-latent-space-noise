import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .cond_sampler import CondSampler
from .train import SAVE_PATH
from .dataset import SQGLeadTimeDataset, DATA_100, DATA_500, DATA_PATH



# capsulate the forecasting as a function
# here i input 5 data samples, but here only one forecasting result returns
def perform_invert(
    data_dir=DATA_500,
    invert_traj_num=100,
    parallel_trajs = 10,
    steps = 300       
):

    # there are parallel trajs for each batch
    bs, max_lead, levels, H, W = (parallel_trajs, 24, 2, 64, 64)
    # the dataset is used to extract the target and the condition
    # this would return a sampled trajectory
    target_dataset = SQGLeadTimeDataset(
        data_dir         = data_dir,
        split            = 'val',
        random_lead_time = False,
        eval_traj_num    = invert_traj_num
    )
    # for each iteration, dataloader would return (batch sieze, 24) frames
    # in this context, batch size actually means how many trajectories it would return

    loader = DataLoader(target_dataset, batch_size=bs, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = CondSampler(model_path=SAVE_PATH, device=device, steps=steps)
    save_dir = DATA_PATH / f'invert_steps={steps}'
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_no = 0

    for initial, target, lead_time, traj_idx, anchor_t in tqdm(loader, total=len(loader)):
        # target (bs, max_lead, levels, H, W)
        # initial (bs, levels, H, W)
        # leadtime (bs, max_lead)
        initial = initial.repeat_interleave(max_lead, dim=0) 
        lead_time = lead_time.reshape(bs*max_lead, ) # I don't know whether we need to transpose
        target = target.reshape(bs*max_lead, levels, H, W) # same as above

        batch_inverted = sampler.sample(z0=target, x_t=initial, lead_times=lead_time, invert=True)
        # batch_inverted_data (bs*max_lead, levels, H, W)
        batch_inverted = batch_inverted.reshape(bs, max_lead, levels, H, W).numpy()
        batch_no += 1

        for j in range(bs):
            fname = f"invert_traj{traj_idx[j].item()}initial{anchor_t[j].item()}.npy"
            np.save(save_dir / fname, batch_inverted[j])
     


if __name__ == '__main__':
    perform_invert()






