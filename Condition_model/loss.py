import torch


def flow_matching_loss(model, x_t, x_t1, lead_time):
    """
    Conditional flow-matching loss for one batch.

    Args:
        model     : SongUNet with in_channels = 2*C
        x_t       : current state  (B, C, H, W) — the conditioning input
        x_t1      : next state     (B, C, H, W) — the generation target
        lead_time : lead time      (B,)         — scalar condition per sample

    Returns:
        scalar loss
    """
    B = x_t1.shape[0]

    # sample interpolant time s ~ Uniform[0, 1]
    s         = torch.rand(B, device=x_t1.device)       # (B,)
    s_spatial = s.view(B, 1, 1, 1)                       # broadcast over C, H, W

    # sample base noise
    z0 = torch.randn_like(x_t1)                          # (B, C, H, W)

    # form interpolated noisy sample at time s
    z_s = (1 - s_spatial) * z0 + s_spatial * x_t1        # (B, C, H, W)

    # target velocity (same at every s along the linear path)
    target = x_t1 - z0                                   # (B, C, H, W)

    # model prediction
    # noise_labels = s   (interpolant time plays the role of noise level)
    # class_labels = x_t (spatially concatenated inside SongUNet before the encoder)
    # time_labels  = lead_time (scalar lead-time condition)
    pred = model(z_s, s, class_labels=x_t, time_labels=lead_time)   # (B, C, H, W)

    return ((pred - target) ** 2).mean()
