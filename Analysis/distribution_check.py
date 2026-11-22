import numpy as np
from utils import *

invert_path = ROOT.parent / 'data' / 'invert' / 'distribution_check_val'
invert_path_rk4 = ROOT.parent / 'data' / 'invert' / 'from_val_rk4'

def distribution_check(path, ds='euler'):
    Z, _, _ = load_invert(path)
    N, L, C, H, W = Z.shape
    D = C * H * W
    Zf = Z.reshape(N, L, D)
    ref = np.random.randn(N, L, C, H, W)

    pss, pz, pchi2 = pixel_level_test(Z, ds=ds)
    config = dict(save=True, fine_bins=True, ds=ds)
    p_histogram(pss, "Sharpiral-Wilk", **config)
    p_histogram(pz, "z-test", **config)
    p_histogram(pchi2,  "chi-square", flex_scale=True, **config)
    zf_mean, zf_var = global_parameters(Z)
    ref_mean, ref_var = global_parameters(ref)
    plot_global_parameters((zf_mean, zf_var), (ref_mean, ref_var), save=True, ds=ds)
    power_spectrum(Z, save=True, ds=ds)

distribution_check(invert_path)
distribution_check(invert_path_rk4, ds='rk4')








