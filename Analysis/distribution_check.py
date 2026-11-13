import numpy as np
from utils import *

invert_path = ROOT.parent / 'data' / 'invert' / 'distribution_check_val'
invert_path_rk4 = ROOT.parent / 'data' / 'invert' / 'from_val_rk4'


Z, _, _ = load_invert(invert_path)
# N, L, C, H, W = Z.shape
# D = C * H * W
# Zf = Z.reshape(N, L, D)
# ref = np.random.randn(N, L, C, H, W)

# pss, pz, pchi2 = pixel_level_test(Z)
# p_histogram(pss, "Sharpiral-Wilk", save=True, fine_bins=True)
# p_histogram(pz, "z-test", save=True, fine_bins=True)
# p_histogram(pchi2,  "chi-square", flex_scale=True, save=True, fine_bins=True)
# zf_mean, zf_var = global_parameters(Z)
# ref_mean, ref_var = global_parameters(ref)
# plot_global_parameters((zf_mean, zf_var), (ref_mean, ref_var), save=True)
power_spectrum(Z, save=True)
Z_rk4, _, _ = load_invert(invert_path_rk4)
power_spectrum(Z_rk4, save=True, ds='rk4')




