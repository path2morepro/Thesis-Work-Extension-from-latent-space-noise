from scipy.optimize import minimize
import numpy as np

def matern32(t, ell):
    x = np.sqrt(3) * t / ell
    return (1+x) * np.exp(-x)

def exponential(t, ell):
    return np.exp(-t/ell)

def squared_exponential(t, ell):
    return np.exp(-0.5*(t/ell)**2)

def stretched_exponential(t, ell, p):
    return np.exp(-(t/ell)**p)

def neg_loglik(alpha, z0, zt):
    '''
    Args:
        alpha: (leads, )
        z0: (N, 2, 64, 64)
        zt: (N, leads, 2, 64, 64)

    Return:
        -product(log(p(zt | z0))) = -Sum_N Sum_T [-d/2log(1-alpha_t^2) - ||zt - alpha * z0||^2 / 2(1-alpha_t^2)]
    '''
    # alpha: (T,)    z0: (N, 2, 64, 64)    zt: (N, T, 2, 64, 64), lags t = 1..T
    D = np.prod(z0.shape[1:])                       # 8192
    a = np.clip(alpha, 0, 1 - 1e-6)[None, :, None, None, None]
    diff = zt - a * z0[:, None]                     # (N, T, 2, 64, 64)
    sq = np.sum(diff**2, axis=(2, 3, 4))            # (N, T)
    var = 1 - a[..., 0, 0, 0]**2                    # (1, T)
    ll = -D/2 * np.log(var) - sq / (2 * var)
    return -np.sum(ll)

def objective(log_params, kernel, z0, zt):
    ell = np.exp(log_params)
    t = np.arange(1, zt.shape[1] + 1)
    return neg_loglik(kernel(t, ell), z0, zt)

def objective_stretch(log_params, z0, zt):
    ell, p = np.exp(log_params)
    t = np.arange(1, zt.shape[1] + 1)
    return neg_loglik(stretched_exponential(t, ell, p), z0, zt)


def RMSE(estimation, truth, t):
    rmse = np.sqrt(np.mean((truth[:, 0:t, ...] - estimation[0:t])**2))
    return rmse