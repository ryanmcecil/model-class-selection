########################################
# Functions to compute omega value for universal inference method in simulations.
########################################
import numpy as np
from functools import partial
import os

def compute_scc(i, data_split, dgp, alg_null, alg_alt, n):
    x = dgp(len(data_split[0]) + len(data_split[1]))
    alg_model = alg_alt(x[data_split[0]])
    alg_null_model = alg_null(x[data_split[0]])
    vals = alg_null_model.score_samples(x[data_split[1]], n) - alg_model.score_samples(x[data_split[1]], n)
    return np.sum(vals)

def estimate_scc(dgp, alpha, data_split, alg_null, alg_alt, n, B, pool, np_file,
                 omegas=np.logspace(-4, 2, 2000)):
    omegas = np.concatenate([[0.0], omegas])
    if os.path.exists(np_file):
        cached = np.load(np_file)
        coverage = cached['coverage']
    else:
        func = partial(compute_scc, dgp=dgp, data_split=data_split,
                       alg_null=alg_null, alg_alt=alg_alt, n=n)
        out = np.asarray(pool.map(func, range(B)))
        # phi_UI rejects iff exp(omega * S) > 1/alpha  <=>  omega * S > log(1/alpha)
        coverage = np.mean(omegas * out[:, None] <= np.log(1 / alpha), axis=0)
        np.savez(np_file, omegas=omegas, coverage=coverage, alpha=alpha)
    # Dey et al. (2025, Alg. 2): coverage closest to 1 - alpha; ties broken toward largest omega
    dist = np.abs(coverage - (1 - alpha))
    idx = np.flatnonzero(dist == dist.min()).max()
    if idx == len(omegas) - 1:
        print(f'Note: type I error below alpha on the whole grid ({np_file}); omega_or = grid max.')
    return omegas[idx]
