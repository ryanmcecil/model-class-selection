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

def estimate_scc(dgp, alpha, data_split, alg_null, alg_alt, n, B, pool, np_file):
    omegas = np.linspace(0,1,10000)
    print(np_file)
    if os.path.exists(np_file):
        estimates = np.load(np_file)
    else:
        func = partial(compute_scc, dgp = dgp, data_split = data_split, alg_null = alg_null, alg_alt=alg_alt, n = n)
        out = np.asarray(pool.map(func, range(B)))
        vals = np.exp(omegas*np.expand_dims(out, axis=-1))
        covereage_rates = np.mean(vals < 1/alpha, axis=0)
        estimates = np.abs(covereage_rates - (1-alpha))
        np.save(np_file, estimates)
    idx = np.argmin(estimates)
    return omegas[idx]






