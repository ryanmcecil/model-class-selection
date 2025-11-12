########################################
# Feature selection experiment
########################################
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from mcsm import MCSM
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.linalg import toeplitz
import itertools
import numpy as np
import os
from matplotlib.lines import Line2D
from config import Env
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import argparse
from estimate_scc import estimate_scc
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument(
    "-model_type", help="Model Type to Consider", type=str, default='mlp'
)
args = parser.parse_args()

class Alg():
    def __init__(self, model_type: str = 'mlp', coef_to_use = None, coef = None):
        if model_type == 'mlp':
            self.model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, )
        elif model_type == 'linear':
            self.model = LinearRegression(fit_intercept=False)
        else:
            raise Exception()
        self.coef_to_use = coef_to_use
        self.coef = coef

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        if self.coef_to_use is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X[:, self.coef_to_use], y)
        return self
    
    def score_samples(self, data, n):
        X = data[:, :-1]
        y = data[:, -1]
        if self.coef is None:
            if self.coef_to_use is None:
                yhat = self.model.predict(X)
            else:
                yhat = self.model.predict(X[:, self.coef_to_use])
        else:
            yhat = X @ self.coef
        residuals = y - yhat
        return residuals**2
    
def linear_model_sample(n, p = 10, s = 5, rho = 0.35, snr = 1, seed = None):
    rng = np.random.default_rng(seed)
    c = rho ** np.arange(p)
    Sigma = toeplitz(c)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

    beta = np.zeros(p)
    positions = np.asarray([True, False] * (p // 2))
    beta[positions] = 1

    signal_var = beta @ Sigma @ beta
    nu = snr

    noise_std = np.sqrt(signal_var / nu)
    eps = rng.normal(0, noise_std, size=n)

    y = X @ beta + eps

    return np.concatenate([X, y.reshape(-1, 1)], axis=1)


def generate_boolean_combinations_except_all(p):
    if p < 1:
        return []

    all_combinations = list(itertools.product([False, True], repeat=p))

    # Exclude the all-false and all-true combinations
    excluded_combinations = []
    if all_combinations:
        excluded_combinations.append(tuple([False] * p))
        excluded_combinations.append(tuple([True] * p))
        excluded_combinations.append(tuple([True, False] * (p // 2)))

    filtered_combinations = [
        combo for combo in all_combinations if combo not in excluded_combinations
    ]

    filtered_combinations = filtered_combinations + [tuple([True, False] * (p // 2))]

    return [np.array(combo) for combo in filtered_combinations]

def alg_null_coef(data, model_type, coef_to_use):
    return Alg(model_type = model_type, coef_to_use = coef_to_use).fit(data)
p = 6
null_combos = generate_boolean_combinations_except_all(p)
alg_null = [partial(alg_null_coef, coef_to_use = combo) for combo in null_combos]

def alg_null_star(data, coef):
    return Alg(model_type='linear', coef = coef)

def alg_alt(data, model_type):
    return Alg(model_type = model_type).fit(data)

def compute_sim(index, n, snr, p, rho, data_splits, method_args, alpha, alg_alt, alg_null):
    x = linear_model_sample(n, snr = snr, rho = rho, p=p)
    mcsm = MCSM(data_splits).fit(x, alg_alt = alg_alt, alg_null = alg_null)

    ds = np.empty((len(method_args), len(alg_null)))
    for i, (method, args) in enumerate(method_args.items()):
            ds[i, :] = mcsm.get_decision(args, alpha)

    mstat = mcsm.get_estimates()
    return ds, mstat

def recompute_sim(index, mstats, n, snr, p, rho, data_splits, method_args, alpha):
    mcsm = MCSM(data_splits, mstats = mstats[index])    
    ds = np.empty((len(method_args), len(alg_null)))
    for i, (method, args) in enumerate(method_args.items()):
        ds[i, :] = mcsm.get_decision(args, alpha)
    return ds
def parse_bool_array(s):
    s_clean = s.strip()[1:-1]  # remove the '[' and ']'
    tokens = s_clean.split()   # split on whitespace
    bool_array = np.array([token == 'True' for token in tokens])
    return bool_array

if __name__ == '__main__':

    env = Env()

    method_args_from_name = {
         r'Uniform $\text{UI-}\omega_{or}$': {
            'standardization': 'ui',
            'cons': True,
            'decision_type': 'exp'
        },
        'Selective SUI':{
            'standardization': 'clt',
            'cons': False,
            'decision_type': 'normal'
        }
    }

    base_filename = os.path.basename(__file__)[:-3]
    data_folder = os.path.join(env.data_dir, base_filename)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    alpha = 0.1
    n_trials = 300
    snrs = np.linspace(0.25, 4, 30)
    #rhos = [.35]
    rho = .35
    model_types = ['linear', 'rf']
    n = 200
    
    methods = list(method_args_from_name)
    splits = ['Holdout', '10-Fold CV']
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', ':', '-.']

    linear_combinations = [
        ('LRT', 'Single'),
        (r'Uniform $\text{UI-}\omega_{or}$', 'Holdout'), 
        ('Selective SUI', 'Holdout'),
                           ('Selective SUI', '10-Fold CV')]
    
    rf_combinations = [
        ('Selective SUI', 'Holdout'),
                           ('Selective SUI', '10-Fold CV')]
    
    method_combinations = [linear_combinations, rf_combinations]

    method_names = ['$\phi_{\mathrm{UI-}\omega_{or}}$', '$\phi^{\mathrm{Holdout}}_{\mathrm{CLT}}$', '$\phi^{\mathrm{CV}}_{\mathrm{CLT}}$']

    bool_combs = generate_boolean_combinations_except_all(p)
    correct_idx = np.asarray([i for i,t in enumerate(bool_combs) if t[0] and t[2] and t[4]])

    from sklearn.model_selection import KFold
    with Pool(env.num_cores) as po:
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        for m, model_type in enumerate(model_types):
            print(model_type)
            ralg_null = [partial(_, model_type = model_type) for _ in alg_null]
            ralg_alt = partial(alg_alt, model_type = model_type)
            data_splits_from_name = {
                'Holdout': [(np.array(range(n // 2)), np.array(range(n // 2, n)))],
                '10-Fold CV': list(KFold(n_splits = 10).split(range(n)))
            }
            cnt = m
            for j, (data_split_name, data_splits) in enumerate(data_splits_from_name.items()):
                    print(data_split_name)
                    power = np.empty((len(method_args_from_name), len(snrs)))
                    coverage = np.empty((len(method_args_from_name), len(snrs)))
                    size = np.empty((len(method_args_from_name), len(snrs)))
                    method_args = method_args_from_name
                    for k, snr in enumerate(snrs):
                        for method in method_args_from_name.keys():
                            if model_type == 'linear':
                                if method_args_from_name[method]['standardization'] == 'ui':
                                    cons = method_args_from_name[method]['cons']
                                    print(cons)
                                    if not cons:
                                        omega_null_alg = partial(alg_null_coef, model_type = model_type, coef_to_use=np.asarray([True,False]*(p//2)))
                                    else:
                                        omega_null_alg = partial(alg_null_star, coef = np.asarray([1, 0]* (p // 2)))
                                    if model_type == 'linear':
                                        file = os.path.join(data_folder, f'scc_estimates_cons={cons}_n={len(data_splits[0][0])}_snr={snr}_rho={rho}_p={p}.npy')
                                    else:
                                        file = os.path.join(data_folder, f'scc_estimates_cons={cons}_n={len(data_splits[0][0])}_model_type={model_type}_snr={snr}_rho={rho}_p={p}.npy')
                                    omega = estimate_scc(dgp = partial(linear_model_sample, snr = snr, rho = rho, p = p), 
                                                alg_null = omega_null_alg,
                                                alg_alt=partial(alg_alt, model_type = model_type),
                                                alpha = alpha,
                                                data_split = data_splits[0],
                                                B = 1000,
                                                pool = po,
                                                n = len(data_splits[0][0]),
                                                np_file = file)
                                    method_args_from_name[method]['omega'] = omega
                                    print(omega)
                            else:
                                method_args_from_name[method]['omega'] = 1
                        if model_type == 'linear':
                            np_file = os.path.join(os.path.join(env.data_dir, 'feature_selection_linear_sims'), f'stats_n={n}_split={data_split_name}_snr={snr:.3f}.npy')
                        else:
                            np_file = os.path.join(data_folder, f'stats_n={n}_model={model_type}_split={data_split_name}_snr={snr:.3f}.npy')
                        if os.path.exists(np_file):
                            mstat = np.load(np_file)
                            func = partial(recompute_sim, n=n, mstats = mstat, snr = snr, p = p, rho = rho, data_splits = data_splits, method_args = method_args, alpha = alpha)
                            ds = po.map(func, range(n_trials)) 
                        else:
                            func = partial(compute_sim, n=n, snr = snr, p = p, rho = rho,  data_splits = data_splits, method_args = method_args, alpha = alpha, alg_null = ralg_null, alg_alt = ralg_alt)
                            out = po.map(func, range(n_trials))
                            ds = [d[0] for d in out]
                            mstats = [d[1] for d in out]
                            mstat = np.asarray(mstats)
                            np.save(np_file, mstat)

                        ds = np.asarray(ds)
                        size[:, k] = 1-np.mean(ds, axis=(0,2))
                        power[:, k] = np.mean(ds[:, :, -1], axis=0)
                        coverage[:, k] = np.mean(np.all(ds[:,:,correct_idx] == 0, axis=2), axis=0)
                        print(coverage[:, k])
                    for i,method  in enumerate(method_args):
                        if (method, data_split_name) in method_combinations[m]:
                            print((method, data_split_name))
                            axs[0,m].errorbar(snrs, power[i,:],
                                                    yerr=1.96*np.sqrt(power[i,:]*(1-power[i,:]) / n_trials),
                                                    fmt='o',              
                                                    capsize=2,             
                                                    elinewidth=1,         
                                                    linewidth=1.2,
                                                    markeredgewidth=1,    
                                                    markersize=2,
                                                    color = plt.cm.tab10(cnt),
                                                    linestyle=linestyles[cnt]
                                                )
                            axs[1,m].errorbar(snrs, 62*size[i,:],
                                                    yerr=62*(1.96*np.sqrt(size[i,:]*(1-size[i,:]) / n_trials)),
                                                    fmt='o',          
                                                    capsize=2,           
                                                    elinewidth=1,     
                                                    linewidth=1.2,
                                                    markeredgewidth=1,   
                                                    markersize=2,
                                                    color = plt.cm.tab10(cnt),
                                                    linestyle=linestyles[cnt]
                                                )
                            axs[2,m].errorbar(snrs, coverage[i,:],
                                                    yerr=1.96*np.sqrt(coverage[i,:]*(1-coverage[i,:]) / n_trials),
                                                    fmt='o',                
                                                    capsize=2,             
                                                    elinewidth=1,        
                                                    linewidth=1.2,
                                                    markeredgewidth=1,  
                                                    markersize=2,
                                                    color = plt.cm.tab10(cnt),
                                                    linestyle=linestyles[cnt]
                                                )
                            cnt += 1

        print('Saving Plot')
        fsz = 22
        axs[0,0].set_title('Linear Models', fontsize=fsz)
        axs[0,1].set_title('Random Forests', fontsize=fsz)
        fig.supxlabel(r'$\nu$', fontsize = fsz)
        axs[0,0].set_ylabel(r'$P(r^* \notin \hat{\Theta})$', fontsize=fsz)
        axs[1,0].set_ylabel(r'$|\hat{\Theta}|$', fontsize = fsz)
        axs[2,0].set_ylabel(r'$P(\Theta \subseteq \hat{\Theta})$', fontsize = fsz)

        for m in range(len(model_types)):
            axs[0,m].set_ylim(bottom = 0, top = 0.35)
            axs[0,m].axhline(y = alpha, linestyle = 'dotted', color = 'red')
            axs[1,m].set_ylim(bottom = 0, top = 50)
            axs[2,m].set_ylim(bottom = 0.35, top = 1.05)

        axs[0,1].tick_params(axis='y', which='both', left=False, right=False)
        axs[0,1].set_yticklabels([])
        axs[1,1].tick_params(axis='y', which='both', left=False, right=False)
        axs[1,1].set_yticklabels([])
        axs[2,1].tick_params(axis='y', which='both', left=False, right=False)
        axs[2,1].set_yticklabels([])

        axs[0,0].tick_params(axis='x', which = 'both')
        axs[0,0].set_xticklabels([])
        axs[0,1].tick_params(axis='x', which = 'both')
        axs[0,1].set_xticklabels([])
        axs[1,0].tick_params(axis='x', which = 'both')
        axs[1,0].set_xticklabels([])
        axs[1,1].tick_params(axis='x', which = 'both')
        axs[1,1].set_xticklabels([])

        axs[0,0].tick_params(axis='both', labelsize=20)
        axs[1,0].tick_params(axis='both', labelsize=20)
        axs[2,0].tick_params(axis='both', labelsize=20)
        axs[2,1].tick_params(axis='both', labelsize=20)

        method_handles = [Line2D([0], [0], color=colors[i], linestyle = linestyles[i], lw=2) for i in range(3)]
        method_legend = axs[0,0].legend(method_handles, method_names, title="Method",
                        loc='upper left', title_fontsize = fsz-2, fontsize = fsz-4)
        axs[0,0].add_artist(method_legend)

        plt.savefig(os.path.join(env.plot_dir, f'{base_filename}_rho={rho}_p={p}_n={n}.png'))