########################################
# Nonlinear regression sim comparing linear models to random forests
########################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from functools import partial
from mcsm import MCSM
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from scipy.special import expit
import os
from estimate_scc import estimate_scc
from config import Env

class Alg():
    def __init__(self, model_type: str = 'lm', n_estimators: int = 100, coef = None):
        self.coef = coef
        if model_type == 'lm':
            self.model = LinearRegression()
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=n_estimators)
        else:
            raise Exception

    def fit(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        self.model.fit(X, y)
        return self

    def score_samples(self, data, n = None):
        X = data[:, :-1]
        y = data[:, -1]
        if self.coef is not None:
            yhat = X @ self.coef
        else:
            yhat = self.model.predict(X)
        return np.square(yhat - y)

def normal_sample(n, gamma):
    rng = np.random.default_rng()
    X = rng.uniform(low = 0, high = 1, size = (n, 5))
    eps = rng.normal(0, 10, size=n)
    y = gamma*(10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*np.square(X[:,2] - 0.05)) + 10*X[:,3] + 5*X[:,4] + eps
    return np.concatenate([X, y.reshape(-1, 1)], axis=1)

def alg_null(data):
    return Alg(model_type = 'lm').fit(data)

def alg_nullstar(data, coef):
    return Alg(model_type = 'lm', coef = coef)

def alg_alt(data, n_estimators):
    return Alg(model_type = 'rf', n_estimators = n_estimators).fit(data)

def compute_sim(index, n, gamma, n_estimators, data_splits, method_args, alpha):
    x = normal_sample(n, gamma)
    mcsm = MCSM(data_splits).fit(x, alg_alt = partial(alg_alt, n_estimators = n_estimators), alg_null = alg_null)

    ds = np.empty((len(method_args)))
    for i, (method, args) in enumerate(method_args.items()):
            ds[i] = mcsm.get_decision(args, alpha)[0]

    mstat = mcsm.get_estimates()
    return ds, mstat

def recompute_sim(index, mstats, n, gamma, data_splits, method_args, alpha):
    mcsm = MCSM(data_splits, mstats = mstats[index])    
    ds = np.empty((len(method_args)))
    for i, (method, args) in enumerate(method_args.items()):
        ds[i] = mcsm.get_decision(args, alpha)[0]
    return ds

if __name__ == '__main__':

    env = Env()

    method_args_from_name = { 
         r'Unif $\text{UI-}\omega_{or}$': {
            'standardization': 'ui',
            'cons': True,
            'decision_type': 'exp'
        },
        'Sel SUI':{
            'standardization': 'clt',
            'cons': False,
            'decision_type': 'normal'
        }
    }
    base_filename = os.path.basename(__file__)[:-3]
    data_folder = os.path.join(env.data_dir, base_filename)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    alpha = 0.05
    n_trials = 1000
    gammas = np.linspace(0, 6, 30)
    ns = [200, 500, 1000]
    n_estimator = 30

    methods = list(method_args_from_name)
    splits = ['Holdout', '10-Fold CV']
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', ':', '-.']
    ignore_combinations = [(r'Unif $\text{UI-}\omega_{or}$', '2-Fold CV'), 
                           (r'Unif $\text{UI-}\omega_{or}$', '10-Fold CV'),
                           (r'Sel $\text{UI-}\omega_{or}$', 'Holdout')]
    method_names = ['$\phi_{\mathrm{UI-}\omega_{or}}$', '$\phi^{\mathrm{Holdout}}_{\mathrm{CLT}}$', '$\phi^{\mathrm{CV}}_{\mathrm{CLT}}$']

    from sklearn.model_selection import KFold
    count = 0
    with Pool(env.num_cores) as p:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(12, 5)
        ax_by_dim = {200: ax1, 500: ax2, 1000: ax3}
        for n in ns:
            data_splits_from_name = {   
                'Holdout': [(np.array(range(n // 2)), np.array(range(n // 2, n)))],
                '10-Fold CV': list(KFold(n_splits = 10).split(range(n)))
            }
            cnt = 0
            for j, (data_split_name, data_splits) in enumerate(data_splits_from_name.items()):
                    
                    power = np.empty((len(method_args_from_name), len(gammas)))
                    
                    for method in method_args_from_name.keys():
                        if method_args_from_name[method]['standardization'] == 'ui':
                            cons = method_args_from_name[method]['cons']
                            if not cons:
                                omega_null_alg = alg_null
                            else:
                                omega_null_alg = partial(alg_nullstar, coef = np.asarray([0,0,0,10,5]))
                            omega = estimate_scc(dgp = partial(normal_sample, gamma = 0), 
                                         alg_null = omega_null_alg,
                                         alg_alt=partial(alg_alt, n_estimators = n_estimator),
                                         alpha = alpha,
                                         data_split = data_splits[0],
                                         B = 10000,
                                         pool = p,
                                         n = len(data_splits[0][0]),
                                         np_file = os.path.join(data_folder, f'scc_estimates_cons={cons}_n={len(data_splits[0][0])}_n_estimator={n_estimator}.npy'))
                            method_args_from_name[method]['omega'] = omega
                            print(cons)
                            print(omega)

                    for k, gamma in enumerate(gammas):
                        np_file = os.path.join(data_folder, f'stats_n={n}_split={data_split_name}_ne={n_estimator}_gamma={gamma:.3f}.npy')
                        if os.path.exists(np_file):
                            mstat = np.load(np_file)
                            func = partial(recompute_sim, n=n, mstats = mstat, gamma = gamma, data_splits = data_splits, method_args = method_args_from_name, alpha = alpha)
                            ds = p.map(func, range(n_trials)) 
                        else:
                            func = partial(compute_sim, n=n, gamma = gamma, n_estimators = n_estimator, data_splits = data_splits, method_args = method_args_from_name, alpha = alpha)
                            out = p.map(func, range(n_trials))
                            ds = [d[0] for d in out]
                            mstats = [d[1] for d in out]
                            mstat = np.asarray(mstats)
                            np.save(np_file, mstat)
                        power[:, k] = np.mean(np.asarray(ds), axis=0)
                    for i, (method, method_args) in enumerate(method_args_from_name.items()):
                        if (method, data_split_name) not in ignore_combinations:
                            print(method)
                            print(data_split_name)
                            print(i)
                            ax_by_dim[n].errorbar(gammas, power[i,:],
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
                            cnt += 1

                            print('Saving Plot')
                            ax1.set_title(r'$n=200$', fontsize = 14)
                            ax2.set_title(r'$n=500$', fontsize = 14)
                            ax3.set_title(r'$n=1000$', fontsize = 14)
                            ax1.set_ylim((-0.05, 1.05))
                            ax2.set_ylim((-0.05, 1.05))
                            ax3.set_ylim((-0.05, 1.05))
                            fig.supxlabel(r"$\gamma$", fontsize = 14)
                            fig.supylabel('Probability of Rejection', fontsize = 14)
                            plt.subplots_adjust(bottom=0.15, left=.08)

                            method_handles = [Line2D([0], [0], color=colors[i], linestyle = linestyles[i], lw=2) for i in range(3)]
                            method_legend = ax1.legend(method_handles, method_names, title="Method",
                                            loc='upper left')
                            ax1.add_artist(method_legend)
                            plt.savefig(os.path.join(env.plot_dir, f'{base_filename}_{count}.png'))
                            count += 1
    plt.savefig(os.path.join(env.plot_dir, f'{base_filename}.png'))