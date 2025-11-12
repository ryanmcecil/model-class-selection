########################################
# Conservative test example in appendix
########################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from functools import partial
from mcsm import MCSM
from matplotlib.lines import Line2D
import os
from config import Env

class LinReg():
    def __init__(self, coef_to_use = None):
        self.model = LinearRegression(fit_intercept=False)
        self.coef_to_use = coef_to_use

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
        if self.coef_to_use is None:
            yhat = self.model.predict(X)
        else:
            yhat = self.model.predict(X[:, self.coef_to_use])
        residuals = y - yhat
        return residuals**2

def normal_sample(n):
    rng = np.random.default_rng()
    X = rng.multivariate_normal(mean=np.zeros(3), cov = np.eye(3), size=n)
    eps = rng.normal(0, 1, size=n)
    y = X @ np.asarray([1,1,1]) + eps
    return np.concatenate([X, y.reshape(-1, 1)], axis=1)

def alg_null(data):
    return LinReg(coef_to_use=[True, True, False]).fit(data)

def alg_alt(data):
    return LinReg(coef_to_use=[True, False, True]).fit(data)

def compute_sim(index, n, data_splits, method_args, alpha):
    x = normal_sample(n)
    mcsm = MCSM(data_splits).fit(x, alg_alt = alg_alt, alg_null = alg_null)

    ds = np.empty((len(method_args)))
    for i, (method, args) in enumerate(method_args.items()):
            ds[i] = mcsm.get_decision(args, alpha)[0]
    return ds

if __name__ == '__main__':

    env = Env()

    method_args_from_name = { 
        'SUI':{
            'standardization': 'clt',
            'cons': False,
            'decision_type': 'normal'
        },
        'SUIInf':{
           'standardization': 'clt',
            'cons': True,
            'decision_type': 'normal' 
        }
    }

    base_filename = os.path.basename(__file__)[:-3]

    alpha = 0.05
    n_trials = 1000
    ns = np.astype(np.floor(10**np.linspace(2, 5, num=10)), np.int32)

    def get_data_split(n, i):
        if i == 0:
            ntr = n // 2
        elif i == 1:
            ntr = int(n ** (1. / 2))
        else:
            ntr = int(n ** (1. / 3))
        return [(np.array(range(ntr)), np.array(range(ntr, n)))]


    lstls = ['-', '--', ':', '-.']

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    with Pool(env.num_cores) as p:
        for split in range(0,3):
            print(split)
            power = np.empty((len(ns), len(method_args_from_name)))
            for i, n in enumerate(ns):
                print(n)
                data_splits = get_data_split(n, split)
                func = partial(compute_sim, n=n, data_splits = data_splits, method_args = method_args_from_name, alpha = alpha)
                out = p.map(func, range(n_trials))
                out = np.asarray(out)
                for j in range(len(method_args_from_name)):
                    power[i, j] = np.mean(out[:,j])
            for j in range(len(method_args_from_name)):
                axs[j].errorbar(ns, power[:,j],
                                                yerr=1.96*np.sqrt(power[:,j]*(1-power[:,j]) / n_trials),
                                                fmt='o',            
                                                capsize=2,         
                                                elinewidth=1,        
                                                linewidth=1.2,
                                                markeredgewidth=1, 
                                                markersize=2,
                                                color = plt.cm.tab10(split),
                                                linestyle=lstls[split]
                                            )
            
    print('Saving Plot')
    fig.supxlabel('Sample Size', fontsize=18)
    fig.supylabel('Estimated Type I Erorr', fontsize=18)
    axs[0].set_title(r'$\phi^{\mathrm{Holdout}}_{\mathrm{CLT}}$', fontsize = 20)
    axs[1].set_title(r'$\phi^{inf}_{\mathrm{CLT}}$', fontsize = 20)
    for j in range(len(axs)):
        axs[j].set_xscale('log')
        axs[j].tick_params(axis='x', which='minor', bottom=False)
        axs[j].set_xlim(80, 1e5 + 1000)
        axs[j].hlines(.05, xmin = 1e2, xmax = 1e5, color = 'red', linestyles='-.')
        axs[j].set_ylim(0,.55)
        ticks = [1e2, 1e3, 1e4, 1e5]
        axs[j].set_xticks(ticks, [r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'], fontsize = 14)
    legend_elements = [
    Line2D([0], [0], color=plt.cm.tab10(0), lw=2, label=r'$n$', linestyle=lstls[0]),
    Line2D([0], [0], color=plt.cm.tab10(1), lw=2, label=r'$\sqrt{n}$', linestyle=lstls[1]),
    Line2D([0], [0], color=plt.cm.tab10(2), lw=2, label=r'$\sqrt[3]{n}$', linestyle=lstls[2]),
]
    
    axs[1].legend(handles=legend_elements, loc='upper right', title = r'Training Set Size ($n_{tr}$)')
    plt.savefig(os.path.join(env.plot_dir, f'{base_filename}.png'))
