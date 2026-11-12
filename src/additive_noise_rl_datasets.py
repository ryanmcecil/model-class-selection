########################################
# Experiment adding noise to datasets and seeing impact on MCS results
########################################
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method
from functools import partial
from mcsm import MCSM
from matplotlib.lines import Line2D
import numpy as np
import os
from config import Env
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Model():
    def __init__(self, reg, model):
        if reg:
            if model == 'linear':
                self.model = make_pipeline(StandardScaler(), LinearRegression())
            else:
                self.model = RandomForestRegressor(criterion='squared_error')
        else:
            if model == 'linear':
                self.model = make_pipeline(StandardScaler(), LogisticRegression(penalty = None, max_iter=10000))
            else:
                self.model = RandomForestClassifier(criterion='log_loss', n_estimators=1000)
        self.reg = reg

    def fit(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        self.model.fit(X, y)
        return self
    
    def score_samples(self, data, n):
        X = data[:, :-1]
        y = data[:, -1]
        if self.reg:
            yhat = self.model.predict(X)
            return (yhat - y)**2
        else:
            eps = 1e-15
            yhat = self.model.predict_proba(X)[:,1]
            yhat = np.clip(yhat, eps, 1-eps)
            return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))

def alg(data, reg, model):
    return Model(model = model, reg = reg).fit(data)
alg_null = partial(alg, model = 'linear')
alg_alt = partial(alg, model = 'rf')

def add_noise(data, rho, reg):
    data = np.copy(data)
    y = data[:, -1] 
    if reg:
        if rho != 0:
            sd = np.std(y)
            eps = np.random.normal(loc = 0, scale = rho*4*sd, size=len(y))
            data[:, -1] = y + eps
    else:
        samples = np.random.rand(len(data)) < rho
        y = np.where(samples, np.abs(y - 1), y)
        data[:, -1] = y
    np.random.shuffle(data)
    return data

def compute_sim(index, data, rho, reg, data_splits, method_args, alpha, alg_alt, alg_null):
    x = add_noise(data, rho, reg)
    mcsm = MCSM(data_splits).fit(x, alg_alt = partial(alg_alt, reg=reg), alg_null = partial(alg_null, reg = reg))

    ds = mcsm.get_decision(method_args, alpha)
    mstat = mcsm.get_estimates()
    return ds, mstat

def recompute_sim(index, mstats, data_splits, method_args, alpha):
    mcsm = MCSM(data_splits, mstats = mstats[index])    
    ds = np.empty((1,1))
    ds = mcsm.get_decision(method_args, alpha)
    return ds

if __name__ == '__main__':

    set_start_method('spawn') # Fix rng linux duplicate problem

    env = Env()

    method_args = {
            'standardization': 'clt',
            'cons': False,
            'decision_type': 'normal'
        }

    base_filename = os.path.basename(__file__)[:-3]
    data_folder = os.path.join(env.data_dir, base_filename)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    alpha = 0.1
    n_trials = 1000
    noise_probs = np.linspace(0, 0.25, 30)
    datasets = ['monks', 'boston', 'concrete', 'servo']
    regs = [False, True, True, True]
    linestyles = ['-', '--', ':', '-.']

    from sklearn.model_selection import KFold
    with Pool(env.num_cores) as po:

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(10, 4)
        split_to_ax = {'Holdout': ax1, '10-Fold CV': ax2}
        for d, dataset in enumerate(datasets):

            print(dataset)
            print(f'Regression: {regs[d]}')
            X = np.loadtxt(os.path.join(env.datamat_dir, f'{dataset}_X_mat.csv'), delimiter=',', skiprows=1)
            y = np.loadtxt(os.path.join(env.datamat_dir, f'{dataset}_Y_mat.csv'), delimiter=',', skiprows=1)
            data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

            n = len(data)
            data_splits_from_name = {
                'Holdout': [(np.array(range(n // 2)), np.array(range(n // 2, n)))],
                '10-Fold CV': list(KFold(n_splits = 10, shuffle=True).split(range(n)))
            }

            for j, (data_split_name, data_splits) in enumerate(data_splits_from_name.items()):
                    power = np.empty(len(noise_probs))
                    for k, rho in enumerate(noise_probs):
                        print(rho)
                        np_file = os.path.join(data_folder, f'dataset={dataset}_reg={regs[d]}_rho={rho:.4f}_split={data_split_name}.npy')
                        if os.path.exists(np_file):
                            mstat = np.load(np_file)
                            func = partial(recompute_sim, mstats = mstat, data_splits = data_splits, method_args = method_args, alpha = alpha)
                            ds = po.map(func, range(n_trials)) 
                        else:
                            func = partial(compute_sim, data=data, rho = rho, reg = regs[d], alg_null = alg_null, alg_alt = alg_alt, data_splits = data_splits, method_args = method_args, alpha = alpha)
                            out = po.map(func, range(n_trials)) #[ trial, methods]
                            ds = [d[0] for d in out]
                            mstats = [d[1] for d in out]
                            mstat = np.asarray(mstats) # [observation, null_models, stat type]
                            np.save(np_file, mstat)

                        power[k] = np.mean(np.asarray(ds), axis=0)

                    split_to_ax[data_split_name].errorbar(noise_probs, power,
                                                yerr=1.96*np.sqrt(power*(1-power) / n_trials),
                                                fmt='o',                # Circle marker at each point
                                                capsize=2,              # Length of the little |-|
                                                elinewidth=1,         # Line width of CI bar\
                                                linewidth=1.2,
                                                markeredgewidth=1,    # Edge of marker
                                                markersize=2,
                                                color = plt.cm.tab10(d),
                                                linestyle=linestyles[d]
                                            )

    print('Saving Plot')
    fig.supxlabel(r'$\rho$', fontsize = 14)
    fig.supylabel('Probability of Rejection', fontsize = 14)

    ax1.set_title('Holdout')
    ax2.set_title('10-Fold CV')
    ax1.set_ylim((-0.05, 1.05))
    ax2.set_ylim((-0.05, 1.05))
    plt.subplots_adjust(bottom=0.15, left=.08)

    method_handles = [Line2D([0], [0], color=plt.cm.tab10(d), linestyle=linestyles[d], lw=2) for d, dataset in enumerate(datasets)]
    method_legend = ax2.legend(method_handles, datasets, title="Dataset",
                    loc='lower left')
    ax2.add_artist(method_legend)
    plt.savefig(os.path.join(env.plot_dir, f'{base_filename}.png'))
