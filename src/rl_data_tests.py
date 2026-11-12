########################################
# Real-world data experiment
########################################
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mcsm import MCSM
import numpy as np
import os
from config import Env
from pathlib import Path


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
        self.model_type = model

    def fit(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        if self.reg and self.model_type == 'linear' and X.shape[1] >= X.shape[0]:
            self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 6, 50)))
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

if __name__ == '__main__':

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
    datasets = ['adult', 'german', 'compas', 'telco', 'monks',
                'boston', 'auto', 'bike', 'abalone', 'concrete', 'cpm', 'csm', 'fb', 'wine', 'servo', 'meltingpoint']
    dataset_names = ['Adult Income',
                     'German Credit',
                     'COMPAS',
                     'Telco',
                     'Monks',
                     'Boston Housing',
                     'Auto',
                     'Bike Sharing',
                     'Abalone Age',
                     'Concrete Strength',
                     'CPU',
                     'CSM',
                     'Facebook Metrics',
                     'Wine Quality',
                     'Servo System',
                     'Melting Point'
                     ]
    regs = [False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True]

    splits = ['Holdout', '10-Fold CV']
    linestypes = {
        'Holdout': '-',
        '10-Fold CV': ':'
    }

    ncols = 4
    nrows = 4
    fig, axes = plt.subplots(ncols, nrows, figsize=(5*ncols, 6*nrows), squeeze=False)

    latex_table = r"""\begin{table}[p]
\centering
\begin{tabular}{l l c c c l}
\hline
Dataset & Task & n & Features & Statistic & Decision \\
\hline
"""

    np.random.seed(92526)
    from sklearn.model_selection import KFold
    for d, dataset in enumerate(datasets):

        X = np.loadtxt(os.path.join(env.datamat_dir, f'{dataset}_X_mat.csv'), delimiter=',', skiprows=1)
        y = np.loadtxt(os.path.join(env.datamat_dir, f'{dataset}_Y_mat.csv'), delimiter=',', skiprows=1)
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

        data_split = list(KFold(n_splits = 10, shuffle=True).split(range(len(data))))

        r, c = divmod(d, ncols)
        ax = axes[r][c]

        test_filebase = os.path.join(data_folder, f'dataset={dataset}_reg={regs[d]}')

        if os.path.exists(test_filebase + '_mstat.npy'):
            mstat = np.load(test_filebase + '_mstat.npy')
            m00 = np.load(test_filebase + '_m00.npy')
            m01 = np.load(test_filebase + '_m01.npy')
            mcsm = MCSM(data_split, mstats=mstat)
            ds, stats = mcsm.get_decision(method_args, alpha, return_stat=True)
        else:
            mcsm, m00, m01 = MCSM(data_split).fit(data, alg_alt = partial(alg_alt, reg=regs[d]), alg_null = partial(alg_null, reg = regs[d]), return_errors=True)
            m00 = np.nanmean(m00, axis=(0,2))
            m01 = np.nanmean(m01, axis=(1))
            ds, stats = mcsm.get_decision(method_args, alpha, return_stat=True)
            mstat = mcsm.get_estimates()
            np.save(test_filebase + '_mstat.npy', mstat)
            np.save(test_filebase + '_m00.npy', m00)
            np.save(test_filebase + '_m01.npy', m01)

        if ds[0]:
            decision = 'Reject'
        else:
            decision = 'Fail to Reject'
        if regs[d]: task = 'Regression' 
        else: task = 'Classification'
        features = X.shape[1]
        print(f"{dataset}: {decision}. n: {len(data)}. Stat: {stats[0]}. Task: {task}. Number of Features: {features}")
        latex_table += f"{dataset} & {task} & {len(data)} & {features} & {stats[0]:.2f} & {decision} \\\\\n"

        ax.boxplot([m00, m01], labels = ['LM', 'RF'])
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_title(f'{dataset_names[d]}', fontsize=22)

        ax.set
    latex_table += r"""\hline
\end{tabular}
\caption{Experiment Results}
\label{fig:rl-cv-results}
\end{table}
"""
    Path(os.path.join(env.table_dir, f'{base_filename}.tex')).write_text(latex_table)
    plt.savefig(os.path.join(env.plot_dir, f'{base_filename}.png'))
