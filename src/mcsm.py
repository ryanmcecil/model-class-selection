########################################
# Functions for conducting model class selection given data splits
########################################
from sklearn.model_selection import KFold
import numpy as np
import scipy.stats as sps
import unittest
from scipy.stats import norm
from typing import List, Tuple, Dict

np.seterr(over="ignore")

class MCSM():
    def __init__(self,
                 data_splits: List[Tuple[np.array]],
                 mstats = None):

        self.data_splits = data_splits
        self.n_folds = len(data_splits)
        self.test_fold_sizes = [len(fold[1]) for fold in data_splits]
        self.n_heldout = sum(self.test_fold_sizes)

        if mstats is not None:
            self.stats, self.stats_cons, self.sds = mstats[0], mstats[1], mstats[2]
        else:
            self.stats, self.stats_cons, self.sds = None, None, None

    def get_estimates(self):
        return np.stack((self.stats, self.stats_cons, self.sds), axis=0)

    def get_decision(self, 
                     method_args: Dict,
                     alpha: float = .05,
                     return_stat: bool = False):
        stats = self._stats_tf(**method_args)
        if not return_stat:
            return self._stats_to_decision(stats, alpha = alpha, **method_args)
        else:
            return self._stats_to_decision(stats, alpha = alpha, **method_args), stats
    
    def _stats_to_decision( self, 
                            stats,
                            alpha: int = .05,
                            decision_type: str = 'exp',
                            chi_df: int = 5,
                            **kwargs):
        if decision_type == 'exp':
            return stats > 1/alpha
        elif decision_type == 'normal':
            return stats > sps.norm.ppf(1-alpha)
        elif decision_type == 'chi':
            return stats > sps.chi2.ppf(1-alpha, df = chi_df)
        else:
            raise Exception(f'Unknown decision threshold type {decision_type}')

    def _stats_tf(self,
                 standardization: str = None,
                 cons: bool = True,
                 omega: int = 1,
                 **kwargs):
        if cons:
            stats = self.stats_cons
        else:
            stats = self.stats

        if standardization == 'clt':
            stats = np.sum(stats, axis=1)
            sds = np.mean(self.sds, axis=1)
            stats = stats / np.maximum((np.sqrt(self.n_heldout)*sds), 1e-10)
        elif standardization == 'ui':
            stats = np.sum(np.exp(omega*stats - np.log(self.n_folds)), axis=1)
        else:
            raise Exception(f'Unknown standardization type {standardization}')
        
        return stats
    
    def fit(self,
            data,
            alg_null,
            alg_alt,
            return_errors = False):
        
        max_fold_size = max(self.test_fold_sizes)
        m01 = np.full((self.n_folds, max_fold_size), np.nan)
        for i, (train_index, test_index) in enumerate(self.data_splits):
            loss_vals = alg_alt(data[train_index]).score_samples(data[test_index], n = len(data))
            m01[i, :len(loss_vals)] = loss_vals

        if not isinstance(alg_null, list):
            alg_null = [alg_null]

        m00 = np.full((len(alg_null), self.n_folds, max_fold_size), np.nan)
        m00_cons = np.full((len(alg_null), self.n_folds, max_fold_size), np.nan)
        for j, alg in enumerate(alg_null):
            if self.n_folds > 1:
                cons_null_model = alg(data)
            else:
                cons_null_model = alg(data[self.data_splits[0][1]])
            for i, (train_index, test_index) in enumerate(self.data_splits):
                null_model = alg(data[train_index])
                loss_vals = null_model.score_samples(data[test_index], n = len(data))
                m00[j, i, :len(loss_vals)] = loss_vals
                loss_vals = cons_null_model.score_samples(data[test_index], n = len(data))
                m00_cons[j, i, :len(loss_vals)] = loss_vals

        mdiff = m00 - m01
        mdiff_cons = m00_cons - m01

        self.stats = np.nansum(mdiff, axis = 2)
        self.stats_cons = np.nansum(mdiff_cons, axis=2)
        self.sds = np.nanstd(mdiff, axis=2)

        if not return_errors:
            return self
        else:
            return self, m00, m01

class TestMCSMethods(unittest.TestCase):

    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)

        class Normal():
            def __init__(self, mean = None, sd = None):
                self.mean = mean
                self.sd = sd

            def fit(self, X):
                self.mean = np.mean(X)
                return self
            
            def log_lik(self, X):
                return norm.logpdf(X, loc=self.mean, scale=self.sd)
            
            def score_samples(self, X, n = None):
                return -norm(self.mean, self.sd).logpdf(X)

        def alg_null(data):
            return Normal(sd = .5).fit(data)
        self.alg_null = alg_null

        def alg_alt(data):
            return Normal(sd = 1).fit(data)
        self.alg_alt = alg_alt

    def test_ui(self):
        n = 200
        x = norm().rvs(size = n)

        # Test standard univeral inference
        data_splits = [(np.array(range(n // 2)), np.array(range(n // 2, n)))]
        args = {
            'standardization': 'ui',
            'cons': True,
            'decision_type': 'exp'
        }
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]

        d0 = x[100:]
        d1 = x[0:100]

        fit0 = self.alg_null(d0)
        fit1 = self.alg_alt(d1)
        ll0 = fit0.log_lik(d0)
        ll1 = fit1.log_lik(d0)
        tmp_stat = sum(ll1) - sum(ll0)
        self.assertAlmostEqual(stat, tmp_stat, places = 3)

        # Test generalized universal inference
        args['omega'] = 2
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]
        tmp_stat = 2*(sum(ll1) - sum(ll0))
        self.assertAlmostEqual(stat, tmp_stat, places = 3)

        # Test less conservative version of generalized universal inference
        args['cons'] = False
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]
        fit0 = self.alg_null(d1)
        ll0 = fit0.log_lik(d0)
        tmp_stat = 2*(sum(ll1) - sum(ll0))
        self.assertAlmostEqual(stat, tmp_stat, places = 3)

    def test_holdout(self):
        n = 200
        x = norm().rvs(size = 200)
        data_splits = [(np.array(range(n // 2)), np.array(range(n // 2, n)))]
        args = {
            'standardization': 'clt',
            'data_split': 'holdout',
            'cons': True,
        }
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]

        d0 = x[100:]
        d1 = x[0:100]
        fit0 = self.alg_null(d1)
        ll0 = fit0.log_lik(d0)
        fit1 = self.alg_alt(d1)
        ll1 = fit1.log_lik(d0)
        sd = np.std(ll1 - ll0)
        fit0 = self.alg_null(d0)
        fit1 = self.alg_alt(d1)
        ll0 = fit0.log_lik(d0)
        ll1 = fit1.log_lik(d0)
        tmp_stat = sum(ll1 - ll0)/(np.sqrt(len(d0))*sd)
                                   
        self.assertAlmostEqual(stat, tmp_stat, places = 4)

        # Test less conservative version

        args['cons'] = False
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]
        fit0 = self.alg_null(d1)
        ll0 = fit0.log_lik(d0)
        tmp_stat = sum(ll1 - ll0)/max((np.sqrt(len(d0))*np.std(ll1 - ll0)), 1e-6)
        self.assertAlmostEqual(stat, tmp_stat, places = 4)


    def test_cv(self):
        n = 200
        x = norm().rvs(size = 200)

        # Test less conservative version of cross-validation
        data_splits = list(KFold(n_splits = 10).split(x))
        args = {
            'standardization': 'clt',
            'cons': False,
            'decision_type': 'normal'
        }
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]
        means = list()
        sds = list()
        for i, (train_index, test_index) in enumerate(data_splits):
            alt_model = self.alg_alt(x[train_index])
            null_model = self.alg_null(x[train_index])
            l01 = alt_model.score_samples(x[test_index])
            l00 = null_model.score_samples(x[test_index])
            diffs = l00 - l01
            means.append(np.mean(diffs))
            sds.append(np.std(diffs))
        
        grand_mean = np.mean(np.array(means))
        grand_sd = np.mean(np.array(sds))
        tmp_stat = np.sqrt(200)*grand_mean/(grand_sd)
        self.assertAlmostEqual(stat, tmp_stat, places = 3)

        # Test conservative version of cross-validation
        args['cons'] = True
        stat = MCSM(data_splits).fit(x, alg_alt = self.alg_alt, alg_null = self.alg_null)._stats_tf(**args)[0]
        data_splits = KFold(n_splits = 10).split(x)
        null_model = self.alg_null(x)
        means = list()
        sds = list()
        for i, (train_index, test_index) in enumerate(data_splits):
            alt_model = self.alg_alt(x[train_index])
            l01 = alt_model.score_samples(x[test_index])
            l00 = null_model.score_samples(x[test_index])
            diffs = l00 - l01
            means.append(np.mean(diffs))
            sds.append(np.std(diffs))
        
        grand_mean = np.mean(np.array(means))
        tmp_stat = np.sqrt(200)*grand_mean/(grand_sd)
        self.assertAlmostEqual(stat, tmp_stat, places = 3)

if __name__ == '__main__':
    unittest.main()