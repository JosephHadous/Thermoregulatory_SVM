#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:14:48 2025

@author: Joseph Hadous

"""
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.stats import false_discovery_control, ttest_1samp

# %% Data Processing


def getWarm(d):
    return pd.concat([d[72:112], d[336:376]])


def getCool(d):
    return pd.concat([d[168:208], d[456:496]])


verbose = 10
min_acc_score = 0.65  # any model with a score less than this won't be passed to a null-check
min_boot_acc = (
    0.7  # any model with a score less than this won't have it's weights saved
)


csv_dir = "/Users/hh1850/Dropbox/COLDFMRI_csv/"
csv_list = [
    pd.read_csv(csv_dir + d) for d in os.listdir(csv_dir) if ".csv" in d
]
X = np.concatenate(
    (
        [getWarm(d).corr() for d in csv_list],
        [getCool(d).corr() for d in csv_list],
    ),
    axis=0,
)
n_subs = len(csv_list)
n_regs = X[0].shape[0]
n_feats = int(n_regs * (n_regs - 1) / 2)
y = np.concatenate((np.zeros(n_subs), np.ones(n_subs)))

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True, random_state=42
)

X_train = np.zeros((len(x_train[:,]), n_feats))
X_test = np.zeros((len(x_test[:,]), n_feats))


def upperTri(ar):
    ui = np.triu_indices(ar.shape[0], k=1)
    return ar[ui]


for i in range(len(x_train[:,])):
    X_train[i][:][:] = np.arctanh(upperTri(x_train[i][:][:]))
for i in range(len(x_test[:,])):
    X_test[i][:][:] = np.arctanh(upperTri(x_test[i][:][:]))
# %% Hyper-parameter Search

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC())])
from itertools import product

params = product(
    [{"C": i} for i in 10.0 ** np.arange(-5, 3)],
    [{"tol": i} for i in 10.0 ** np.arange(-5, 3)],
    [{"loss": "hinge"}, {"loss": "squared_hinge"}],
)


def cv_score(
    estimator,
    X_train,
    y_train,
    cv,
) -> float:
    """
    Just a more convenient version of cross_val_score
    """
    acc = cross_val_score(clone(estimator), X_train, y_train, cv=cv, n_jobs=1)
    return np.mean(acc)


def null_model_check(estimator, X, y, cv, n_shuffles: int = 10, real_acc=None):
    if real_acc is None:
        real_acc = cv_score(estimator, X, y, cv)
    null_scores = []
    for _ in range(n_shuffles):
        y_shuffled = np.random.permutation(y)
        null_score = cv_score(clone(estimator), X, y_shuffled, cv)
        null_scores.append(null_score)
    null_mean = np.mean(null_scores)
    null_std = np.std(null_scores)
    passed = real_acc > null_mean + 2 * null_std
    return passed, real_acc, null_mean, null_std


def search_cv(pipe, params, X_, Y_, n_jobs=-1, cv=cv, verbose=0):

    estimators = []
    d = dict()
    for i in params:
        [d.update(j) for j in i]
        estimators.append(
            Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(**d))])
        )

    def _score(est, X_t, Y_t, cv):
        p, s, *_ = null_model_check(est, X_t, Y_t, cv)
        return est, p * s

    results = Parallel(n_jobs=n_jobs, verbose=verbose, return_as="generator")(
        delayed(_score)(est, X_, Y_, cv) for est in estimators
    )
    best_score = 0
    best_est = None
    for res in results:
        if res[1] > best_score:
            best_est, best_score = res

    return best_est, best_score


def boot(i, best, X_train, y_train, cv) -> dict:
    est = clone(best)
    est.set_params(svc__random_state=i + 1)
    est.fit(X_train, y_train)
    weights = est.get_params()["svc"].coef_[0]
    passed, score, _, _ = null_model_check(est, X_train, y_train, cv)
    if passed:
        return weights, score
    return None


coarse_grid, coarse_score = search_cv(
    pipe, params, X_train, y_train, n_jobs=-1, cv=cv, verbose=verbose
)
print(coarse_grid)
print(coarse_score)
best_c = coarse_grid.get_params()["svc__C"]
best_tol = coarse_grid.get_params()["svc__tol"]
best_loss = coarse_grid.get_params()["svc__loss"]
params = product(
    [
        {"C": i}
        for i in np.concat(
            [
                np.arange(best_c / 10 * 2.5, best_c, best_c / 4),
                [best_c],
                np.arange(best_c * 2.5, best_c * 10, best_c * 10 / 4),
            ]
        )
    ],
    [{"tol": i} for i in [best_tol / 10, best_tol, best_tol * 10]],
    [{"loss": best_loss}],
)


grid, score = search_cv(
    pipe, params, X_train, y_train, n_jobs=-1, cv=cv, verbose=verbose
)
print(grid)
print(score)
"""
    Worth double checking that this is the most optimal model by
    running another small search. Our study has a small sample      
    size so more refined searches don't imporve accuracy.
"""
# %% bootstrap / feature weight extraction


boot_res = Parallel(n_jobs=-1, verbose=verbose, return_as="generator")(
    delayed(boot)(i, grid, X_train, y_train, cv) for i in range(10_000)
)
weights = []
saved_scores = []
for res in boot_res:
    if res is None:
        continue
    w, s = res
    if s >= int(min_boot_acc * y_train.shape[0]) / y_train.shape[0]:
        weights.append(w)
        saved_scores.append(s)
# %% Identification of significant features

ws = np.mean(weights, 0)
_, ps = ttest_1samp(weights, 0)
p = false_discovery_control(ps) < 0.05

# Feature weights only have relative meaning so we should still
# consider the full distribution of weights for the top 99%
sigs = (abs(ws) > np.percentile(abs(ws), 99)) * ws * p

r, c = np.triu_indices(n_regs, 1)
w_2d = np.zeros((n_regs, n_regs))
w_2d[np.triu_indices(n_regs, 1)] = sigs

np.savez_compressed("all_weights", full_weight_vector=weights)


# %% Feature reduction step:

top_features = np.argsort(abs(ws))[::-1]
train_features = []
train_scores = []
test_scores = []
from tqdm import tqdm

for i in tqdm(range(342)):
    train_features.append(top_features[i])
    X_t = X_train[:, train_features]
    passed, score, *_ = null_model_check(grid, X_t, y_train, cv, n_shuffles=50)
    train_scores.append(score)
    if not passed:
        test_scores.append(0)
        continue

    test_score = grid.fit(X_t, y_train).score(
        X_test[:, train_features], y_test
    )
    if abs(score - test_score) < 0.1:
        test_scores.append(test_score)
        continue
    test_scores.append(0)
train_scores = np.array(train_scores)
test_scores = np.array(test_scores)
print(np.argmax(train_scores * (test_scores != 0)) + 1)

np.savez_compressed(
    "results.npz",
    mean_weights=ws,
    significnance_matrix=w_2d,
    p_values=ps,
    feature_reduction=np.argmax(train_scores * (test_scores != 0)),
)
with open("model_res.txt", "w") as file:
    file.write(str(grid))
    file.write(str(score))
