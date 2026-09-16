#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:14:48 2026

@author: Joseph Hadous

"""
import numpy as np
import os
import pandas as pd
from sklearn.utils import resample
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeavePGroupsOut,
    GridSearchCV,
    cross_val_score,
)
from sklearn.svm import LinearSVC


base_dir = "/Users/hh1850/Dropbox/COLD_FC_SVM_CODE/"


def getWarm(d):
    return pd.concat([d[72:112], d[336:376]])


def getCool(d):
    return pd.concat([d[168:208], d[456:496]])

csv_folder = "/Path/To/CSVs/"
csv_list = [
    pd.read_csv(csv_folder+ d)
    for d in os.listdir(csv_folder)
    if ".csv" in d
]
X = np.concatenate(
    (
        [getWarm(d).corr() for d in csv_list],
        [getCool(d).corr() for d in csv_list],
    ),
    axis=0,
)
np.save(arr=X, file="W_C.npy")
y = np.concatenate((np.zeros(61), np.ones(61)))


def upperTri(ar):
    ui = np.triu_indices(ar.shape[0], k=1)
    return ar[ui]


X_ = np.zeros((122, 34191))
for i in range(122):
    X_[i] = np.atanh(upperTri(X[i]))

groups = np.array([i for i in range(61)] * 2)
lpgo = LeavePGroupsOut(n_groups=1)
lpgo.get_n_splits(groups=groups)

pipe = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC())])
param_grid = {"svc__C": 10.0 ** np.arange(-4, 2)}
coarse_grid = GridSearchCV(
    pipe,
    param_grid,
    n_jobs=-1,
    cv=lpgo,
    verbose=0,
)
coarse_grid.fit(X_, y, groups=groups)
print(coarse_grid.best_score_)
best_c = coarse_grid.best_estimator_.get_params()["svc__C"]
param_grid = {
    "svc_C": np.concat(
        np.arange(best_c / 10 * 2.5, best_c, best_c / 4),
        [best_c],
        np.arange(best_c * 2.5, best_c * 10, best_c * 10 / 4),
    )
}

grid = GridSearchCV(
    pipe,
    param_grid,
    n_jobs=-1,
    cv=lpgo,
    verbose=0,
)
grid.fit(X_, y, groups=groups)
print(grid.best_score_)
# Manually check to ensure that this actually imporved accuracy,
# May need to check more parameters to see if this is truly best


weights = np.zeros((10000, 34191))
svm = grid.best_estimator_.get_params()["svc"]


def gw():
    svm = grid.best_estimator_.get_params()["svc"]
    s = StandardScaler()
    x1, y1 = resample(X_, y, stratify=y)
    x1 = s.fit_transform(x1)
    return svm.fit(x1, y1).coef_.flatten()


ws = Parallel(n_jobs=-1, verbose=10, return_as="generator")(
    delayed(gw)() for i in range(10000)
)


for i, w in enumerate(ws):
    weights[i] = w


stable = ((weights > 0).mean(0) > 0.95) | (((weights < 0).mean(0) > 0.95))


def mad_outliers(x, threshold=3.5):
    """
    Detects outliers using the Median Absolute Deviation (MAD) method.

    Parameters:
        x (array-like): 1D array of numeric data
        threshold (float): modified z-score threshold to identify outliers

    Returns:
        mask (np.ndarray): Boolean array where True indicates an outlier
    """
    x = np.asarray(x)
    median = np.median(x)
    mad = np.median(np.abs(x - median))

    if mad == 0:
        # Handle zero MAD (e.g., constant or near-constant data)
        return np.zeros(len(x), dtype=bool)

    modified_z_scores = 0.6745 * (x - median) / mad
    return (modified_z_scores) > threshold


MAD = mad_outliers(abs(np.median(weights, 0)))
sigs = MAD & stable
sig = np.zeros((262, 262))
sig[np.triu_indices(262, 1)] = np.median(weights, 0) * sigs
sig += sig.T
pipe = Pipeline([("scaler", StandardScaler()), ("svc", svm)])

print(grid.best_score_)
scores = cross_val_score(
    pipe, X_, y, cv=lpgo, n_jobs=-1, groups=groups, scoring="accuracy"
)
roc_auc_scores = cross_val_score(
    pipe, X_, y, cv=lpgo, n_jobs=-1, groups=groups, scoring="roc_auc"
)  # may fail or error


print("Accuracy: %.3f +/-(%.3f)" % (np.mean(scores), np.std(scores)))

os.makedirs(f"{base_dir}/pydata", exist_ok=True)
np.save(f"{base_dir}/pydata/Sig.npy", sig)
with open("acc_score.txt", "w") as file:
    file.write("Accuracy: %.3f +/-(%.3f)" % (np.mean(scores), np.std(scores)))
    file.write(
        "AUC: %.3f +/-(%.3f)"
        % (np.mean(roc_auc_scores), np.std(roc_auc_scores))
    )
with open(f"{base_dir}/model.txt", "w") as file:
    file.write(str(pipe))

np.savez_compressed(
    f"{base_dir}/pydata/all_weights.npz", weights, allow_pickle=True
)
np.save(arr=X, file=f"{base_dir}/pydata/W_C.npy")
