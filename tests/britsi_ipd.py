import json 
import numpy as np
import matplotlib.pyplot as plt
from pypots.optim import Adam
from pypots.imputation import CSDI, BRITS
from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae
import pickle
import sys
from time import time
set_random_seed(1234)
# check that GPU acceleration is enabled
import torch
torch.cuda.device_count()
# print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CUDA ENABLED: {torch.cuda.is_available()}")


def evaluate_fold(model, fold, Xs, ys, fold_idxs, windows_by_pm):
    # make the splits
    X_train_fold = Xs[fold_idxs[fold]["train"]]
    y_train_fold = ys[fold_idxs[fold]["train"]]
    X_test_fold = Xs[fold_idxs[fold]["test"]]
    y_test_fold = ys[fold_idxs[fold]["test"]]
    # check class distributions
    counts_tr = np.unique(y_train_fold, return_counts=True)[1]
    print(f"Training class distribution: {counts_tr/np.sum(counts_tr)}")
    counts_te = np.unique(y_test_fold, return_counts=True)[1]
    print(f"Testing class distribution: {counts_te/np.sum(counts_te)}")
    print(f"Training model on fold {fold}...", end="")
    model.fit(train_set={'X':X_train_fold})
    print("Finished training!")
    # loop over % missing
    errs = {}
    for pm, windows in windows_by_pm.items():
        errs_by_pm = np.zeros((len(windows), X_test_fold.shape[0]))

        for (idx, widx) in enumerate(windows):
            X_test_corrupted = X_test_fold.copy()
            X_test_corrupted[:, widx] = np.nan
            mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
            imputed = model.impute(test_set={'X': X_test_corrupted})
            errs_by_pm[idx, :] = np.fromiter(map(calc_mae, imputed, X_test_fold, mask), dtype=np.float64) # get individual errors for uncertainty quantification

        errs[pm] = errs_by_pm
    return errs

def evaluate_folds(model, nfolds, Xs, ys, fold_idxs, windows_by_pm):
    # make the splits
    errs = {}
    nsamples =  Xs[fold_idxs[0]["test"]].shape[0]

    for pm, windows in windows_by_pm.items():
        errs[pm] = -np.ones((nfolds, len(windows), nsamples))

    tstart = time()
    for fold in range(nfolds):
        X_train_fold = Xs[fold_idxs[fold]["train"]]
        y_train_fold = ys[fold_idxs[fold]["train"]]
        X_test_fold = Xs[fold_idxs[fold]["test"]]
        y_test_fold = ys[fold_idxs[fold]["test"]]
        

        # check class distributions
        counts_tr = np.unique(y_train_fold, return_counts=True)[1]
        print(f"Training class distribution: {counts_tr/np.sum(counts_tr)}")
        counts_te = np.unique(y_test_fold, return_counts=True)[1]
        print(f"Testing class distribution: {counts_te/np.sum(counts_te)}")

        print(f"(t={round(time() - tstart,2)}s) Training model on fold {fold}/{nfolds}...")
        model.fit(train_set={'X':X_train_fold})

        # loop over % missing
        for pm, windows in windows_by_pm.items():
            print(f"(t={round(time() - tstart,2)}s) fold {fold}: Testing model on {pm}% missing")
            for (idx, widx) in enumerate(windows):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
                imputed = model.impute(test_set={'X': X_test_corrupted})
                errs[pm][fold, idx, :] = np.fromiter(map(calc_mae, imputed, X_test_fold, mask), dtype=np.float64) # get individual errors for uncertainty quantification

    return errs

train_f = np.loadtxt("../Folds/IPD/ItalyPowerDemand_TRAIN.txt")
test_f = np.loadtxt("../Folds/IPD/ItalyPowerDemand_TEST.txt")
X_train = train_f[:, 1:]
y_train = train_f[:, 0]
X_test = test_f[:, 1:]
y_test = test_f[:, 0]

# reshape data because pypots wants multivariate
X_train_3D = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_3D =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_3D = y_train
y_test_3D = y_test

# stack for resampling
Xs = np.vstack([X_train_3D, X_test_3D])
print(Xs.shape)
ys = np.concatenate([y_train_3D, y_test_3D])
print(ys.shape)



# load resample fold indices
with open("../Folds/IPD/ipd_resample_folds_python_idx.json", "r") as f:
    resample_fold_idxs_f = json.load(f)
resample_fold_idxs = {int(k): v for k, v in resample_fold_idxs_f.items()}

# load imputation window indices
with open("../Folds/IPD/ipd_windows_python_idx.json", "r") as f:
    window_idxs_f = json.load(f)
window_idxs = {int(float(k)*100): v for k, v in window_idxs_f.items()}
# print(window_idxs.keys())


n_steps = len(X_test_3D[0])
n_features = 1
rnn_hidden_size=128
batch_size=32
epochs=250
optimizer=Adam(lr=1e-3)
num_workers=0
device=None # infer the best device to use
model_saving_strategy=None

britsi = BRITS(
    n_steps=n_steps,
    n_features=n_features,
    rnn_hidden_size=rnn_hidden_size,
    batch_size=batch_size,
    use_BRITSI=True,
    epochs=epochs,
    optimizer=optimizer,
    num_workers=num_workers,
    device=device, 
    model_saving_strategy=model_saving_strategy
)

# # fold_scores_brits = evaluate_folds(brits, 3, Xs, ys, resample_fold_idxs, [range(5,15)])
fold_scores_britsi = evaluate_folds(britsi, 30, Xs, ys, resample_fold_idxs, window_idxs)


# print("IPD BRITS Mean MAE: {}".format(np.mean(fold_scores_brits)))
print("IPD BRITS-I Mean MAE:")
for pm in window_idxs:
    print(f"{pm}%:", np.mean(fold_scores_britsi[pm]))

# fold_scores_britsi = {5:10, 10:20}
with open("IPD_britsi_results.pkl", "wb") as f:
    pickle.dump(fold_scores_britsi, f)

