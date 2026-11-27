from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

K = 9
ROOT = Path(__file__).resolve().parents[1]
# указать пути моделей для сборки ансамбля
files = [ROOT / "preds" / "pred_cap_gmm_tied_seed21.npy",
         ROOT / "preds" / "pred_cap_agglo_quantile.npy",
         ROOT / "preds" / "pred_cap_kmeans_robust.npy"]
P = [np.load(f, allow_pickle=False).astype(int) for f in files]
N = len(P[0])
for f, p in zip(files, P):
    assert p.shape == (N,) and p.min() >= 0 and p.max() < K, f

def align(ref, pred):
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (ref, pred), 1)
    r, c = linear_sum_assignment(-cm)
    m = np.zeros(K, dtype=int)
    m[c] = r                      # метка pred -> метка ref
    return m[pred]

n = len(P)
ari = np.array([[adjusted_rand_score(P[i], P[j]) for j in range(n)] for i in range(n)])
mean_ari = (ari.sum(1) - 1) / (n - 1)
ref = int(mean_ari.argmax())
print("опорная:", files[ref])
print(np.round(ari, 2))

A = [P[ref] if i == ref else align(P[ref], p) for i, p in enumerate(P)]
w = mean_ari / mean_ari.sum()
votes = np.zeros((N, K))
for a, wi in zip(A, w):
    votes[np.arange(N), a] += wi
ens = votes.argmax(1)

print("размеры классов:", np.bincount(ens, minlength=K))
np.save(ROOT / "preds" / "pred_dania_vote.npy", ens)
