"""python report/audit_baseline.py path/to/submission_baseline.csv"""
from pathlib import Path
import io
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


root = Path(__file__).resolve().parents[1]
ref = "ddd0085"
data = pd.read_parquet(root / "data" / "clean_data.parquet")
csv = pd.read_csv(sys.argv[1])
assert csv.shape[1] == 2 and csv.columns[0] == "ID"
assert data.ID.is_unique and data.ID.equals(csv.ID) and data.Time.is_monotonic_increasing
baseline = csv.iloc[:, 1].to_numpy()
assert baseline.dtype.kind in "iu" and set(baseline) == set(range(9))
files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref], cwd=root).decode().splitlines()


def switches(labels):
    return int(np.sum(labels[1:] != labels[:-1]))


def short_returns(labels):
    return int(np.sum((labels[:-2] == labels[2:]) & (labels[:-2] != labels[1:-1])))


for path in [p for p in files if p.endswith(".npy")]:
    raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=root)
    labels = np.load(io.BytesIO(raw), allow_pickle=False)
    assert labels.shape == baseline.shape and set(labels) == set(range(9))
    print(path, "ARI", round(adjusted_rand_score(baseline, labels), 4),
          "switches", switches(labels), "ABA", short_returns(labels))
    if path == "pred_dania_vote.npy":
        vote = labels
print("baseline", switches(baseline), short_returns(baseline), np.bincount(baseline))

# Окна после разрыва более 10 секунд начинаются заново.
gap = data.Time.diff().dt.total_seconds().gt(10)
cols = ["TravelingBlockPosition", "BitPosition"]
indexed = data.set_index("Time")
window = indexed[cols].groupby(gap.cumsum().to_numpy()).rolling("30s", min_periods=2)
span = (window.max() - window.min()).set_axis(data.index)
eligible = data.InletFlow.eq(0) & data.Turn_rotor.abs().lt(1)
examples = []
for threshold in [0.3, 0.5, 1.0]:
    for name, bit in [("block_only", span.BitPosition.lt(0.05)), ("both", span.BitPosition.gt(threshold))]:
        mask = eligible & span.TravelingBlockPosition.gt(threshold) & bit
        group = (mask.ne(mask.shift()) | gap).cumsum()
        intervals = []
        for _, part in data[mask].groupby(group[mask]):
            duration = (part.Time.iloc[-1] - part.Time.iloc[0]).total_seconds()
            if duration >= 120:
                intervals.append((duration, part))
        if not intervals:
            print(name, threshold, "no intervals")
            continue
        idx = np.concatenate([s.index for _, s in intervals])
        def medians(col):
            return round(float(np.median([s[col].median() for _, s in intervals])), 3)
        print(name, threshold, len(intervals), len(idx), "episode medians wedge/HL/load",
              [medians(c) for c in ["WedgePosition","HL","Load"]])
        if threshold == 0.5:
            for duration, s in intervals:
                examples.append(dict(pattern=name, first_ID=int(s.ID.iloc[0]), last_ID=int(s.ID.iloc[-1]),
                                     start_time=s.Time.iloc[0], seconds=duration, rows=len(s),
                                     wedge_median=s.WedgePosition.median(), hl_median=s.HL.median(),
                                     baseline_mode=int(np.bincount(baseline[s.index]).argmax()),
                                     vote_mode=int(np.bincount(vote[s.index]).argmax())))
            for method, arr in [("baseline",baseline), ("vote",vote)]:
                print(method, np.bincount(arr[idx],minlength=9))
            for duration,s in sorted(intervals,key=lambda x:-x[0])[:3]:
                print("example",s.ID.iloc[[0,-1]].tolist(),duration,
                      s[cols].max().sub(s[cols].min()).round(2).tolist(),
                      "wedge/HL",s[["WedgePosition","HL"]].median().round(2).tolist(),
                      "dominant",*[int(np.bincount(a[s.index]).argmax()) for a in (baseline,vote)])

catalog = pd.DataFrame(examples).sort_values("first_ID")
catalog.to_csv(root / "report" / "motion_intervals.csv", index=False)
for name, g in catalog.groupby("pattern"):
    print(name, "start days", g.start_time.dt.date.nunique(), "wedge median > 0.5:",
          int((g.wedge_median > 0.5).sum()), "/", len(g))
