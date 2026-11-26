"""Локальное сравнение разметок v1 и v2 без эталонных классов."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Проверить медиану HL поверх v2")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    v1 = pd.read_parquet(project / "features_v1.parquet")
    v2 = pd.read_parquet(project / "features_v2.parquet")
    pd.testing.assert_frame_equal(v1, v2[v1.columns])
    variants = (("v1", v1), ("v2", v2))
    if args.load:
        median = v2.rolling("30s", on="Time", min_periods=2)["HL"].median()
        variants = (("v2", v2), ("v2+HL", v2.assign(hl_median_30s=median)))
    examples = [18126, 115695, 116560, 33731]
    rows = v1.set_index("ID").index.get_indexer(examples)
    if (rows < 1).any() or (rows >= len(v1) - 1).any():
        raise ValueError("Для одного из выбранных ID отсутствует окно из трёх строк")

    # Одни и те же девять исходных замеров задают начальные центры в каждой паре запусков.
    for seed in (0, 21, 123):
        centers = np.random.default_rng(seed).choice(len(v1), size=9, replace=False)
        for name, data in variants:
            x = data.drop(columns=["ID", "Time"])
            x = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(x))
            model = KMeans(n_clusters=9, init=x[centers], n_init=1, random_state=seed)
            with threadpool_limits(limits=1):
                labels = model.fit_predict(x)
            boundaries = labels[:-1] != labels[1:]
            aba = boundaries[:-1] & (labels[:-2] == labels[2:])
            print(f"{seed=} {name}: classes={len(np.unique(labels))} "
                  f"switches={boundaries.sum()} ABA={aba.sum()}")
            for point_id, i in zip(examples, rows):
                print(f"  ID {point_id}: {labels[i-1:i+2].tolist()}")


if __name__ == "__main__":
    main()
