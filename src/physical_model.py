"""Пробная модель девяти классов на устойчивых интервалах разметки Юли.

Запуск из корня проекта:
python src/physical_model.py "путь/к/submission_yulia_v11_phys_snap.csv"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score


FEATURES = [
    "InletFlow", "ManifoldPressure", "Turn_rotor", "WedgePosition",
    "HL", "Load", "ROP", "TravelingBlockPosition", "BitPosition",
    "block_speed", "bit_speed", "depth_rate", "block_range_30s",
    "flow_median_30s", "flow_deviation_30s",
    "block_change_60s", "bit_change_60s",
]
HOLDOUT_DAY = "2026-05-15"


def anchor_mask(data, labels):
    gap = data["Time"].diff().dt.total_seconds().gt(10)
    runs = (labels.ne(labels.shift()) | gap).cumsum()
    first_time = data["Time"].groupby(runs).transform("first")
    last_time = data["Time"].groupby(runs).transform("last")
    return (
        (last_time - first_time).dt.total_seconds().ge(120)
        & (data["Time"] - first_time).dt.total_seconds().ge(30)
        & (last_time - data["Time"]).dt.total_seconds().ge(30)
    )


def balanced_sample(mask, labels, limit, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(9):
        indices = np.flatnonzero(mask.to_numpy() & labels.eq(class_id).to_numpy())
        if len(indices) == 0:
            raise ValueError(f"Нет обучающих примеров класса {class_id}")
        selected.extend(rng.choice(indices, min(limit, len(indices)), replace=False))
    return np.sort(selected)


def fit_predict(train_rows, data, labels, predict_rows):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(data.iloc[train_rows][FEATURES])
    x_predict = imputer.transform(data.iloc[predict_rows][FEATURES])
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=18,
        class_weight="balanced_subsample",
        n_jobs=4,
        random_state=42,
    )
    model.fit(x_train, labels.iloc[train_rows])
    return model.predict(x_predict)


def remove_short_returns(labels, gaps, max_length=2):
    result = labels.copy()
    while True:
        starts = np.r_[
            0, np.flatnonzero((result[1:] != result[:-1]) | gaps[1:]) + 1
        ]
        ends = np.r_[starts[1:], len(result)]
        short = [
            j for j in range(1, len(starts) - 1)
            if ends[j] - starts[j] <= max_length
            and not gaps[starts[j]]
            and not gaps[ends[j]]
            and result[starts[j - 1]] == result[starts[j + 1]]
        ]
        if not short:
            return result
        for j in short:
            result[starts[j]:ends[j]] = result[starts[j - 1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path, help="CSV Юли с ID,operation")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data = pd.read_parquet(project / "features_v2.parquet")
    times = data["Time"].to_numpy()
    previous = np.searchsorted(times, times - np.timedelta64(60, "s"), side="right") - 1
    valid = previous >= 0
    valid[valid] &= (
        (times[valid] - times[previous[valid]]) / np.timedelta64(1, "s") <= 70
    )
    for source, feature in (
        ("TravelingBlockPosition", "block_change_60s"),
        ("BitPosition", "bit_change_60s"),
    ):
        change = np.full(len(data), np.nan)
        values = data[source].to_numpy()
        change[valid] = values[valid] - values[previous[valid]]
        data[feature] = change
    supplied = pd.read_csv(args.labels)
    if supplied.columns.tolist() != ["ID", "operation"]:
        raise ValueError("Ожидаются колонки ID,operation")
    if not data["ID"].equals(supplied["ID"]):
        raise ValueError("ID или порядок строк не совпадают")
    labels = supplied["operation"]
    if labels.isna().any() or sorted(labels.unique().tolist()) != list(range(9)):
        raise ValueError("Ожидаются девять целых меток 0..8 без пропусков")

    anchors = anchor_mask(data, labels)
    gaps = data["Time"].diff().dt.total_seconds().gt(10).to_numpy()
    holdout = data["Time"].dt.strftime("%Y-%m-%d").eq(HOLDOUT_DAY)
    train_rows = balanced_sample(anchors & ~holdout, labels, 4000, seed=42)
    test_rows = np.flatnonzero(holdout.to_numpy())
    test_predictions = remove_short_returns(
        fit_predict(train_rows, data, labels, test_rows), gaps[test_rows]
    )
    true = labels.iloc[test_rows].to_numpy()
    present = np.unique(true)
    print(f"Отложенный день {HOLDOUT_DAY}: {len(test_rows)} строк")
    print(f"Согласие с Юлей: {accuracy_score(true, test_predictions):.3f}")
    print(f"Macro F1 по присутствующим классам {present.tolist()}: "
          f"{f1_score(true, test_predictions, labels=present, average='macro'):.3f}")
    print("Расхождения по классам Юли:")
    print(pd.crosstab(pd.Series(true, name="Юля"), pd.Series(test_predictions, name="Модель")))

    all_rows = balanced_sample(anchors, labels, 4000, seed=42)
    predictions = remove_short_returns(
        fit_predict(all_rows, data, labels, np.arange(len(data))), gaps
    )
    if len(predictions) != len(data) or set(np.unique(predictions)) != set(range(9)):
        raise ValueError("Предсказание должно содержать 9 классов для всех ID")
    output = project / "preds" / "pred_nikita_physical_v1.npy"
    output.parent.mkdir(exist_ok=True)
    np.save(output, predictions.astype(np.int8))
    saved = np.load(output)
    if not np.array_equal(predictions, saved):
        raise ValueError("Сохранённые предсказания отличаются от расчёта")
    print(f"Сохранено {len(saved)} меток в {output}")
    print("Распределение классов:", np.bincount(saved, minlength=9).tolist())
    print("Переключений:", int(np.count_nonzero(saved[1:] != saved[:-1])))


if __name__ == "__main__":
    main()
