"""Проверка двух наблюдаемых типов движения на отложенных днях."""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    path = Path(__file__).resolve().parents[1] / "report" / "motion_intervals.csv"
    with path.open(newline="", encoding="utf-8") as source:
        intervals = list(csv.DictReader(source))

    labels = np.array([row["pattern"] == "both" for row in intervals])
    days = np.array([row["start_time"][:10] for row in intervals])
    for columns in (["wedge_median"], ["hl_median"], ["wedge_median", "hl_median"]):
        values = np.array([[float(row[column]) for column in columns] for row in intervals])
        predicted = np.empty_like(labels)
        for day in np.unique(days):
            train = days != day
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced"),
            )
            model.fit(values[train], labels[train])
            predicted[~train] = model.predict(values[~train])
        mistakes = [
            (row["first_ID"], row["last_ID"])
            for row, expected, actual in zip(intervals, labels, predicted)
            if expected != actual
        ]
        print(f"{', '.join(columns)}: {len(intervals) - len(mistakes)}/{len(intervals)}")
        print(f"  Ошибки по ID: {mistakes}")


if __name__ == "__main__":
    main()
