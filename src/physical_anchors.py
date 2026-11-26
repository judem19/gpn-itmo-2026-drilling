"""Собрать проверенные якорные строки двух типов движения."""

import csv
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    intervals = project / "report" / "motion_intervals.csv"
    output = project / "report" / "physical_anchors.csv"

    anchors = {}
    accepted = 0
    skipped = []
    with intervals.open(newline="", encoding="utf-8") as source:
        for interval in csv.DictReader(source):
            pattern = interval["pattern"]
            if pattern not in {"block_only", "both"}:
                raise ValueError(f"Неизвестный тип интервала: {pattern}")
            wedge = float(interval["wedge_median"])
            consistent = wedge <= 0.5 if pattern == "block_only" else wedge > 0.5
            if not consistent:
                skipped.append((int(interval["first_ID"]), int(interval["last_ID"])))
                continue
            accepted += 1
            for row_id in range(int(interval["first_ID"]), int(interval["last_ID"]) + 1):
                if row_id in anchors:
                    raise ValueError(f"ID {row_id} попал в два интервала")
                anchors[row_id] = pattern

    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["ID", "anchor_type"])
        writer.writerows(sorted(anchors.items()))

    print(f"{accepted} интервалов, {len(anchors)} якорных строк")
    print(f"Исключены переходные интервалы: {skipped}")
    print(output)


if __name__ == "__main__":
    main()
