from pathlib import Path

import pandas as pd


def main():
    project = Path(__file__).resolve().parents[1]
    data = pd.read_parquet(project / "clean_data.parquet")
    original_ids = data["ID"].copy()
    original_columns = data.columns

    seconds = data["Time"].diff().dt.total_seconds()
    movement = data["TravelingBlockPosition"].diff()
    # При совпадении времени скорость определить нельзя.
    elapsed_seconds = seconds.where(seconds > 0)
    data["block_speed"] = movement / elapsed_seconds

    # Показывает, зарегистрировал ли датчик поток в этом замере.
    flow = data["InletFlow"]
    data["flow_on"] = flow.gt(0).where(flow.notna()).astype("Int8")

    # Модуль показания позволяет сравнивать величину оборотов при разном знаке.
    data["rotor_speed_abs"] = data["Turn_rotor"].abs()

    # Изменение показания положения долота за секунду.
    data["bit_speed"] = data["BitPosition"].diff() / elapsed_seconds

    # Время с первого наблюдения текущего состояния потока.
    state = data["flow_on"]
    new_run = state.ne(state.shift()).fillna(True)
    run_start = data["Time"].where(new_run).ffill()
    data["flow_state_seconds"] = (data["Time"] - run_start).dt.total_seconds().where(state.notna())

    # Изменения между соседними замерами в единицах соответствующего датчика за секунду.
    data["depth_rate"] = data["Depth"].diff() / elapsed_seconds
    data["flow_rate_change"] = flow.diff() / elapsed_seconds
    data["pressure_rate"] = data["ManifoldPressure"].diff() / elapsed_seconds
    data["hook_load_rate"] = data["HL"].diff() / elapsed_seconds

    # Размах положения блока за последние 30 секунд, включая текущий замер.
    block_window = data.rolling("30s", on="Time", min_periods=2)["TravelingBlockPosition"]
    data["block_range_30s"] = block_window.max() - block_window.min()

    output = project / "features_v1.parquet"
    data.to_parquet(output, index=False)
    saved_ids = pd.read_parquet(output, columns=["ID"])["ID"]
    if len(saved_ids) != len(original_ids) or not saved_ids.equals(original_ids):
        raise ValueError("После записи изменилось число строк или порядок ID")

    print(f"{output.name}: {len(saved_ids)} строк, порядок ID проверен.")
    preview = ["ID", "Time", *(name for name in data.columns if name not in original_columns)]
    print(data[preview].head().to_string(index=False))


if __name__ == "__main__":
    main()
