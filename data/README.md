# Данные

- `clean_data.parquet` — очищенные исходные замеры.
- `features_v1.parquet` — те же 180 901 строк и 10 расчётных признаков.
- `features_v2.parquet` — все признаки v1 и два дополнительных признака потока.

Порядок `ID` во всех трёх файлах одинаковый. Таблицы признаков пересобираются из корня проекта командами `python .\src\features.py --version v1` и `python .\src\features.py --version v2`. Описание формул — в [`report/nikita_features.md`](../report/nikita_features.md).
