from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the uploaded file does not match the required schema."""


def _resolve_timestamp_column(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    normalized = {str(column).strip().lower(): str(column).strip() for column in frame.columns}
    target = timestamp_column.strip().lower()
    if target not in normalized:
        available = ", ".join(str(column) for column in frame.columns)
        raise DataValidationError(
            f"Обязательный столбец времени '{timestamp_column}' не найден. "
            f"Доступные столбцы: {available}."
        )

    actual_name = normalized[target]
    if actual_name != timestamp_column:
        frame = frame.rename(columns={actual_name: timestamp_column})
    else:
        frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _read_dataframe(file_source: str | Path | bytes | bytearray | BinaryIO, file_name: str | None) -> pd.DataFrame:
    if isinstance(file_source, (str, Path)):
        file_path = Path(file_source)
        file_name = file_name or file_path.name
        extension = file_path.suffix.lower()
        source = file_path
    else:
        if not file_name:
            raise DataValidationError('Не удалось определить имя файла и его формат.')
        extension = Path(file_name).suffix.lower()
        raw_bytes = file_source if isinstance(file_source, (bytes, bytearray)) else file_source.read()
        source = io.BytesIO(raw_bytes)

    try:
        if extension == '.csv':
            return pd.read_csv(source)
        if extension == '.xlsx':
            return pd.read_excel(source, engine='openpyxl')
    except Exception as exc:
        raise DataValidationError(f"Не удалось прочитать файл '{file_name}': {exc}") from exc

    raise DataValidationError(
        'Поддерживаются только файлы форматов CSV и XLSX. '
        f"Получено расширение: '{extension or 'без расширения'}'."
    )


def validate_timeseries_dataframe(
    frame: pd.DataFrame,
    timestamp_column: str = 'timestamp',
) -> pd.DataFrame:
    if frame.empty:
        raise DataValidationError('Файл пустой. Загрузите набор данных хотя бы с одной строкой.')

    frame = _resolve_timestamp_column(frame.copy(), timestamp_column=timestamp_column)
    sensor_columns = [column for column in frame.columns if column != timestamp_column]
    if not sensor_columns:
        raise DataValidationError(
            'После столбца времени не найдено ни одного технологического параметра.'
        )

    validation_errors: list[str] = []
    for column in sensor_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            converted = pd.to_numeric(series, errors='coerce')
        else:
            normalized = (
                series.astype(str)
                .str.strip()
                .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'null': pd.NA})
                .str.replace(' ', '', regex=False)
                .str.replace(',', '.', regex=False)
            )
            converted = pd.to_numeric(normalized, errors='coerce')
            converted[series.isna()] = pd.NA

        invalid_mask = series.notna() & converted.isna()
        if invalid_mask.any():
            samples = ', '.join(map(str, series[invalid_mask].astype(str).head(3).tolist()))
            validation_errors.append(
                f"{column}: найдено {int(invalid_mask.sum())} нечисловых значений "
                f"(например: {samples})"
            )

        if converted.isna().all():
            validation_errors.append(
                f'{column}: столбец не содержит ни одного корректного числового значения.'
            )
        frame[column] = converted

    if validation_errors:
        raise DataValidationError(
            'Структура файла не прошла валидацию:\n- ' + '\n- '.join(validation_errors)
        )

    return frame


def load_timeseries(
    file_source: str | Path | bytes | bytearray | BinaryIO,
    file_name: str | None = None,
    timestamp_column: str = 'timestamp',
) -> pd.DataFrame:
    frame = _read_dataframe(file_source=file_source, file_name=file_name)
    return validate_timeseries_dataframe(frame, timestamp_column=timestamp_column)