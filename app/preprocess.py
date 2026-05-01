from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.utils import infer_time_delta


@dataclass
class PreprocessConfig:
    timestamp_column: str = "timestamp"
    window_size: int = 20
    window_stride: int = 5
    scaling_method: str = "standard"


@dataclass
class PreparedData:
    cleaned_df: pd.DataFrame
    scaled_df: pd.DataFrame
    raw_window_features: pd.DataFrame
    model_features: pd.DataFrame
    sensor_columns: list[str]
    feature_columns: list[str]
    timestamp_column: str
    window_size: int
    window_stride: int
    time_delta: pd.Timedelta
    scaling_method: str


def _build_scaler(method: str):
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValueError("Поддерживаются только методы масштабирования 'standard' и 'minmax'.")


def _clean_frame(frame: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, list[str]]:
    cleaned = frame.copy()
    timestamp_column = config.timestamp_column
    sensor_columns = [column for column in cleaned.columns if column != timestamp_column]

    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")
    invalid_timestamps = int(cleaned[timestamp_column].isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"Не удалось преобразовать {invalid_timestamps} значений столбца времени в datetime."
        )

    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values(timestamp_column).reset_index(drop=True)
    cleaned = cleaned.groupby(timestamp_column, as_index=False)[sensor_columns].mean()

    if cleaned.empty:
        raise ValueError("После удаления дубликатов не осталось данных для анализа.")

    numeric = cleaned[sensor_columns].copy()
    if numeric.isna().all().any():
        bad_columns = numeric.columns[numeric.isna().all()].tolist()
        raise ValueError(
            "Следующие параметры полностью состоят из пропусков и не могут быть обработаны: "
            + ", ".join(bad_columns)
        )

    numeric = numeric.interpolate(method="linear", limit_direction="both")
    numeric = numeric.ffill().bfill()
    cleaned[sensor_columns] = numeric
    return cleaned, sensor_columns


def _compute_window_features(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    timestamp_column: str,
    window_size: int,
    window_stride: int,
) -> pd.DataFrame:
    total_rows = len(frame)
    if total_rows < 4:
        raise ValueError("Для анализа требуется минимум 4 наблюдения.")

    effective_window = max(4, min(window_size, total_rows))
    starts = list(range(0, max(total_rows - effective_window + 1, 1), max(1, window_stride)))
    last_start = max(total_rows - effective_window, 0)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    starts = sorted(set(starts))

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for window_id, start_idx in enumerate(starts):
        end_idx = min(start_idx + effective_window, total_rows)
        window = frame.iloc[start_idx:end_idx]
        row: dict[str, float | int | pd.Timestamp] = {
            "window_id": window_id,
            "window_start": window[timestamp_column].iloc[0],
            "window_end": window[timestamp_column].iloc[-1],
            "observation_count": len(window),
        }
        positions = np.arange(len(window))

        for sensor in sensor_columns:
            values = window[sensor].to_numpy(dtype=float)
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0))
            slope = float(np.polyfit(positions, values, 1)[0]) if len(values) > 1 else 0.0
            row[f"{sensor}__mean"] = mean_value
            row[f"{sensor}__std"] = std_value
            row[f"{sensor}__var"] = float(np.var(values, ddof=0))
            row[f"{sensor}__min"] = float(np.min(values))
            row[f"{sensor}__max"] = float(np.max(values))
            row[f"{sensor}__median"] = float(np.median(values))
            row[f"{sensor}__range"] = float(np.max(values) - np.min(values))
            row[f"{sensor}__last"] = float(values[-1])
            row[f"{sensor}__delta"] = float(values[-1] - values[0])
            row[f"{sensor}__slope"] = slope
            row[f"{sensor}__last_dev"] = float(values[-1] - mean_value)
        rows.append(row)

    return pd.DataFrame(rows)


def preprocess_timeseries(
    frame: pd.DataFrame,
    config: PreprocessConfig | None = None,
) -> PreparedData:
    config = config or PreprocessConfig()
    cleaned_df, sensor_columns = _clean_frame(frame=frame, config=config)

    point_scaler = _build_scaler(config.scaling_method)
    scaled_values = point_scaler.fit_transform(cleaned_df[sensor_columns])
    scaled_df = cleaned_df[[config.timestamp_column]].copy()
    scaled_df[sensor_columns] = scaled_values

    raw_window_features = _compute_window_features(
        frame=cleaned_df,
        sensor_columns=sensor_columns,
        timestamp_column=config.timestamp_column,
        window_size=config.window_size,
        window_stride=config.window_stride,
    )

    feature_columns = [
        column
        for column in raw_window_features.columns
        if column not in {"window_id", "window_start", "window_end", "observation_count"}
    ]
    feature_scaler = _build_scaler(config.scaling_method)
    model_matrix = feature_scaler.fit_transform(raw_window_features[feature_columns])
    model_features = pd.DataFrame(model_matrix, columns=feature_columns, index=raw_window_features.index)

    effective_window_size = int(raw_window_features["observation_count"].max())
    return PreparedData(
        cleaned_df=cleaned_df,
        scaled_df=scaled_df,
        raw_window_features=raw_window_features,
        model_features=model_features,
        sensor_columns=sensor_columns,
        feature_columns=feature_columns,
        timestamp_column=config.timestamp_column,
        window_size=effective_window_size,
        window_stride=max(1, config.window_stride),
        time_delta=infer_time_delta(cleaned_df[config.timestamp_column]),
        scaling_method=config.scaling_method,
    )