from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.detect import DetectionResult
from app.preprocess import PreparedData


FEATURE_REASON_MAP = {
    'mean': 'сместился средний уровень параметра {sensor}',
    'std': 'увеличилась локальная дисперсия параметра {sensor}',
    'var': 'вырос разброс значений параметра {sensor}',
    'min': 'минимальные значения параметра {sensor} заметно ушли от нормы',
    'max': 'пиковые значения параметра {sensor} вышли за рабочий диапазон',
    'median': 'сместилась медиана параметра {sensor}',
    'range': 'расширился диапазон колебаний параметра {sensor}',
    'last': 'текущий уровень параметра {sensor} резко изменился',
    'delta': 'скорость изменения параметра {sensor} резко возросла',
    'slope': 'по параметру {sensor} наблюдается выраженный тренд',
    'last_dev': 'последние значения параметра {sensor} отклоняются от локального среднего',
}


def _feature_phrase(sensor: str, metric: str) -> str:
    template = FEATURE_REASON_MAP.get(metric, 'параметр {sensor} заметно отклонился от нормального режима')
    return template.format(sensor=sensor)


def _compose_text(reasons: list[str], sensors: list[str]) -> str:
    if not reasons:
        sensors_text = ', '.join(sensors) if sensors else 'нескольких параметров'
        return f'Участок признан нестабильным из-за совокупного отклонения {sensors_text}.'
    if len(reasons) == 1:
        return f'Участок признан нестабильным, потому что {reasons[0]}.'
    if len(reasons) == 2:
        return f'Участок признан нестабильным из-за того, что {reasons[0]} и {reasons[1]}.'
    return (
        'Участок признан нестабильным из-за того, что '
        + ', '.join(reasons[:-1])
        + f' и {reasons[-1]}.'
    )


def build_interval_explanations(
    prepared: PreparedData,
    detection: DetectionResult,
    top_n: int = 5,
) -> pd.DataFrame:
    if detection.intervals.empty:
        return pd.DataFrame(columns=['interval_id', 'text', 'top_sensors', 'top_features'])

    windows = detection.windows.copy()
    feature_matrix = prepared.raw_window_features[prepared.feature_columns]
    baseline_mask = windows['state'] == 'normal'
    baseline = feature_matrix.loc[baseline_mask]
    if baseline.empty:
        baseline = feature_matrix

    center = baseline.median()
    spread = baseline.std(ddof=0).replace(0, np.nan).fillna(1.0)

    rows: list[dict[str, Any]] = []
    for interval in detection.intervals.itertuples(index=False):
        interval_windows = windows.loc[
            (windows['window_start'] <= interval.end)
            & (windows['window_end'] >= interval.start)
            & (windows['state'] != 'normal')
        ]
        if interval_windows.empty:
            continue

        interval_index = interval_windows.index
        interval_profile = feature_matrix.loc[interval_index].mean(axis=0)
        deviation = ((interval_profile - center).abs() / spread).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_features_series = deviation.sort_values(ascending=False).head(top_n)

        top_features: list[dict[str, Any]] = []
        reasons: list[str] = []
        sensors: list[str] = []
        for feature_name, score in top_features_series.items():
            sensor, metric = feature_name.split('__', 1)
            sensors.append(sensor)
            phrase = _feature_phrase(sensor=sensor, metric=metric)
            if phrase not in reasons:
                reasons.append(phrase)
            top_features.append(
                {
                    'feature': feature_name,
                    'sensor': sensor,
                    'metric': metric,
                    'score': round(float(score), 4),
                }
            )

        ordered_sensors = list(dict.fromkeys(sensors))
        rows.append(
            {
                'interval_id': int(interval.interval_id),
                'text': _compose_text(reasons[:3], ordered_sensors[:3]),
                'top_sensors': ordered_sensors[:3],
                'top_features': top_features,
            }
        )

    return pd.DataFrame(rows)