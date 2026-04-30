from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.preprocess import PreparedData


@dataclass
class DetectionConfig:
    contamination: float = 0.08
    random_state: int = 42
    warning_quantile: float = 0.8


@dataclass
class DetectionResult:
    windows: pd.DataFrame
    intervals: pd.DataFrame
    score_thresholds: dict[str, float]
    model: IsolationForest


def _assess_periodic_stability(
    cleaned_df: pd.DataFrame,
    sensor_columns: list[str],
) -> dict[str, object]:
    if len(cleaned_df) < 60:
        return {'is_periodic_stable': False, 'sensor_metrics': []}

    stable_sensors = 0
    sensor_metrics: list[dict[str, float | str | bool]] = []
    for sensor in sensor_columns:
        series = cleaned_df[sensor].astype(float).reset_index(drop=True)
        centered = series - series.mean()
        max_lag = min(len(series) // 3, 120)

        correlations: list[float] = []
        for lag in range(3, max_lag + 1):
            left = centered.iloc[:-lag]
            right = centered.iloc[lag:]
            if left.std(ddof=0) == 0 or right.std(ddof=0) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(left, right)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            correlations.append(corr)

        best_corr = max(correlations) if correlations else 0.0
        roll_window = max(8, len(series) // 20)
        min_periods = max(4, roll_window // 2)

        rolling_std = series.rolling(roll_window, min_periods=min_periods).std().dropna()
        amplitude_cv = (
            float(rolling_std.std(ddof=0) / (rolling_std.mean() or 1.0))
            if not rolling_std.empty
            else 0.0
        )

        abs_diff = series.diff().abs().dropna()
        rolling_diff = abs_diff.rolling(roll_window, min_periods=min_periods).mean().dropna()
        diff_cv = (
            float(rolling_diff.std(ddof=0) / (rolling_diff.mean() or 1.0))
            if not rolling_diff.empty
            else 0.0
        )

        abs_accel = series.diff().diff().abs().dropna()
        spike_ratio = (
            float(abs_accel.quantile(0.99) / (abs_accel.median() or 1.0))
            if not abs_accel.empty
            else 0.0
        )

        is_stable_periodic_sensor = (
            best_corr >= 0.97
            and amplitude_cv <= 0.55
            and diff_cv <= 0.55
            and spike_ratio <= 6.0
        )
        if is_stable_periodic_sensor:
            stable_sensors += 1

        sensor_metrics.append(
            {
                'sensor': sensor,
                'best_corr': round(best_corr, 4),
                'amplitude_cv': round(amplitude_cv, 4),
                'diff_cv': round(diff_cv, 4),
                'spike_ratio': round(spike_ratio, 4),
                'is_stable_periodic_sensor': is_stable_periodic_sensor,
            }
        )

    required_stable_sensors = max(1, int(np.ceil(len(sensor_columns) * 0.75)))
    return {
        'is_periodic_stable': stable_sensors >= required_stable_sensors,
        'sensor_metrics': sensor_metrics,
    }


def _compute_feature_deviation(
    raw_window_features: pd.DataFrame,
    feature_columns: list[str],
    normal_mask: pd.Series,
) -> pd.DataFrame:
    baseline = raw_window_features.loc[normal_mask, feature_columns]
    if baseline.empty:
        baseline = raw_window_features[feature_columns]

    center = baseline.median()
    spread = (baseline - center).abs().median()
    spread = spread.replace(0, np.nan)
    spread = spread.fillna(baseline.std(ddof=0).replace(0, np.nan)).fillna(1.0)
    deviation = ((raw_window_features[feature_columns] - center).abs() / spread).fillna(0.0)
    return deviation


def _attach_contributors(
    prepared: PreparedData,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    normal_mask = windows['state'] == 'normal'
    feature_deviation = _compute_feature_deviation(
        raw_window_features=prepared.raw_window_features,
        feature_columns=prepared.feature_columns,
        normal_mask=normal_mask,
    )

    contributor_scores = pd.DataFrame(index=windows.index)
    for sensor in prepared.sensor_columns:
        sensor_features = [column for column in prepared.feature_columns if column.startswith(f'{sensor}__')]
        contributor_scores[sensor] = feature_deviation[sensor_features].mean(axis=1)

    windows = windows.copy()
    windows['main_contributor'] = contributor_scores.idxmax(axis=1)
    windows['contributor_score'] = contributor_scores.max(axis=1).round(4)
    windows['dominant_feature'] = feature_deviation.idxmax(axis=1)
    return windows


def _merge_abnormal_windows(windows: pd.DataFrame, gap_tolerance: pd.Timedelta) -> pd.DataFrame:
    abnormal = windows.loc[windows['state'] != 'normal'].sort_values('window_start')
    if abnormal.empty:
        return pd.DataFrame(
            columns=[
                'interval_id',
                'start',
                'end',
                'duration',
                'average_score',
                'max_score',
                'peak_state',
                'main_contributor',
                'window_count',
            ]
        )

    merged: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for row in abnormal.itertuples(index=False):
        if current is None:
            current = {
                'start': row.window_start,
                'end': row.window_end,
                'scores': [float(row.anomaly_score)],
                'states': [row.state],
                'contributors': [row.main_contributor],
                'window_count': 1,
            }
            continue

        if row.window_start <= current['end'] + gap_tolerance:
            current['end'] = max(current['end'], row.window_end)
            current['scores'].append(float(row.anomaly_score))
            current['states'].append(row.state)
            current['contributors'].append(row.main_contributor)
            current['window_count'] += 1
            continue

        merged.append(current)
        current = {
            'start': row.window_start,
            'end': row.window_end,
            'scores': [float(row.anomaly_score)],
            'states': [row.state],
            'contributors': [row.main_contributor],
            'window_count': 1,
        }

    if current is not None:
        merged.append(current)

    rows = []
    for interval_id, interval in enumerate(merged, start=1):
        states = interval['states']
        contributors = pd.Series(interval['contributors'])
        rows.append(
            {
                'interval_id': interval_id,
                'start': interval['start'],
                'end': interval['end'],
                'duration': interval['end'] - interval['start'],
                'average_score': round(float(np.mean(interval['scores'])), 4),
                'max_score': round(float(np.max(interval['scores'])), 4),
                'peak_state': 'unstable' if 'unstable' in states else 'warning',
                'main_contributor': contributors.mode().iloc[0],
                'window_count': int(interval['window_count']),
            }
        )
    return pd.DataFrame(rows)


def detect_instability(
    prepared: PreparedData,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    config = config or DetectionConfig()
    contamination = min(max(config.contamination, 0.01), 0.45)
    model = IsolationForest(
        contamination=contamination,
        random_state=config.random_state,
        n_estimators=300,
    )
    model.fit(prepared.model_features)

    periodic_report = _assess_periodic_stability(
        cleaned_df=prepared.cleaned_df,
        sensor_columns=prepared.sensor_columns,
    )
    if periodic_report['is_periodic_stable']:
        windows = prepared.raw_window_features[['window_id', 'window_start', 'window_end', 'observation_count']].copy()
        windows['anomaly_score'] = 0.0
        windows['state'] = 'normal'
        windows['main_contributor'] = prepared.sensor_columns[0]
        windows['contributor_score'] = 0.0
        windows['dominant_feature'] = ''
        intervals = pd.DataFrame(
            columns=[
                'interval_id',
                'start',
                'end',
                'duration',
                'average_score',
                'max_score',
                'peak_state',
                'main_contributor',
                'window_count',
            ]
        )
        return DetectionResult(
            windows=windows,
            intervals=intervals,
            score_thresholds={
                'warning': 1.0,
                'unstable': 1.0,
                'periodic_guard': 1.0,
            },
            model=model,
        )

    raw_scores = -model.score_samples(prepared.model_features)
    score_range = float(raw_scores.max() - raw_scores.min()) or 1.0
    normalized_scores = (raw_scores - raw_scores.min()) / score_range
    predictions = model.predict(prepared.model_features)

    warning_threshold = float(np.quantile(normalized_scores, config.warning_quantile))
    unstable_mask = predictions == -1
    unstable_threshold = float(np.min(normalized_scores[unstable_mask])) if unstable_mask.any() else float(np.quantile(normalized_scores, 0.9))

    states = np.where(
        predictions == -1,
        'unstable',
        np.where(normalized_scores >= warning_threshold, 'warning', 'normal'),
    )

    windows = prepared.raw_window_features[['window_id', 'window_start', 'window_end', 'observation_count']].copy()
    windows['anomaly_score'] = np.round(normalized_scores, 4)
    windows['state'] = states
    windows = _attach_contributors(prepared=prepared, windows=windows)

    intervals = _merge_abnormal_windows(
        windows=windows,
        gap_tolerance=prepared.time_delta * prepared.window_stride,
    )

    return DetectionResult(
        windows=windows,
        intervals=intervals,
        score_thresholds={
            'warning': round(warning_threshold, 4),
            'unstable': round(unstable_threshold, 4),
        },
        model=model,
    )