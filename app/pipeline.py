from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.detect import DetectionConfig, DetectionResult, detect_instability
from app.explain import build_interval_explanations
from app.forecast import ForecastConfig, ForecastResult, forecast_parameter
from app.preprocess import PreparedData, PreprocessConfig, preprocess_timeseries
from app.utils import dataframe_to_records, ensure_directory, make_json_safe
from app.visualize import build_summary_metrics


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AnalysisArtifacts:
    source_name: str
    analysis_time: datetime
    raw_df: pd.DataFrame
    prepared: PreparedData
    detection: DetectionResult
    forecast: ForecastResult
    explanations: pd.DataFrame
    summary: dict[str, Any]
    conclusion: str


def _build_conclusion(
    intervals_df: pd.DataFrame,
    forecast: ForecastResult,
    target_sensor: str,
) -> str:
    if intervals_df.empty and forecast.risk_level == 'low':
        return (
            f'По анализируемому архиву выраженные нестабильные режимы не обнаружены, '
            f'а прогноз по параметру {target_sensor} указывает на низкий краткосрочный риск.'
        )

    if forecast.risk_level == 'high':
        return (
            f'Система требует внимания: обнаружены признаки нестабильности, а прогноз по параметру '
            f'{target_sensor} показывает высокий риск перехода в неблагоприятный режим.'
        )

    return (
        f'В данных обнаружены нестабильные участки. Прогноз по параметру {target_sensor} '
        f'показывает {forecast.risk_level} уровень краткосрочного риска, поэтому системе '
        f'нужен дополнительный мониторинг.'
    )


def run_analysis(
    frame: pd.DataFrame,
    source_name: str,
    target_sensor: str,
    window_size: int = 20,
    window_stride: int = 5,
    horizon: int = 12,
    contamination: float = 0.08,
    scaling_method: str = 'standard',
    model_dir: str | Path | None = None,
) -> AnalysisArtifacts:
    prepared = preprocess_timeseries(
        frame=frame,
        config=PreprocessConfig(
            window_size=window_size,
            window_stride=window_stride,
            scaling_method=scaling_method,
        ),
    )

    if target_sensor not in prepared.sensor_columns:
        raise ValueError(
            f"Параметр '{target_sensor}' отсутствует среди доступных столбцов: "
            + ', '.join(prepared.sensor_columns)
        )

    detection = detect_instability(
        prepared=prepared,
        config=DetectionConfig(contamination=contamination),
    )
    forecast = forecast_parameter(
        cleaned_df=prepared.cleaned_df,
        windows_df=detection.windows,
        target_sensor=target_sensor,
        config=ForecastConfig(target_sensor=target_sensor, horizon=horizon),
        model_dir=model_dir or ensure_directory(ROOT_DIR / 'models'),
        timestamp_column=prepared.timestamp_column,
    )

    explanations = build_interval_explanations(prepared=prepared, detection=detection)
    detection.intervals = detection.intervals.merge(explanations, on='interval_id', how='left')
    summary = build_summary_metrics(
        cleaned_df=prepared.cleaned_df,
        intervals_df=detection.intervals,
        windows_df=detection.windows,
        forecast_risk=forecast.risk_score,
    )
    conclusion = _build_conclusion(
        intervals_df=detection.intervals,
        forecast=forecast,
        target_sensor=target_sensor,
    )

    return AnalysisArtifacts(
        source_name=source_name,
        analysis_time=datetime.now(),
        raw_df=frame,
        prepared=prepared,
        detection=detection,
        forecast=forecast,
        explanations=explanations,
        summary=summary,
        conclusion=conclusion,
    )


def analysis_to_dict(analysis: AnalysisArtifacts) -> dict[str, Any]:
    return {
        'source_name': analysis.source_name,
        'analysis_time': analysis.analysis_time.isoformat(),
        'sensors': analysis.prepared.sensor_columns,
        'selected_sensor': analysis.forecast.target_sensor,
        'summary': make_json_safe(analysis.summary),
        'conclusion': analysis.conclusion,
        'cleaned_preview': dataframe_to_records(analysis.prepared.cleaned_df.head(20)),
        'windows': dataframe_to_records(analysis.detection.windows),
        'intervals': dataframe_to_records(analysis.detection.intervals),
        'forecast': dataframe_to_records(analysis.forecast.forecast_df),
        'forecast_meta': make_json_safe(
            {
                'risk_score': analysis.forecast.risk_score,
                'risk_level': analysis.forecast.risk_level,
                'metrics': analysis.forecast.metrics,
                'model_path': analysis.forecast.model_path,
                'normal_band': analysis.forecast.normal_band,
            }
        ),
    }