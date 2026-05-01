from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.utils import format_duration


def create_timeseries_figure(
    cleaned_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    sensor: str,
    timestamp_column: str = 'timestamp',
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=cleaned_df[timestamp_column],
            y=cleaned_df[sensor],
            mode='lines',
            name=sensor,
            line={'color': '#0f766e', 'width': 2},
        )
    )

    for interval in intervals_df.itertuples(index=False):
        color = 'rgba(239, 68, 68, 0.18)' if interval.peak_state == 'unstable' else 'rgba(245, 158, 11, 0.18)'
        figure.add_vrect(
            x0=interval.start,
            x1=interval.end,
            fillcolor=color,
            line_width=0,
            annotation_text=f'{interval.peak_state} #{interval.interval_id}',
            annotation_position='top left',
        )

    figure.update_layout(
        title=f'Временной ряд параметра {sensor}',
        template='plotly_white',
        xaxis_title='Время',
        yaxis_title='Значение',
        legend_title='Серия',
        margin={'l': 30, 'r': 20, 't': 50, 'b': 30},
        hovermode='x unified',
    )
    return figure


def create_forecast_figure(
    cleaned_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sensor: str,
    normal_band: tuple[float, float],
    timestamp_column: str = 'timestamp',
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=cleaned_df[timestamp_column],
            y=cleaned_df[sensor],
            mode='lines',
            name='История',
            line={'color': '#1d4ed8', 'width': 2},
        )
    )

    forecast_x = [cleaned_df[timestamp_column].iloc[-1]] + list(forecast_df[timestamp_column])
    forecast_y = [cleaned_df[sensor].iloc[-1]] + list(forecast_df['forecast'])
    figure.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines+markers',
            name='Прогноз',
            line={'color': '#dc2626', 'width': 2, 'dash': 'dash'},
        )
    )

    figure.add_trace(
        go.Scatter(
            x=list(forecast_df[timestamp_column]) + list(forecast_df[timestamp_column][::-1]),
            y=list(forecast_df['upper_bound']) + list(forecast_df['lower_bound'][::-1]),
            fill='toself',
            fillcolor='rgba(220, 38, 38, 0.12)',
            line={'color': 'rgba(0,0,0,0)'},
            hoverinfo='skip',
            showlegend=True,
            name='Доверительный коридор',
        )
    )
    figure.add_hrect(
        y0=normal_band[0],
        y1=normal_band[1],
        fillcolor='rgba(16, 185, 129, 0.08)',
        line_width=0,
        annotation_text='Условно нормальный диапазон',
        annotation_position='bottom right',
    )
    figure.update_layout(
        title=f'Краткосрочный прогноз для {sensor}',
        template='plotly_white',
        xaxis_title='Время',
        yaxis_title='Значение',
        legend_title='Серия',
        margin={'l': 30, 'r': 20, 't': 50, 'b': 30},
        hovermode='x unified',
    )
    return figure


def build_summary_metrics(
    cleaned_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    forecast_risk: float,
) -> dict[str, str | int | float]:
    total_duration = intervals_df['duration'].sum() if not intervals_df.empty else pd.Timedelta(0)
    max_score = float(windows_df['anomaly_score'].max()) if not windows_df.empty else 0.0
    return {
        'records_analyzed': int(len(cleaned_df)),
        'unstable_intervals': int(len(intervals_df)),
        'total_unstable_duration': format_duration(total_duration),
        'max_anomaly_score': round(max_score, 4),
        'forecast_risk': round(float(forecast_risk), 4),
    }