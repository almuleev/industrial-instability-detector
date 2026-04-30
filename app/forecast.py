from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from app.utils import ensure_directory, infer_time_delta


@dataclass
class ForecastConfig:
    target_sensor: str
    horizon: int = 12
    n_lags: int = 12
    n_estimators: int = 300
    random_state: int = 42


@dataclass
class ForecastResult:
    target_sensor: str
    forecast_df: pd.DataFrame
    risk_score: float
    risk_level: str
    metrics: dict[str, float]
    model_path: str
    normal_band: tuple[float, float]


def _safe_lag_count(series_length: int, desired_lags: int) -> int:
    adaptive_lags = min(desired_lags, max(3, series_length // 4))
    if series_length <= adaptive_lags + 5:
        adaptive_lags = max(2, series_length - 5)
    if adaptive_lags < 2:
        raise ValueError('Недостаточно наблюдений для построения прогноза.')
    return adaptive_lags


def _build_feature_row(history: list[float], n_lags: int) -> dict[str, float]:
    recent = history[-n_lags:]
    feature_row = {f'lag_{lag}': float(history[-lag]) for lag in range(1, n_lags + 1)}
    feature_row['rolling_mean_3'] = float(np.mean(recent[-3:]))
    feature_row['rolling_std_3'] = float(np.std(recent[-3:], ddof=0))
    feature_row['rolling_mean_5'] = float(np.mean(recent[-5:])) if len(recent) >= 5 else float(np.mean(recent))
    feature_row['rolling_std_5'] = float(np.std(recent[-5:], ddof=0)) if len(recent) >= 5 else float(np.std(recent, ddof=0))
    feature_row['diff_1'] = float(recent[-1] - recent[-2]) if len(recent) >= 2 else 0.0
    feature_row['diff_3'] = float(recent[-1] - recent[-4]) if len(recent) >= 4 else feature_row['diff_1']
    return feature_row


def _build_supervised_frame(series: pd.Series, n_lags: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    values = series.astype(float).tolist()
    for idx in range(n_lags, len(values)):
        feature_row = _build_feature_row(values[:idx], n_lags=n_lags)
        feature_row['target'] = float(values[idx])
        rows.append(feature_row)
    supervised = pd.DataFrame(rows)
    if supervised.empty:
        raise ValueError('Не удалось сформировать лаговые признаки для прогноза.')
    return supervised


def _series_stability_metrics(series: pd.Series) -> dict[str, float]:
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

    return {
        'amplitude_cv': round(float(amplitude_cv), 4),
        'diff_cv': round(float(diff_cv), 4),
        'spike_ratio': round(float(spike_ratio), 4),
    }


def _build_harmonic_design(positions: np.ndarray, period: float, harmonics: int = 1) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    safe_period = max(float(period), 1.0)
    columns = [np.ones_like(positions)]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * positions / safe_period
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns)


def _fit_harmonic_regression(
    values: np.ndarray,
    positions: np.ndarray,
    period: float,
    harmonics: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    design = _build_harmonic_design(positions=positions, period=period, harmonics=harmonics)
    coefficients, *_ = np.linalg.lstsq(design, values.astype(float), rcond=None)
    fitted = design @ coefficients
    return coefficients, fitted


def _fit_quality(values: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    residuals = values - fitted
    rmse = float(np.sqrt(mean_squared_error(values, fitted)))
    mae = float(mean_absolute_error(values, fitted))
    signal_std = float(np.std(values, ddof=0))
    residual_std = float(np.std(residuals, ddof=0))
    signal_var = float(np.var(values, ddof=0))
    if signal_var <= 1e-12:
        r2 = 1.0 if residual_std <= 1e-6 else 0.0
    else:
        r2 = 1.0 - float(np.sum((values - fitted) ** 2) / (len(values) * signal_var))
    return {
        'mae': mae,
        'rmse': rmse,
        'residual_std': residual_std,
        'residual_ratio': residual_std / (signal_std or 1.0),
        'r2': r2,
    }


def _search_best_period(values: np.ndarray, candidate_periods: np.ndarray) -> dict[str, float] | None:
    positions = np.arange(len(values), dtype=float)
    best_candidate: dict[str, float] | None = None

    for period in candidate_periods:
        coefficients, fitted = _fit_harmonic_regression(
            values=values,
            positions=positions,
            period=float(period),
            harmonics=1,
        )
        quality = _fit_quality(values=values, fitted=fitted)
        candidate = {
            'period': float(period),
            'mae': quality['mae'],
            'rmse': quality['rmse'],
            'residual_std': quality['residual_std'],
            'residual_ratio': quality['residual_ratio'],
            'r2': quality['r2'],
            'bias': float(coefficients[0]),
            'sin_coef': float(coefficients[1]),
            'cos_coef': float(coefficients[2]),
        }
        if best_candidate is None:
            best_candidate = candidate
            continue
        if candidate['r2'] > best_candidate['r2'] + 1e-6:
            best_candidate = candidate
            continue
        if abs(candidate['r2'] - best_candidate['r2']) <= 1e-6 and candidate['rmse'] < best_candidate['rmse']:
            best_candidate = candidate

    return best_candidate


def _assess_periodic_target(series: pd.Series) -> dict[str, float | bool]:
    series = pd.Series(series).astype(float).reset_index(drop=True)
    if len(series) < 60:
        return {
            'is_periodic': False,
            'period': 0.0,
            'r2': 0.0,
            'residual_ratio': 1.0,
            'cycles_covered': 0.0,
            'amplitude_cv': 0.0,
            'diff_cv': 0.0,
            'spike_ratio': 0.0,
        }

    search_series = series.iloc[-min(len(series), 1200):].reset_index(drop=True)
    values = search_series.to_numpy(dtype=float)
    min_period = max(12.0, len(values) / 25.0)
    max_period = min(len(values) / 1.5, 240.0)
    if max_period <= min_period + 1.0:
        return {
            'is_periodic': False,
            'period': 0.0,
            'r2': 0.0,
            'residual_ratio': 1.0,
            'cycles_covered': 0.0,
            **_series_stability_metrics(search_series),
        }

    coarse_count = int(min(360, max(140, round((max_period - min_period) * 3))))
    coarse_candidates = np.linspace(min_period, max_period, num=coarse_count)
    best_coarse = _search_best_period(values=values, candidate_periods=coarse_candidates)
    if best_coarse is None:
        return {
            'is_periodic': False,
            'period': 0.0,
            'r2': 0.0,
            'residual_ratio': 1.0,
            'cycles_covered': 0.0,
            **_series_stability_metrics(search_series),
        }

    coarse_step = float(coarse_candidates[1] - coarse_candidates[0]) if len(coarse_candidates) > 1 else 1.0
    refine_min = max(min_period, best_coarse['period'] - 3.0 * coarse_step)
    refine_max = min(max_period, best_coarse['period'] + 3.0 * coarse_step)
    refine_candidates = np.linspace(refine_min, refine_max, num=180)
    best_candidate = _search_best_period(values=values, candidate_periods=refine_candidates) or best_coarse

    stability = _series_stability_metrics(search_series)
    cycles_covered = float(len(values) / max(best_candidate['period'], 1.0))
    strong_fit = (
        best_candidate['r2'] >= 0.96
        and best_candidate['residual_ratio'] <= 0.18
        and cycles_covered >= 1.75
    )
    near_ideal_fit = (
        best_candidate['r2'] >= 0.995
        and best_candidate['residual_ratio'] <= 0.08
        and cycles_covered >= 1.45
    )
    is_periodic = (
        (strong_fit or near_ideal_fit)
        and stability['amplitude_cv'] <= 0.6
        and stability['diff_cv'] <= 0.6
        and stability['spike_ratio'] <= 4.5
    )

    return {
        'is_periodic': is_periodic,
        'period': round(float(best_candidate['period']), 4),
        'r2': round(float(best_candidate['r2']), 4),
        'residual_ratio': round(float(best_candidate['residual_ratio']), 4),
        'cycles_covered': round(cycles_covered, 4),
        **stability,
    }


def _derive_normal_band(
    cleaned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    target_sensor: str,
    timestamp_column: str = 'timestamp',
) -> tuple[float, float]:
    if windows_df.empty:
        series = cleaned_df[target_sensor]
        return float(series.quantile(0.1)), float(series.quantile(0.9))

    normal_windows = windows_df.loc[windows_df['state'] == 'normal']
    if normal_windows.empty:
        series = cleaned_df[target_sensor]
        return float(series.quantile(0.1)), float(series.quantile(0.9))

    mask = pd.Series(False, index=cleaned_df.index)
    for row in normal_windows.itertuples(index=False):
        mask = mask | cleaned_df[timestamp_column].between(row.window_start, row.window_end)

    normal_series = cleaned_df.loc[mask, target_sensor]
    if normal_series.empty:
        normal_series = cleaned_df[target_sensor]
    return float(normal_series.quantile(0.1)), float(normal_series.quantile(0.9))


def _calculate_risk_score(
    cleaned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    target_sensor: str,
    forecast_values: list[float],
    normal_band: tuple[float, float],
) -> tuple[float, str]:
    series = cleaned_df[target_sensor].astype(float)
    lower_bound, upper_bound = normal_band
    band_width = max(upper_bound - lower_bound, float(series.std(ddof=0)) or 1.0)

    max_excess = max(0.0, max(forecast_values) - upper_bound, lower_bound - min(forecast_values))
    level_risk = min(max_excess / band_width, 1.0)

    forecast_volatility = float(np.std(forecast_values, ddof=0)) / (float(series.std(ddof=0)) or 1.0)
    trend_shift = abs(forecast_values[-1] - float(series.iloc[-1])) / (float(series.std(ddof=0)) or 1.0)
    recent_context = float(windows_df['anomaly_score'].tail(3).mean()) if not windows_df.empty else 0.0

    risk_score = np.clip(
        0.5 * level_risk + 0.2 * min(forecast_volatility, 1.0) + 0.15 * min(trend_shift, 1.0) + 0.15 * recent_context,
        0.0,
        1.0,
    )
    if risk_score < 0.35:
        risk_level = 'low'
    elif risk_score < 0.65:
        risk_level = 'medium'
    else:
        risk_level = 'high'
    return round(float(risk_score), 4), risk_level


def _build_forecast_frame(
    cleaned_df: pd.DataFrame,
    forecast_values: list[float],
    residual_std: float,
    timestamp_column: str,
) -> pd.DataFrame:
    delta = infer_time_delta(cleaned_df[timestamp_column])
    future_index = [cleaned_df[timestamp_column].iloc[-1] + delta * step for step in range(1, len(forecast_values) + 1)]
    return pd.DataFrame(
        {
            timestamp_column: future_index,
            'forecast': np.round(forecast_values, 4),
            'lower_bound': np.round(np.array(forecast_values) - 1.96 * residual_std, 4),
            'upper_bound': np.round(np.array(forecast_values) + 1.96 * residual_std, 4),
        }
    )


def _save_forecast_artifact(model_dir: str | Path, target_sensor: str, payload: dict[str, object]) -> str:
    models_path = ensure_directory(model_dir)
    model_path = models_path / f'{target_sensor}_forecast_model.joblib'
    joblib.dump(payload, model_path)
    return str(model_path)


def _forecast_periodic_series(
    cleaned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    target_sensor: str,
    profile: dict[str, float | bool],
    config: ForecastConfig,
    model_dir: str | Path,
    timestamp_column: str,
) -> ForecastResult:
    values = cleaned_df[target_sensor].astype(float).to_numpy()
    selected_period = float(profile['period'])
    harmonics = 1

    fit_span = int(min(len(values), max(round(selected_period * 2.5), 60)))
    fit_values = values[-fit_span:]
    fit_positions = np.arange(len(values) - fit_span, len(values), dtype=float)

    validation_size = min(max(config.horizon, 4), max(4, fit_span // 6))
    metrics: dict[str, float]
    residual_std: float
    if fit_span > validation_size + max(int(round(selected_period * 0.75)), 12):
        train_values = fit_values[:-validation_size]
        train_positions = fit_positions[:-validation_size]
        validation_values = fit_values[-validation_size:]
        validation_positions = fit_positions[-validation_size:]

        validation_coefficients, _ = _fit_harmonic_regression(
            values=train_values,
            positions=train_positions,
            period=selected_period,
            harmonics=harmonics,
        )
        validation_design = _build_harmonic_design(
            positions=validation_positions,
            period=selected_period,
            harmonics=harmonics,
        )
        validation_predictions = validation_design @ validation_coefficients
        validation_residuals = validation_values - validation_predictions
        residual_std = float(np.std(validation_residuals, ddof=0)) or float(np.std(fit_values, ddof=0)) * 0.05 or 1.0
        metrics = {
            'mae': round(float(mean_absolute_error(validation_values, validation_predictions)), 4),
            'rmse': round(float(np.sqrt(mean_squared_error(validation_values, validation_predictions))), 4),
        }
    else:
        coefficients_tmp, fitted_tmp = _fit_harmonic_regression(
            values=fit_values,
            positions=fit_positions,
            period=selected_period,
            harmonics=harmonics,
        )
        quality = _fit_quality(values=fit_values, fitted=fitted_tmp)
        residual_std = float(quality['residual_std']) or float(np.std(fit_values, ddof=0)) * 0.05 or 1.0
        metrics = {
            'mae': round(float(quality['mae']), 4),
            'rmse': round(float(quality['rmse']), 4),
        }

    coefficients, fitted = _fit_harmonic_regression(
        values=fit_values,
        positions=fit_positions,
        period=selected_period,
        harmonics=harmonics,
    )
    future_positions = np.arange(len(values), len(values) + config.horizon, dtype=float)
    forecast_design = _build_harmonic_design(
        positions=future_positions,
        period=selected_period,
        harmonics=harmonics,
    )
    forecast_values = forecast_design @ coefficients

    forecast_df = _build_forecast_frame(
        cleaned_df=cleaned_df,
        forecast_values=list(map(float, forecast_values)),
        residual_std=residual_std,
        timestamp_column=timestamp_column,
    )
    normal_band = _derive_normal_band(
        cleaned_df=cleaned_df,
        windows_df=windows_df,
        target_sensor=target_sensor,
        timestamp_column=timestamp_column,
    )
    risk_score, risk_level = _calculate_risk_score(
        cleaned_df=cleaned_df,
        windows_df=windows_df,
        target_sensor=target_sensor,
        forecast_values=list(map(float, forecast_values)),
        normal_band=normal_band,
    )
    model_path = _save_forecast_artifact(
        model_dir=model_dir,
        target_sensor=target_sensor,
        payload={
            'model_type': 'harmonic_regression',
            'target_sensor': target_sensor,
            'period': selected_period,
            'harmonics': harmonics,
            'coefficients': coefficients.tolist(),
            'fit_span': fit_span,
            'periodic_profile': dict(profile),
        },
    )
    return ForecastResult(
        target_sensor=target_sensor,
        forecast_df=forecast_df,
        risk_score=risk_score,
        risk_level=risk_level,
        metrics=metrics,
        model_path=model_path,
        normal_band=(round(normal_band[0], 4), round(normal_band[1], 4)),
    )


def _build_direct_supervised_frame(series: pd.Series, n_lags: int, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    values = series.astype(float).tolist()
    for idx in range(n_lags, len(values) - horizon + 1):
        feature_row = _build_feature_row(values[:idx], n_lags=n_lags)
        for step in range(1, horizon + 1):
            feature_row[f'target_{step}'] = float(values[idx + step - 1])
        rows.append(feature_row)
    return pd.DataFrame(rows)



def _recursive_forest_forecast(
    history_values: list[float],
    n_lags: int,
    horizon: int,
    model: RandomForestRegressor,
) -> list[float]:
    history = history_values.copy()
    forecast_values: list[float] = []
    for _ in range(horizon):
        features = pd.DataFrame([_build_feature_row(history, n_lags=n_lags)])
        next_value = float(model.predict(features)[0])
        forecast_values.append(next_value)
        history.append(next_value)
    return forecast_values



def _candidate_lag_counts(series_length: int, desired_lags: int, horizon: int) -> list[int]:
    max_candidate = min(max(18, series_length // 3), 60)
    candidates = [desired_lags, 18, 24, 36, 48]
    normalized: list[int] = []
    for lag in candidates:
        bounded_lag = min(max(6, int(lag)), max_candidate)
        if series_length > bounded_lag + horizon + 4 and bounded_lag not in normalized:
            normalized.append(bounded_lag)
    if not normalized:
        normalized.append(_safe_lag_count(series_length, desired_lags))
    return sorted(normalized)



def _build_direct_candidates(
    config: ForecastConfig,
    n_estimators: int | None = None,
) -> dict[str, MultiOutputRegressor]:
    tree_count = n_estimators or config.n_estimators
    return {
        'random_forest_direct': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=tree_count,
                random_state=config.random_state,
                min_samples_leaf=2,
            )
        ),
        'extra_trees_direct': MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=tree_count,
                random_state=config.random_state,
                min_samples_leaf=2,
            )
        ),
    }



def _evaluate_tree_candidates(
    series: pd.Series,
    n_lags: int,
    config: ForecastConfig,
) -> dict[str, object]:
    values = series.astype(float).to_numpy()
    horizon = config.horizon
    if len(values) <= max(n_lags + horizon + 4, 24):
        raise ValueError('Слишком мало данных для обучения устойчивого прогноза.')

    train_values = values[:-horizon]
    validation_values = values[-horizon:]
    evaluation_tree_count = min(config.n_estimators, 120)
    results: list[dict[str, object]] = []

    for candidate_lag in _candidate_lag_counts(len(values), n_lags, horizon):
        direct_frame = _build_direct_supervised_frame(pd.Series(train_values), n_lags=candidate_lag, horizon=horizon)
        target_columns = [f'target_{step}' for step in range(1, horizon + 1)]
        feature_row = pd.DataFrame([_build_feature_row(train_values.tolist(), n_lags=candidate_lag)])
        if not direct_frame.empty:
            x_direct = direct_frame.drop(columns=target_columns)
            y_direct = direct_frame[target_columns]
            for model_type, estimator in _build_direct_candidates(config, n_estimators=evaluation_tree_count).items():
                estimator.fit(x_direct, y_direct)
                prediction = estimator.predict(feature_row)[0].astype(float)
                rmse = float(np.sqrt(mean_squared_error(validation_values, prediction)))
                results.append(
                    {
                        'strategy': 'direct',
                        'model_type': model_type,
                        'n_lags': candidate_lag,
                        'rmse': rmse,
                        'mae': float(mean_absolute_error(validation_values, prediction)),
                        'residual_std': float(np.std(validation_values - prediction, ddof=0)) or rmse or 1.0,
                        'validation_predictions': prediction.tolist(),
                        'feature_columns': x_direct.columns.tolist(),
                        'target_columns': target_columns,
                    }
                )

        supervised = _build_supervised_frame(series=pd.Series(train_values), n_lags=candidate_lag)
        if not supervised.empty:
            x_train = supervised.drop(columns=['target'])
            y_train = supervised['target']
            recursive_model = RandomForestRegressor(
                n_estimators=evaluation_tree_count,
                random_state=config.random_state,
                min_samples_leaf=2,
            )
            recursive_model.fit(x_train, y_train)
            recursive_prediction = _recursive_forest_forecast(
                history_values=train_values.tolist(),
                n_lags=candidate_lag,
                horizon=horizon,
                model=recursive_model,
            )
            recursive_array = np.asarray(recursive_prediction, dtype=float)
            rmse = float(np.sqrt(mean_squared_error(validation_values, recursive_array)))
            results.append(
                {
                    'strategy': 'recursive',
                    'model_type': 'random_forest_recursive',
                    'n_lags': candidate_lag,
                    'rmse': rmse,
                    'mae': float(mean_absolute_error(validation_values, recursive_array)),
                    'residual_std': float(np.std(validation_values - recursive_array, ddof=0)) or rmse or 1.0,
                    'validation_predictions': recursive_prediction,
                    'feature_columns': x_train.columns.tolist(),
                    'target_columns': ['target'],
                }
            )

    if not results:
        raise ValueError('Не удалось сформировать кандидаты для прогноза на основе истории.')
    return min(results, key=lambda item: (float(item['rmse']), float(item['mae'])))


def _forecast_with_tree_ensemble(
    cleaned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    target_sensor: str,
    config: ForecastConfig,
    model_dir: str | Path,
    timestamp_column: str,
) -> ForecastResult:
    series = cleaned_df[target_sensor].astype(float)
    base_lags = _safe_lag_count(len(series), config.n_lags)
    evaluation = _evaluate_tree_candidates(series=series, n_lags=base_lags, config=config)
    selected_n_lags = int(evaluation['n_lags'])

    if evaluation['strategy'] == 'direct':
        target_columns = list(evaluation['target_columns'])
        direct_frame = _build_direct_supervised_frame(series=series, n_lags=selected_n_lags, horizon=config.horizon)
        if direct_frame.empty:
            raise ValueError('Недостаточно данных для прямого многогоризонтного прогноза.')
        x_full = direct_frame.drop(columns=target_columns)
        y_full = direct_frame[target_columns]
        estimator = _build_direct_candidates(config)[str(evaluation['model_type'])]
        estimator.fit(x_full, y_full)
        forecast_values = estimator.predict(
            pd.DataFrame([_build_feature_row(series.tolist(), n_lags=selected_n_lags)])
        )[0].astype(float).tolist()
        model_object = estimator
        feature_columns = x_full.columns.tolist()
        target_columns_payload = target_columns
    else:
        supervised = _build_supervised_frame(series=series, n_lags=selected_n_lags)
        x_full = supervised.drop(columns=['target'])
        y_full = supervised['target']
        estimator = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            min_samples_leaf=2,
        )
        estimator.fit(x_full, y_full)
        forecast_values = _recursive_forest_forecast(
            history_values=series.tolist(),
            n_lags=selected_n_lags,
            horizon=config.horizon,
            model=estimator,
        )
        model_object = estimator
        feature_columns = x_full.columns.tolist()
        target_columns_payload = ['target']

    metrics = {
        'mae': round(float(evaluation['mae']), 4),
        'rmse': round(float(evaluation['rmse']), 4),
    }
    residual_std = float(evaluation['residual_std']) or float(evaluation['rmse']) or 1.0
    forecast_df = _build_forecast_frame(
        cleaned_df=cleaned_df,
        forecast_values=[float(value) for value in forecast_values],
        residual_std=residual_std,
        timestamp_column=timestamp_column,
    )
    normal_band = _derive_normal_band(
        cleaned_df=cleaned_df,
        windows_df=windows_df,
        target_sensor=target_sensor,
        timestamp_column=timestamp_column,
    )
    risk_score, risk_level = _calculate_risk_score(
        cleaned_df=cleaned_df,
        windows_df=windows_df,
        target_sensor=target_sensor,
        forecast_values=[float(value) for value in forecast_values],
        normal_band=normal_band,
    )
    model_path = _save_forecast_artifact(
        model_dir=model_dir,
        target_sensor=target_sensor,
        payload={
            'model_type': str(evaluation['model_type']),
            'target_sensor': target_sensor,
            'n_lags': selected_n_lags,
            'horizon': config.horizon,
            'feature_columns': feature_columns,
            'target_columns': target_columns_payload,
            'validation_metrics': metrics,
            'model': model_object,
        },
    )
    return ForecastResult(
        target_sensor=target_sensor,
        forecast_df=forecast_df,
        risk_score=risk_score,
        risk_level=risk_level,
        metrics=metrics,
        model_path=model_path,
        normal_band=(round(normal_band[0], 4), round(normal_band[1], 4)),
    )


def forecast_parameter(
    cleaned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    target_sensor: str,
    config: ForecastConfig | None = None,
    model_dir: str | Path = 'models',
    timestamp_column: str = 'timestamp',
) -> ForecastResult:
    if target_sensor not in cleaned_df.columns:
        raise ValueError(f"Параметр '{target_sensor}' отсутствует в наборе данных.")

    config = config or ForecastConfig(target_sensor=target_sensor)
    series = cleaned_df[target_sensor].astype(float)
    periodic_profile = _assess_periodic_target(series)
    if periodic_profile['is_periodic']:
        return _forecast_periodic_series(
            cleaned_df=cleaned_df,
            windows_df=windows_df,
            target_sensor=target_sensor,
            profile=periodic_profile,
            config=config,
            model_dir=model_dir,
            timestamp_column=timestamp_column,
        )

    return _forecast_with_tree_ensemble(
        cleaned_df=cleaned_df,
        windows_df=windows_df,
        target_sensor=target_sensor,
        config=config,
        model_dir=model_dir,
        timestamp_column=timestamp_column,
    )
