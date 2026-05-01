from __future__ import annotations

from pathlib import Path

from app.loader import load_timeseries
from app.pipeline import run_analysis


def test_pipeline_smoke_demo_dataset(tmp_path: Path) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    frame = load_timeseries(root_dir / "data" / "demo_timeseries.csv")

    analysis = run_analysis(
        frame=frame,
        source_name="demo_timeseries.csv",
        target_sensor="sensor_2",
        horizon=6,
        window_size=20,
        window_stride=5,
        model_dir=tmp_path,
    )

    assert analysis.summary["records_analyzed"] > 0
    assert analysis.forecast.forecast_df.shape[0] == 6
    assert analysis.forecast.risk_level in {"low", "medium", "high"}
    assert analysis.forecast.model_path
