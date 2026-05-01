from __future__ import annotations

import pandas as pd
import pytest

from app.loader import DataValidationError, validate_timeseries_dataframe


def test_validate_timeseries_accepts_numeric_strings() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00", "2026-01-01 00:02:00"],
            "sensor_1": ["10,5", "11.0", "12"],
            "sensor_2": [1, 2, 3],
        }
    )

    validated = validate_timeseries_dataframe(frame)
    assert list(validated.columns) == ["timestamp", "sensor_1", "sensor_2"]
    assert validated["sensor_1"].dtype.kind in {"f", "i"}
    assert validated["sensor_2"].dtype.kind in {"f", "i"}


def test_validate_timeseries_rejects_non_numeric_column() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00", "2026-01-01 00:02:00"],
            "sensor_1": ["ok", "bad", "oops"],
        }
    )

    with pytest.raises(DataValidationError):
        validate_timeseries_dataframe(frame)
