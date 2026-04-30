from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def infer_time_delta(timestamps: pd.Series) -> pd.Timedelta:
    ordered = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    diffs = ordered.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=1)

    mode = diffs.mode()
    if not mode.empty and pd.notna(mode.iloc[0]):
        return pd.to_timedelta(mode.iloc[0])
    return pd.to_timedelta(diffs.median())


def format_duration(value: pd.Timedelta | float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "0 мин"

    if not isinstance(value, pd.Timedelta):
        value = pd.to_timedelta(value)

    total_seconds = int(value.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} мин")
    if seconds and not parts:
        parts.append(f"{seconds} сек")
    return " ".join(parts) if parts else "0 мин"


def slugify(value: str) -> str:
    ascii_only = re.sub(r"[^\w\s-]", "", value, flags=re.ASCII).strip().lower()
    return re.sub(r"[-\s]+", "_", ascii_only) or "report"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return value


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe_frame = frame.copy()
    for column in safe_frame.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_frame[column]):
            safe_frame[column] = safe_frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_timedelta64_dtype(safe_frame[column]):
            safe_frame[column] = safe_frame[column].astype(str)
        elif safe_frame[column].dtype == "object":
            safe_frame[column] = safe_frame[column].map(make_json_safe)
    return safe_frame.to_dict(orient="records")