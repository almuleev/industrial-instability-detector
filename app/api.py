from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

if __package__ in {None, ''}:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

from app.loader import DataValidationError, load_timeseries
from app.pipeline import AnalysisArtifacts, analysis_to_dict, run_analysis
from app.report import generate_reports
from app.utils import dataframe_to_records


app = FastAPI(
    title='Industrial Time-Series Instability MVP',
    version='0.1.0',
    description='Прототип системы анализа технологических временных рядов.',
)

SESSIONS: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    session_id: str
    target_sensor: str
    window_size: int = Field(default=20, ge=4, le=500)
    window_stride: int = Field(default=5, ge=1, le=200)
    horizon: int = Field(default=12, ge=1, le=100)
    contamination: float = Field(default=0.08, ge=0.01, le=0.45)
    scaling_method: str = Field(default='standard')


def _get_session(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Сессия не найдена. Сначала загрузите файл.')
    return session


def _get_analysis(session_id: str) -> AnalysisArtifacts:
    session = _get_session(session_id)
    analysis = session.get('analysis')
    if analysis is None:
        raise HTTPException(status_code=400, detail='Анализ для этой сессии ещё не запускался.')
    return analysis


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/upload')
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        content = await file.read()
        frame = load_timeseries(content, file.filename)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = str(uuid4())
    sensor_columns = [column for column in frame.columns if column != 'timestamp']
    preview = dataframe_to_records(frame.head(10))
    SESSIONS[session_id] = {
        'source_name': file.filename,
        'frame': frame,
    }
    return {
        'message': 'Файл успешно загружен.',
        'session_id': session_id,
        'source_name': file.filename,
        'rows': int(len(frame)),
        'columns': list(frame.columns),
        'sensors': sensor_columns,
        'preview': preview,
    }


@app.post('/analyze')
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    session = _get_session(request.session_id)
    try:
        analysis = run_analysis(
            frame=session['frame'],
            source_name=session['source_name'],
            target_sensor=request.target_sensor,
            window_size=request.window_size,
            window_stride=request.window_stride,
            horizon=request.horizon,
            contamination=request.contamination,
            scaling_method=request.scaling_method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session['analysis'] = analysis
    payload = analysis_to_dict(analysis)
    return {
        'message': 'Анализ успешно выполнен.',
        'session_id': request.session_id,
        'summary': payload['summary'],
        'intervals': payload['intervals'],
        'conclusion': payload['conclusion'],
    }


@app.get('/results')
def results(session_id: str = Query(..., description='Идентификатор сессии')) -> dict[str, Any]:
    analysis = _get_analysis(session_id)
    payload = analysis_to_dict(analysis)
    return {'session_id': session_id, **payload}


@app.get('/forecast')
def forecast(session_id: str = Query(..., description='Идентификатор сессии')) -> dict[str, Any]:
    analysis = _get_analysis(session_id)
    payload = analysis_to_dict(analysis)
    return {
        'session_id': session_id,
        'selected_sensor': payload['selected_sensor'],
        'forecast': payload['forecast'],
        'forecast_meta': payload['forecast_meta'],
    }


@app.get('/report')
def report(
    session_id: str = Query(..., description='Идентификатор сессии'),
    format: str = Query(default='html'),
    download: bool = Query(default=False),
) -> Any:
    analysis = _get_analysis(session_id)
    report_format = format.lower()
    if report_format not in {'html', 'txt'}:
        raise HTTPException(status_code=400, detail='Поддерживаются только форматы HTML и TXT.')

    try:
        report_path = generate_reports(analysis=analysis, formats=[report_format])[report_format]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if download:
        media_type = 'text/html' if report_format == 'html' else 'text/plain'
        return FileResponse(path=report_path, media_type=media_type, filename=report_path.name)

    return {
        'session_id': session_id,
        'report_format': report_format,
        'report_path': str(report_path),
        'download_url': f'/report?session_id={session_id}&format={report_format}&download=true',
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app.api:app', host='127.0.0.1', port=8000, reload=False)