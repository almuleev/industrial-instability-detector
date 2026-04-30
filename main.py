from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app.loader import load_timeseries
from app.pipeline import run_analysis
from app.report import generate_reports


ROOT_DIR = Path(__file__).resolve().parent


def launch_ui(host: str, port: int) -> None:
    subprocess.run(
        [
            sys.executable,
            '-m',
            'streamlit',
            'run',
            str(ROOT_DIR / 'app' / 'ui.py'),
            '--server.address',
            host,
            '--server.port',
            str(port),
        ],
        check=True,
    )


def launch_api(host: str, port: int) -> None:
    subprocess.run(
        [
            sys.executable,
            '-m',
            'uvicorn',
            'app.api:app',
            '--host',
            host,
            '--port',
            str(port),
        ],
        check=True,
    )


def run_cli_analysis(file_path: str, sensor: str, horizon: int, window_size: int, window_stride: int) -> None:
    frame = load_timeseries(file_path)
    analysis = run_analysis(
        frame=frame,
        source_name=Path(file_path).name,
        target_sensor=sensor,
        horizon=horizon,
        window_size=window_size,
        window_stride=window_stride,
    )
    generated = generate_reports(analysis=analysis, output_dir=ROOT_DIR / 'reports')
    print('Анализ завершён.')
    print(f"Интервалов найдено: {analysis.summary['unstable_intervals']}")
    print(f"Риск прогноза: {analysis.forecast.risk_score} ({analysis.forecast.risk_level})")
    print('Сформированные отчёты:')
    for fmt, path in generated.items():
        print(f'- {fmt}: {path}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='MVP системы анализа технологических временных рядов.'
    )
    subparsers = parser.add_subparsers(dest='command')

    ui_parser = subparsers.add_parser('ui', help='Запуск Streamlit-интерфейса')
    ui_parser.add_argument('--host', default='127.0.0.1')
    ui_parser.add_argument('--port', type=int, default=8501)

    api_parser = subparsers.add_parser('api', help='Запуск FastAPI-сервера')
    api_parser.add_argument('--host', default='127.0.0.1')
    api_parser.add_argument('--port', type=int, default=8000)

    analyze_parser = subparsers.add_parser('analyze', help='CLI-анализ файла')
    analyze_parser.add_argument('--file', required=True, help='Путь к CSV/XLSX файлу')
    analyze_parser.add_argument('--sensor', required=True, help='Целевой параметр для прогноза')
    analyze_parser.add_argument('--horizon', type=int, default=12)
    analyze_parser.add_argument('--window-size', type=int, default=20)
    analyze_parser.add_argument('--window-stride', type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'ui':
        launch_ui(host=args.host, port=args.port)
        return
    if args.command == 'api':
        launch_api(host=args.host, port=args.port)
        return
    if args.command == 'analyze':
        run_cli_analysis(
            file_path=args.file,
            sensor=args.sensor,
            horizon=args.horizon,
            window_size=args.window_size,
            window_stride=args.window_stride,
        )
        return

    parser.print_help()


if __name__ == '__main__':
    main()