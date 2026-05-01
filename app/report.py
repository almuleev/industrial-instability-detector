from __future__ import annotations

from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.pipeline import AnalysisArtifacts
from app.utils import ensure_directory, format_duration, slugify


def _build_context(analysis: AnalysisArtifacts) -> dict[str, object]:
    intervals = []
    for row in analysis.detection.intervals.itertuples(index=False):
        intervals.append(
            {
                'interval_id': row.interval_id,
                'start': row.start.strftime('%Y-%m-%d %H:%M:%S'),
                'end': row.end.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': format_duration(row.duration),
                'average_score': row.average_score,
                'max_score': row.max_score,
                'peak_state': row.peak_state,
                'main_contributor': row.main_contributor,
                'explanation': getattr(row, 'text', '') or 'Явное текстовое объяснение не сформировано.',
            }
        )

    forecast_rows = []
    for row in analysis.forecast.forecast_df.itertuples(index=False):
        forecast_rows.append(
            {
                'timestamp': row.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'forecast': row.forecast,
                'lower_bound': row.lower_bound,
                'upper_bound': row.upper_bound,
            }
        )

    return {
        'analysis_time': analysis.analysis_time.strftime('%Y-%m-%d %H:%M:%S'),
        'source_name': analysis.source_name,
        'record_count': len(analysis.prepared.cleaned_df),
        'selected_sensor': analysis.forecast.target_sensor,
        'summary': analysis.summary,
        'forecast_risk': analysis.forecast.risk_score,
        'forecast_risk_level': analysis.forecast.risk_level,
        'forecast_metrics': analysis.forecast.metrics,
        'forecast_rows': forecast_rows,
        'intervals': intervals,
        'conclusion': analysis.conclusion,
    }


def generate_html_report(analysis: AnalysisArtifacts, output_dir: str | Path = 'reports') -> Path:
    reports_dir = ensure_directory(output_dir)
    template_dir = Path(__file__).resolve().parent / 'templates'
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml']),
    )
    template = environment.get_template('report_template.html')
    context = _build_context(analysis)

    file_name = f"{slugify(Path(analysis.source_name).stem)}_analysis_report.html"
    report_path = reports_dir / file_name
    report_path.write_text(template.render(**context), encoding='utf-8')
    return report_path


def generate_text_report(analysis: AnalysisArtifacts, output_dir: str | Path = 'reports') -> Path:
    reports_dir = ensure_directory(output_dir)
    context = _build_context(analysis)

    lines = [
        'Отчёт по анализу временных рядов',
        f"Дата анализа: {context['analysis_time']}",
        f"Исходный файл: {context['source_name']}",
        f"Количество записей: {context['record_count']}",
        f"Выбранный параметр прогноза: {context['selected_sensor']}",
        '',
        'Summary-метрики:',
    ]
    for key, value in context['summary'].items():
        lines.append(f'- {key}: {value}')

    lines.extend(
        [
            '',
            f"Риск по прогнозу: {context['forecast_risk']} ({context['forecast_risk_level']})",
            f"Метрики прогноза: {context['forecast_metrics']}",
            '',
            'Найденные интервалы нестабильности:',
        ]
    )

    intervals = context['intervals']
    if intervals:
        for interval in intervals:
            lines.extend(
                [
                    f"- Интервал #{interval['interval_id']}: {interval['start']} - {interval['end']}",
                    f"  Длительность: {interval['duration']}",
                    f"  Средний score: {interval['average_score']}",
                    f"  Главный параметр-вкладчик: {interval['main_contributor']}",
                    f"  Объяснение: {interval['explanation']}",
                ]
            )
    else:
        lines.append('- Явных нестабильных интервалов не обнаружено.')

    lines.extend(['', 'Итоговый вывод:', str(context['conclusion'])])

    file_name = f"{slugify(Path(analysis.source_name).stem)}_analysis_report.txt"
    report_path = reports_dir / file_name
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path


def generate_reports(
    analysis: AnalysisArtifacts,
    output_dir: str | Path = 'reports',
    formats: Iterable[str] = ('html', 'txt'),
) -> dict[str, Path]:
    generated: dict[str, Path] = {}
    requested = {fmt.lower() for fmt in formats}
    for fmt in requested:
        if fmt == 'html':
            generated[fmt] = generate_html_report(analysis=analysis, output_dir=output_dir)
        elif fmt == 'txt':
            generated[fmt] = generate_text_report(analysis=analysis, output_dir=output_dir)
        else:
            raise ValueError('Поддерживаются только форматы отчётов HTML и TXT.')
    return generated