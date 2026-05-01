from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ''}:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

from app.loader import DataValidationError, load_timeseries
from app.pipeline import AnalysisArtifacts, run_analysis
from app.report import generate_reports
from app.visualize import create_forecast_figure, create_timeseries_figure


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
DEMO_PATH = DATA_DIR / 'demo_timeseries.csv'


def _load_demo_dataset() -> tuple[str, object]:
    return DEMO_PATH.name, load_timeseries(DEMO_PATH)


def _load_uploaded_dataset(uploaded_file) -> tuple[str, object]:
    file_bytes = uploaded_file.getvalue()
    return uploaded_file.name, load_timeseries(file_bytes, uploaded_file.name)


def _list_local_datasets() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(path for path in DATA_DIR.iterdir() if path.suffix.lower() in {'.csv', '.xlsx'})


def _load_local_dataset(path: Path) -> tuple[str, object]:
    return path.name, load_timeseries(path)


def _render_metrics(analysis: AnalysisArtifacts) -> None:
    metrics = analysis.summary
    columns = st.columns(5)
    columns[0].metric('Записей', metrics['records_analyzed'])
    columns[1].metric('Интервалов', metrics['unstable_intervals'])
    columns[2].metric('Суммарная длительность', metrics['total_unstable_duration'])
    columns[3].metric('Макс. score', metrics['max_anomaly_score'])
    columns[4].metric('Риск прогноза', f"{metrics['forecast_risk']:.2f}")


def _render_explanations(analysis: AnalysisArtifacts) -> None:
    st.subheader('Объяснение нестабильных участков')
    intervals = analysis.detection.intervals
    if intervals.empty:
        st.success('По текущему набору данных выраженные нестабильные интервалы не обнаружены.')
        return

    for row in intervals.itertuples(index=False):
        with st.container(border=True):
            st.markdown(
                f"**Интервал #{row.interval_id}**: {row.start:%Y-%m-%d %H:%M:%S} - {row.end:%Y-%m-%d %H:%M:%S}"
            )
            st.write(getattr(row, 'text', 'Текстовое объяснение не сформировано.'))
            st.caption(
                f"Пиковое состояние: {row.peak_state} | "
                f"Средний score: {row.average_score} | "
                f"Главный вкладчик: {row.main_contributor}"
            )


def main() -> None:
    st.set_page_config(
        page_title='graph-anomaly-analyzer',
        layout='wide',
    )
    st.title('graph-anomaly-analyzer')
    st.caption(
        'MVP для поиска нестабильных режимов, краткосрочного прогноза и формирования отчёта.'
    )

    st.session_state.setdefault('source_name', None)
    st.session_state.setdefault('loaded_df', None)
    st.session_state.setdefault('analysis', None)
    st.session_state.setdefault('report_path', None)
    st.session_state.setdefault('upload_signature', None)

    with st.sidebar:
        st.header('Данные и параметры')
        uploaded_file = st.file_uploader('Загрузите CSV или XLSX', type=['csv', 'xlsx'])
        local_datasets = _list_local_datasets()
        local_dataset_names = [path.name for path in local_datasets]
        selected_local_name = st.selectbox(
            'Или выберите файл из data/',
            options=['-'] + local_dataset_names,
            index=0,
        )
        open_local = st.button('Открыть локальный файл')
        use_demo = st.button('Использовать demo-датасет')

        window_size = st.slider('Размер окна', min_value=4, max_value=48, value=20)
        window_stride = st.slider('Шаг окна', min_value=1, max_value=24, value=5)
        horizon = st.slider('Горизонт прогноза', min_value=3, max_value=24, value=12)
        contamination = st.slider('Доля аномальных окон', min_value=0.01, max_value=0.25, value=0.08)
        scaling_method = st.selectbox('Масштабирование', options=['standard', 'minmax'], index=0)

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        current_signature = (
            uploaded_file.name,
            uploaded_file.size,
            hashlib.sha1(uploaded_bytes).hexdigest(),
        )
        if st.session_state.upload_signature != current_signature:
            try:
                source_name, loaded_df = uploaded_file.name, load_timeseries(uploaded_bytes, uploaded_file.name)
                st.session_state.source_name = source_name
                st.session_state.loaded_df = loaded_df
                st.session_state.analysis = None
                st.session_state.report_path = None
                st.session_state.upload_signature = current_signature
            except DataValidationError as exc:
                st.error(str(exc))
                return

    if open_local:
        if selected_local_name == '-':
            st.warning('Выберите файл из папки data/.')
            return
        selected_path = next(path for path in local_datasets if path.name == selected_local_name)
        try:
            source_name, loaded_df = _load_local_dataset(selected_path)
            st.session_state.source_name = source_name
            st.session_state.loaded_df = loaded_df
            st.session_state.analysis = None
            st.session_state.report_path = None
            st.session_state.upload_signature = (
                'local',
                selected_path.name,
                selected_path.stat().st_mtime_ns,
                selected_path.stat().st_size,
            )
        except DataValidationError as exc:
            st.error(str(exc))
            return

    if use_demo:
        source_name, loaded_df = _load_demo_dataset()
        st.session_state.source_name = source_name
        st.session_state.loaded_df = loaded_df
        st.session_state.analysis = None
        st.session_state.report_path = None
        st.session_state.upload_signature = ('demo', DEMO_PATH.name)

    if st.session_state.loaded_df is None:
        st.info('Загрузите файл, откройте локальный набор из папки data/ или используйте demo-датасет, чтобы начать анализ.')
        return

    loaded_df = st.session_state.loaded_df
    source_name = st.session_state.source_name
    sensor_columns = [column for column in loaded_df.columns if column != 'timestamp']

    st.caption(f'Загруженный источник: {source_name}')

    left, right = st.columns([1.2, 1.8])
    with left:
        st.subheader('Параметры')
        st.write(f"Файл: `{source_name}`")
        target_sensor = st.selectbox('Параметр для прогноза', options=sensor_columns, index=0)
        run_clicked = st.button('Запустить анализ', type='primary')

    with right:
        st.subheader('Предпросмотр данных')
        st.dataframe(loaded_df.head(12), use_container_width=True)

    if run_clicked:
        with st.spinner('Выполняется анализ временного ряда...'):
            try:
                analysis = run_analysis(
                    frame=loaded_df,
                    source_name=source_name,
                    target_sensor=target_sensor,
                    window_size=window_size,
                    window_stride=window_stride,
                    horizon=horizon,
                    contamination=contamination,
                    scaling_method=scaling_method,
                )
                st.session_state.analysis = analysis
                st.session_state.report_path = None
            except ValueError as exc:
                st.error(str(exc))
                return

    analysis = st.session_state.analysis
    if analysis is None:
        st.warning('Нажмите «Запустить анализ», чтобы построить интервалы нестабильности и прогноз.')
        return

    _render_metrics(analysis)
    st.success(analysis.conclusion)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            create_timeseries_figure(
                cleaned_df=analysis.prepared.cleaned_df,
                intervals_df=analysis.detection.intervals,
                sensor=analysis.forecast.target_sensor,
            ),
            use_container_width=True,
        )
    with chart_right:
        st.plotly_chart(
            create_forecast_figure(
                cleaned_df=analysis.prepared.cleaned_df,
                forecast_df=analysis.forecast.forecast_df,
                sensor=analysis.forecast.target_sensor,
                normal_band=analysis.forecast.normal_band,
            ),
            use_container_width=True,
        )

    st.subheader('Найденные события')
    if analysis.detection.intervals.empty:
        st.write('Нестабильные интервалы не найдены.')
    else:
        event_table = analysis.detection.intervals[
            ['interval_id', 'start', 'end', 'duration', 'average_score', 'peak_state', 'main_contributor']
        ].copy()
        st.dataframe(event_table, use_container_width=True)

    st.subheader('Прогноз')
    st.dataframe(analysis.forecast.forecast_df, use_container_width=True)
    st.caption(
        f"MAE: {analysis.forecast.metrics['mae']} | "
        f"RMSE: {analysis.forecast.metrics['rmse']} | "
        f"Риск: {analysis.forecast.risk_score} ({analysis.forecast.risk_level})"
    )

    _render_explanations(analysis)

    st.subheader('Отчёт')
    report_format = st.radio('Формат отчёта', options=['html', 'txt'], horizontal=True)
    if st.button('Сформировать отчёт'):
        generated = generate_reports(analysis=analysis, output_dir=ROOT_DIR / 'reports', formats=[report_format])
        st.session_state.report_path = generated[report_format]

    if st.session_state.report_path:
        report_path = Path(st.session_state.report_path)
        st.download_button(
            label=f"Скачать отчёт ({report_path.suffix})",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime='text/html' if report_path.suffix == '.html' else 'text/plain',
        )
        st.caption(f'Отчёт сохранён в {report_path}')


if __name__ == '__main__':
    main()
