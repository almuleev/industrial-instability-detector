# graph-anomaly-analyzer

Python-проект (FastAPI + Streamlit) для анализа технологических временных рядов: поиск нестабильных режимов, краткосрочный прогноз и генерация отчётов.

## Project Status

> Stable MVP. Проект поддерживается автором в best-effort режиме.
> Формальных SLA, гарантированных сроков ответа и регулярных релизов нет.

## Возможности

- загрузка данных в форматах CSV/XLSX;
- валидация структуры и типов с понятными ошибками;
- предобработка временного ряда (очистка, сортировка, заполнение пропусков);
- поиск нестабильных участков (`normal` / `warning` / `unstable`);
- объединение аномальных окон в интервалы событий;
- прогноз по выбранному `sensor_*` и оценка краткосрочного риска;
- текстовые объяснения причин нестабильности;
- интерфейс Streamlit, API FastAPI и генерация HTML/TXT отчётов.

## Быстрый старт

### 1) Требования

- Python 3.10+
- `pip`

### 2) Установка

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Запуск интерфейса

```bash
python main.py ui
```

После старта откройте `http://127.0.0.1:8501`.

### 4) Запуск API

```bash
python main.py api
```

Swagger-документация будет доступна по адресу `http://127.0.0.1:8000/docs`.

### 5) CLI-анализ

```bash
python main.py analyze --file data/demo_timeseries.csv --sensor sensor_2
```

Отчёты сохраняются в `reports/`.

## Формат входных данных

```text
timestamp,sensor_1,sensor_2,sensor_3,...
```

- `timestamp` обязателен;
- остальные столбцы должны быть числовыми или приводимыми к числу;
- пропуски допустимы.

## API endpoints

- `POST /upload` - загрузка файла и создание сессии;
- `POST /analyze` - запуск анализа;
- `GET /results` - полные результаты анализа;
- `GET /forecast` - прогноз и риск;
- `GET /report` - получение отчёта.

## Структура репозитория

```text
.
├── app/                  # Бизнес-логика, UI и API
├── data/                 # Демо-данные
├── models/               # Локальные артефакты моделей (в git не коммитятся)
├── reports/              # Локальные отчёты (в git не коммитятся)
├── tests/                # Автотесты
├── .github/              # CI и шаблоны для GitHub
├── main.py               # Точка входа CLI/UI/API
├── requirements.txt
└── requirements-dev.txt
```

## Разработка

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

CI-пайплайн (`.github/workflows/ci.yml`) выполняет установку зависимостей, компиляционную smoke-проверку и тесты.

## Документы репозитория

В репозитории есть базовые служебные документы:

- `LICENSE`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `SUPPORT.md`
- `CHANGELOG.md`
- issue/PR templates в `.github/`

## Данные и отказ от ответственности

Файлы в `data/` предназначены для демонстрации и разработки. Перед использованием в production убедитесь, что набор данных не содержит чувствительной информации.

## Поддержка

- Проект поддерживается одним автором в best-effort формате.
- Issues и PR принимаются к рассмотрению, но без гарантированных сроков ответа.
- Ключевой фокус: стабильность MVP, читаемость кода и воспроизводимость запуска.
- Использование кода в production требует вашего собственного QA и security-аудита.

## Лицензия

Проект распространяется по лицензии MIT. См. [LICENSE](LICENSE).
