# KikoLens

Private, local-first data analysis. Load a dataset, check its quality, clean it, explore it, train models (including neural networks), forecast time series and export a report. Nothing leaves your machine: language-model features use [Ollama](https://ollama.com/) running locally.

## Features

| Area | What it does |
| :--- | :--- |
| **Data loading** | CSV, TSV, Excel, JSON / NDJSON, Parquet, `http(s)` links, and SQLite / PostgreSQL / MySQL. |
| **Quality and cleaning** | Quality score, detection of missing values, duplicates, constant columns, text that is really numbers or dates, and outliers. Every fix is undoable; the original file is never touched. |
| **Explore** | Scatter, line, bar, histogram, box, violin and density charts with column filters. |
| **Patterns** | Automatic detection of strong correlations, outliers, dominant categories and identifier columns. |
| **Statistics** | Summary table, per-column distributions, correlation matrix and strongest relationships. |
| **Ask** | Question the data in plain language (a local model writes the SQL) or run your own read-only SQL. |
| **Models** | AutoML for regression and classification with a written rationale, feature importance, row explanations and a what-if simulator. |
| **Neural network** | Configurable multilayer perceptron: layers, activation, optimizer, learning rate, regularization, early stopping. Shows the architecture, the training curve, permutation importance, and compares against the classical model. |
| **Time series** | Date detection, resampling, trend and seasonality, and a forecast with a 95% range that is backtested against a naive baseline. |
| **Segments** | k-means clustering with a silhouette-based hint for the number of groups. |
| **Report** | A single self-contained HTML page (print it to PDF from the browser). |

## Install

Requires Python 3.12 or newer.

```bash
git clone https://github.com/Francisco-B-O/kikolens.git
cd kikolens
pip install -e .
```

For the language-model features, install Ollama and pull a model:

```bash
ollama pull llama3.2
```

KikoLens reads these environment variables:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running. |
| `OLLAMA_MODEL` | `llama3.2` | Model used for summaries and plain-language questions. |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for the model. |

Everything except plain-language questions and written summaries works without Ollama.

## Dashboard

```bash
kikolens start            # opens http://localhost:8501
kikolens start --port 9000
```

The dashboard follows the light or dark theme chosen in Streamlit's settings menu.

## Command line

```bash
kikolens analyze data.csv                         # statistics and quality summary
kikolens sql data.csv "SELECT region, SUM(units) FROM df GROUP BY region"
kikolens ask data.csv "Which region sold the most?"
kikolens predict data.csv revenue                 # AutoML
kikolens predict data.csv revenue --nn --layers "64, 32" --save model.pkl
kikolens forecast data.csv date revenue --horizon 30
kikolens clean data.csv --output clean.parquet    # safe automatic cleaning
kikolens report data.csv --target revenue --nn --output report.html
```

In SQL the data is always the table `df`. Only read-only queries run; writes, multiple statements and file-reading functions are rejected.

## Docker

```bash
docker-compose up -d      # dashboard on :8501, Ollama on :11434
```

The first start downloads the model (about 2 GB) into the `ollama_data` volume. Change the model with `OLLAMA_MODEL=llama3.1 docker-compose up -d`.

## Notes on the models

* Dates are turned into numeric features (days since 1970, month, weekday) before training.
* Neural networks use scikit-learn's `MLPRegressor` / `MLPClassifier`; regression targets are standardized internally and predictions are returned in the original scale.
* Forecasts use a linear trend plus Fourier seasonality. They are checked on held-out data and the dashboard says when they do not beat repeating the last value.
* Only load `.pkl` model files you created yourself; unpickling can execute code.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Work happens on feature branches; changes reach `develop` and then `main` through pull requests that pass CI.

## License

MIT
