<p align="center"><img src="docs/images/logo.svg" alt="KikoLens logo" width="96"></p>

# KikoLens

[![CI](https://github.com/Francisco-B-O/kikolens/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/Francisco-B-O/kikolens/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Francisco-B-O/kikolens/actions/workflows/codeql.yml/badge.svg?branch=develop)](https://github.com/Francisco-B-O/kikolens/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/kikolens)](https://pypi.org/project/kikolens/)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Local first](https://img.shields.io/badge/data-never%20leaves%20your%20machine-9a3412)

Private, local-first data analysis. Load a dataset, let the autopilot take a first pass, check its quality and fix it, reshape it, explore it, test hypotheses, compare many models and understand them, forecast, study customers and the effect of a change, watch a data source over time, and keep what you built: a recipe you can replay, a model you can score new data with (or serve behind a small local API), a panel and a report in HTML or PDF. Nothing leaves your machine; language-model features use [Ollama](https://ollama.com/) running locally.

![Model lab leaderboard](docs/images/lab-leaderboard-light.png)

<table>
  <tr>
    <td><img src="docs/images/panel-light.png" alt="Panel"></td>
    <td><img src="docs/images/monitoring-dark.png" alt="Monitoring, dark theme"></td>
  </tr>
  <tr>
    <td><img src="docs/images/lab-errors-light.png" alt="Where a model fails"></td>
    <td><img src="docs/images/text-dark.png" alt="Text analysis, dark theme"></td>
  </tr>
  <tr>
    <td><img src="docs/images/experiments-light.png" alt="Statistical tests"></td>
    <td><img src="docs/images/compare-dark.png" alt="Comparing two datasets, dark theme"></td>
  </tr>
</table>

## What it does

Fifty-two sections in six areas. The first five are the main path; **Advanced** holds the specialised ones.

| Area | Sections | What you get |
| :--- | :--- | :--- |
| **Data** | Home, Data story, Autopilot, Explore, Chart gallery, Quality and cleaning, Privacy, Match records, Transform, Patterns, Statistics, Ask, Analyst | Load CSV, Excel, JSON, Parquet, links, Google Sheets, `s3://` / `gs://` and SQL databases. A quality score with undoable fixes, detection of personal data with k-anonymity, fuzzy joins, and a recipe that records every step so it can be replayed on a new file. The **Autopilot** takes a first full pass in one click, and the **Analyst** chats with the data using a local model that only chooses the analysis: the numbers always come from the analysis itself. |
| **Analysis** | Experiments, Sequential tests, Anomalies, Dimensions, Distribution fit, Compare data, Segments, Why it changed, Causal effect, Survival | The right statistical test chosen for you, with effect sizes and corrections for looking at many things; A/B tests that stay valid when checked every day (mSPRT), Bayesian and CUPED; anomalies with the reason for each row; PCA and t-SNE; drift between two datasets; clustering; "why did this figure move?"; propensity, matching, doubly robust, synthetic control, instrumental variables and uplift; Kaplan-Meier curves. |
| **Models** | Models, Model lab, Feature ideas, Feature selection, Regression, Neural network, Time series, Classical series, Forecast with drivers, Predict, Model health, Monitoring, Run history | Up to 12 models compared with cross-validation against a baseline, tuning (grid and Bayesian), data-leak detection, Shapley values, where the model fails, decision thresholds with costs and conformal ranges; regression with p-values; a configurable neural network (scikit-learn or PyTorch); forecasts tested on the past; ADF / KPSS / STL / ARIMA; saved models that warn when new data no longer looks like the training data; monitoring with alerts. |
| **Learn** | Concept lab, Glossary | Eight simulations that show why statistics works (central limit theorem, p-values and peeking, confidence, power, overfitting, Simpson's paradox, regression to the mean, multiple testing), a bilingual glossary of 74 terms, and a "How to read this page" box in the main sections. No data needed. |
| **Share** | Panel, Report, Export, Project | A panel where a click on a chart filters the others, an HTML or PDF report, a formatted Excel workbook, a Jupyter notebook that reproduces the session, and a single `.kikolens` project file. |
| **Advanced** | Text, Customers, Impact, Recommendations, Survey, Network, Geography, Model governance, Synthetic data | Text mining and zero-shot classification, cohorts and RFM, impact of a change, a recommender evaluated honestly, survey analysis, network and geographic analysis, model stages with an audit log, and synthetic data with fidelity and privacy reports. |

The dashboard is in English and Spanish. The same analyses are available from a [command line](docs/cli.md), from [Python and notebooks](docs/python-api.md) and, for saved models, as a small [JSON API](docs/api-server.md).

## Try it

`examples/` has a synthetic sales dataset with deliberate flaws (missing values, duplicates, prices stored as text) and a weekly pattern, plus companion files:

| File | Use it for |
| :--- | :--- |
| `sales_demo.csv` | Everything: cleaning, models, forecasting. |
| `sales_new_quarter.csv` | A later quarter whose channel mix and prices have shifted. Score it with a model, monitor it, or compare it with the main file in **Compare data**. |
| `regions.csv` | A small table to practice joining (`region`). |
| `reviews.csv` | Customer comments with a rating, for the **Text** section. |
| `report.html` | The report generated from `sales_demo.csv`. |

```bash
kikolens start        # then drop examples/sales_demo.csv into the sidebar
```

A complete session from the terminal:

```bash
kikolens clean examples/sales_demo.csv --output clean.csv --save-recipe recipe.json
kikolens lab clean.csv revenue --exclude units --register sales-model
kikolens monitor create sales-feed clean.csv --model sales-model --recipe recipe.json
kikolens apply examples/sales_new_quarter.csv recipe.json --output new_clean.csv
kikolens score new_clean.csv sales-model --output predictions.csv
kikolens monitor run sales-feed --source examples/sales_new_quarter.csv   # exit code 2: it changed
```

## In a notebook

```python
import kikolens as kl

df = kl.load("sales.csv")
kl.profile(df)                 # quality and problems
kl.autopilot(df)               # a first full analysis
lab = kl.lab(df, "revenue")    # compare models
lab.plot()
```

`kikolens jupyter` writes a quickstart notebook and opens JupyterLab; `%load_ext kikolens` adds magics. See the [Python API guide](docs/python-api.md).

## Install

Requires Python 3.12 or newer.

```bash
pip install kikolens
kikolens doctor        # checks versions, optional extras, Ollama and your configuration
```

Or from the source:

```bash
git clone https://github.com/Francisco-B-O/kikolens.git
cd kikolens
pip install -e .
kikolens doctor        # checks versions, optional extras, Ollama and your configuration
```

Settings (the Ollama model, the port, the language the dashboard opens in) go in a `kikolens.toml`: see [Configuration](docs/configuration.md).
To contribute, read [CONTRIBUTING.md](CONTRIBUTING.md); the documentation is built with `mkdocs` (`docs/`).

Optional extras: `pip install -e ".[cloud]"` for `s3://` and `gs://`, `".[postgres]"` and `".[mysql]"` for those databases, `".[jupyter]"` for JupyterLab, `".[torch]"` for the PyTorch network (a large download, so it is not in `all`), and `".[all]"` for the rest.

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
| `KIKOLENS_HOME` | `~/.kikolens` | Where saved models and monitors are kept. |

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
kikolens clean data.csv --output clean.parquet --save-recipe recipe.json
kikolens apply new.csv recipe.json --output new_clean.csv   # repeat a recipe
kikolens validate data.csv rules.json             # exit code 2 when a rule is broken
kikolens predict data.csv revenue --nn --layers "64, 32"    # AutoML or neural network
kikolens lab data.csv revenue --register my-model           # compare, tune, save
kikolens score new.csv my-model --output predictions.csv    # use a saved model
kikolens models                                             # list, info, delete
kikolens forecast data.csv date revenue --horizon 30 --compare --changes
kikolens forecast data.csv date revenue --by region         # one forecast per group
kikolens anomalies data.csv --share 2 --output odd_rows.csv
kikolens screen data.csv churned                  # which columns relate to a target
kikolens ab data.csv variant converted --control A
kikolens text reviews.csv review --group product --themes 4
kikolens drift reference.csv current.csv          # exit code 2 when the data changed a lot
kikolens monitor create NAME SOURCE / run / list / history / watch / schedule / delete
kikolens panel data.csv --output panel.html
kikolens panel data.csv --where "region in North,West"
kikolens export data.csv out.xlsx --workbook      # formatted workbook with the analysis
kikolens to-db data.csv sqlite:///results.db results
kikolens report data.csv --target revenue --nn --output report.html
kikolens project work.kikolens --output processed.csv
```

In SQL the data is always the table `df`. Only read-only queries run; writes, multiple statements and file-reading functions are rejected. The full list of options is in the [command line reference](docs/cli.md).

## Docker

```bash
docker-compose up -d      # dashboard on :8501, Ollama on :11434
```

The first start downloads the model (about 2 GB) into the `ollama_data` volume. Change the model with `OLLAMA_MODEL=llama3.1 docker-compose up -d`. The image has a health check on Streamlit's health endpoint.

## Good to know

* **Recipes replay, they do not remember data.** Every cleaning and transformation step is recorded with its parameters. A recipe applies to any file with the same columns, and can be exported as JSON or as a Python script. Imputers are fitted again on the data a recipe is applied to.
* **Formulas are not code.** Expressions such as ``price * units`` are read by a small arithmetic interpreter that only allows numbers, columns, operators and a few math functions, so a recipe from someone else cannot run anything. Text that starts with `=` is stored as text in Excel exports, never as a formula.
* **Models are packages.** A saved model carries the columns it expects, their training ranges, its classes and, for regression, the typical size of its errors, so scoring new data can warn you when the data no longer looks like the training data.
* **Monitors do not install anything.** `kikolens monitor schedule` prints the cron and Windows commands; you decide whether to run them. A webhook is called only if you configure one, and only for alerts.
* **Pickles run code.** Model files (`.klmodel`, `.pkl`) are Python pickles. Open only files you created or trust. Project files store data as Parquet and are safe to open; the models inside are loaded only if you tick the box.
* **Writing to a database never overwrites by default.** Replacing a table needs an explicit choice (and a confirmation in the dashboard).
* Dates and numbers stored as text are read as dates and numbers when modeling, and the lab says which columns it converted.
* The tone in the Text section comes from a small word list: it is a rough picture, not a verdict. The dashboard is available in English and Spanish.

## Documentation

* [Dashboard guide](docs/dashboard.md): every section, with screenshots
* [Command line reference](docs/cli.md)
* [Workflows](docs/workflows.md): from raw file to monitored model
* [How the methods work](docs/models.md): preprocessing, models, tests, anomalies, text, forecasting, drift, monitoring
* [Changelog](CHANGELOG.md)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python -m build           # builds the wheel, as CI does
```

The screenshots in `docs/images` and the files in `examples/` are regenerated with `pip install -e ".[docs]" && python docs/make_screenshots.py`.

Work happens on feature branches; changes reach `develop` and then `main` through pull requests that pass CI.

## License

MIT
