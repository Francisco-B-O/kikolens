# Changelog

## Unreleased

* **Analyst:** a chat with your data (dashboard section, `kikolens analyst`, `kl.analyst`) answered by a local model that only *chooses*: it draws charts from a specification that is validated against your columns, runs read-only SQL, or chains KikoLens analyses (thirteen tools) step by step, with every step shown and exportable as Python code. Offline fallback when Ollama is not running.
* **Charts from words** (`kikolens chart`, `kl.chart`), **zero-shot text classification** (`kikolens classify-text`) and **formulas from words** (`kl.formula`, checked with the safe interpreter).

* **Engineering:** lint (ruff) and a compatibility matrix (pandas 2 and 3, Linux and Windows, plus macOS and Python 3.13 as warnings) in CI; CodeQL, Dependabot, a release workflow that builds the package and creates the GitHub release from the changelog; pre-commit; contributing guide, code of conduct, security policy and issue and pull request templates; documentation site with MkDocs (built in strict mode in CI).
* **`kikolens doctor`** checks versions, optional extras, the user folder, Ollama and the configuration (exit code 1 when something required is missing; `--json` for scripts).
* **`kikolens.toml`** configuration (Ollama host, model and timeout, dashboard port and language), with environment variables taking precedence and bad files never stopping startup.
* **`kikolens benchmark`** times the main operations on generated data.
* Tidied dead code, unused imports and 12 repeated Spanish translation keys found by the linter.

* **Home page, data story and chart gallery:** a quality ring, sparklines and one-click actions on a new home page; the data told as chapters with charts; nine new charts (Sankey, treemap, radar, grouped correlations, correlation network, calendar, ridgelines, parallel coordinates, small multiples); built-in example datasets on the welcome screen; coloured tables.
* **Bayesian hyperparameter optimisation** in the model lab (`lab --bayes`), with the search history.
* **Sequential A/B tests** (mSPRT) that stay valid when checked after every visitor, a simulation planner and a Thompson-sampling bandit simulator (`ab-seq`, `bandit`).
* **Forecast with drivers** (scenarios, test of whether drivers help) and **Granger** tests (`forecast-drivers`, `granger`).
* **Feature selection** by consensus of four methods and a cross-validated curve (`select-features`).
* **Segment explorer:** Gaussian mixtures with automatic number of groups, hierarchical clustering with a dendrogram and DBSCAN, with groups described in words (`segment`).
* **Recommender** with an honest evaluation against popularity (`recommend`).
* **Synthetic data** with a fidelity, privacy and utility report (`synth`).
* **Run history** of every lab training, with comparison (`runs`).
* Python API: `select_features`, `segment`, `ab_sequential`, `granger`, `forecast_drivers`, `recommender`, `synthesize` and `runs`.

* **Python API for notebooks:** `import kikolens as kl` gives `load`, `profile`, `autopilot`, `lab`, `forecast`, `why`, `regress`, `survival`, `link`, `query` and more; results display as readable tables in Jupyter and keep every field. See `docs/python-api.md`.
* **IPython magics:** `%load_ext kikolens` adds `%kl_profile`, `%kl_autopilot`, `%kl_privacy` and `%%kl_sql`.
* **`kikolens jupyter`** writes a self-contained quickstart notebook and opens JupyterLab (optional extra `kikolens[jupyter]`).
* **Distribution fit:** which distribution describes a column best, with a Q-Q plot and a normality check; `kikolens fit`.
* **Section finder** in the dashboard sidebar, and `/openapi.json` in the model API.

* **Privacy:** detection of personal data with validated patterns (card check digit, IBAN, DNI/NIE, email, phone, IP, social security number) and by column name; masking, salted fingerprints and removal as recipe steps; k-anonymity. `kikolens privacy`.
* **Match records:** fuzzy join of two tables by names (accents, order, typos) with ambiguity warnings. `kikolens link`.
* **Data dictionary** and **Jupyter notebook** export (dashboard and `kikolens dictionary`, `kikolens notebook`); the notebook runs as generated.
* **Pipelines:** `kikolens run pipeline.json` runs a recipe, validation rules, a drift check against a baseline and model scoring, writes the outputs and exits with `2` when a chosen condition triggers.
* **Champion and challenger:** `kikolens models compare` and a tab in Model health say which saved model predicts better and whether the gap is reliable.
* Fix: the model API now reads the request body before answering an error, so clients on Windows no longer see a dropped connection on 401 and 404.

## 1.4.0

* **Autopilot:** a first full analysis in one click (dashboard and `kikolens autopilot`): safe cleaning as a replayable recipe, column roles, findings, a model against a baseline and a forecast, with a written summary and next steps.
* **PDF reports:** a PDF with quality, distributions, correlations, the model and the forecast; `kikolens report --pdf` and a download in the dashboard.
* **More models:** SVM, AdaBoost, Bagging, voting and stacking ensembles in the lab, and an optional PyTorch network (`pip install "kikolens[torch]"`).
* **Model lab, three new tabs:** conformal prediction ranges and class sets, counterfactuals ("what would change it"), and a fairness check across groups.
* **Feature ideas:** automatic search for ratios, products and other columns that improve a model in cross-validation, applied as recipe steps.
* **New analyses:** survival curves with log-rank test, retention by cohort, RFM segments, funnels, association rules, causal impact of a change and difference-in-differences, Bayesian A/B tests. Matching commands: `survival`, `cohorts`, `rfm`, `basket`, `impact`, `ab-bayes`, `features`.
* **Why it changed:** breaks the change of a total, an average or a count between two periods into the contribution of every region, channel or product (with mix and rate effects for averages), a waterfall and the pairs that moved most; `kikolens why`.
* **Regression explained:** linear and logistic regression with ranges, p-values, odds ratios, plain-language effects and diagnostics (collinearity, changing error spread, related errors, separation); `kikolens regress`.
* **Unusual periods:** anomalies in a time series once trend and season are removed, in the Anomalies section and as `kikolens spikes`.
* **Model health:** compares a saved model's real accuracy on newer data, period by period, with what it had in testing, and warns of a drop or a steady decline; `kikolens performance`.
* **Serving:** `kikolens serve` exposes a saved model as a JSON API on this computer, with a token required for anything else.
* **Spanish interface:** a language menu in the dashboard that translates sections, buttons and labels.

## 1.3.0

* **Panel:** pin metrics, charts and tables with global filters; export as one self-contained page; saved in projects; `kikolens panel`.
* **Monitoring:** monitors with a stored baseline, drift, quality, validation rules and model predictions, alerts, history, `watch`, schedule commands and an optional webhook; `kikolens monitor ...`.
* **Explainability:** Shapley values per row and global, error analysis by segment, decision thresholds with costs, and a row-by-row comparison of two models.
* **Text:** key terms, terms by group and by a number, themes, a classifier, rough tone, search and an optional local-model summary; `kikolens text`.
* **Quality:** missing-data patterns, smart fill (KNN, iterative, by group), merging of similar spellings, near-duplicate rows and validation rules; `kikolens validate`.
* **Time series:** five models tested with rolling-origin validation, an interval built from real test errors, special dates, level and trend changes, one forecast per group; `forecast --compare --changes --by`.
* **Export and connections:** formatted Excel workbooks, more file formats, writing to a database, pasted tables, Excel sheet selection, a database table or read-only query, shared Google Sheets and `s3://` / `gs://`; `kikolens export`, `kikolens to-db`.
* **Packaging:** project metadata, optional extras (`cloud`, `postgres`, `mysql`, `all`), a wheel build in CI and a container health check.

## 1.2.0

* Model lab with cross-validated comparison, tuning, leakage detection and diagnostics.
* Transformations, reproducible recipes and `.kikolens` project files.
* Model packages, batch prediction and a saved-models registry; `kikolens score` and `kikolens models`.
* Anomaly detection, statistical tests, A/B testing, PCA, t-SNE, VIF and drift detection.
* Dashboard areas, new command line tools, rewritten documentation.
* Fixes: dates and numbers stored as text are read as such when modeling; dates and identifiers are not reported as drift.

## 1.1.0

* Neural networks, data quality and cleaning, time series, command line, a new dashboard and documentation.
