# Changelog

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
