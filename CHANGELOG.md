# Changelog

## 1.0.0

First stable release. KikoLens is a private, local-first data analysis workbench: the data never leaves your machine, and the optional language-model features run on a local [Ollama](https://ollama.com/) server.

* **Data:** loading from files, links, Google Sheets, object storage and SQL databases; a quality score with undoable fixes; detection and masking of personal data with k-anonymity; fuzzy joins; replayable recipes with a safe formula interpreter; a one-click autopilot; an analyst that chats with the data using validated specifications.
* **Analysis:** automatic choice of statistical tests, classical, Bayesian and always-valid sequential A/B tests with CUPED and split checks, anomaly detection, PCA and t-SNE, drift, clustering, "why did this figure change", survival, and causal inference (propensity, matching, doubly robust, uplift, regression discontinuity, synthetic control, instrumental variables).
* **Models:** a model lab with cross-validation against a baseline, tuning, leak detection, Shapley values, error analysis, decision thresholds and conformal ranges; regression with inference; a neural network in scikit-learn or PyTorch; time series (backtested forecasts, ADF / KPSS / STL / ARIMA); versioned model packages that warn about unfamiliar data; model health, governance and monitoring.
* **Learn:** a concept lab with eight simulations and a bilingual glossary with a reading guide for the main sections.
* **Advanced:** text mining, customers and cohorts, impact of a change, recommendations, surveys, networks, geography and synthetic data.
* **Share:** a panel with cross-filtering, HTML and PDF reports, a formatted Excel workbook, a Jupyter notebook of the session and a single-file project.
* **Everywhere:** the dashboard (English and Spanish, light and dark), a command line, a Python API with Jupyter magics, a small model server and a local JSON API server.
* **Engineering:** unit and end-to-end tests, CI on Windows, Linux and macOS with pandas 2 and 3, CodeQL, a Docker image and a documentation site.
