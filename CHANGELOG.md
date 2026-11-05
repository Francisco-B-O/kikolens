# Changelog

## Unreleased

* **Concept lab** (new area "Learn", `kikolens learn`, `kl.concepts` and `kl.concept`): simulations that teach why statistics works the way it does, with a plain-language conclusion written from the simulated numbers, a suggestion of what to change next and a seed so results are repeatable. Eight concepts: the central limit theorem, what a p-value is (including how peeking at results inflates false alarms), what "95% confidence" means, how many people are enough (power), bias, variance and overfitting, Simpson's paradox, regression to the mean, and why testing many things finds something (with Bonferroni and Benjamini-Hochberg). It needs no data and opens from the welcome screen. Each is checked in the tests against the theory (for example 5% false alarms, 64 people per group for an effect of 0.5, 64% of studies with a false alarm for 20 tests).

* **Survey analysis** (dashboard section "Survey", `kikolens likert`, `reliability`, `nps`, `crosstab`, `key-drivers`, `survey-groups` and `response-quality`, and `kl.likert`, `kl.cronbach`, `kl.nps`, `kl.crosstab`, `kl.key_drivers`, `kl.compare_items` and `kl.response_quality`): scale questions as numbers or as text ("Strongly agree", "Muy satisfecho") with the favourable and unfavourable share; Cronbach's alpha with its interval, the correlation of each question with the rest and the alpha without it, and a warning for reverse-worded questions; NPS with its interval and the groups that stand out; cross-tabs with adjusted residuals and Cramér's V; key drivers by Johnson's relative weights, which share the explained variance out exactly even when questions overlap, crossed with how well each question scores to say what to improve first; ordinal tests between groups corrected for looking at many questions; and straight-liners, blank answers and speeders.

* **Multi-touch attribution** (new tab "Attribution" in Marketing mix, `kikolens attribution`, `kl.attribution`): from the contacts of each person (when, channel, whether they converted, optionally the value), the credit for the conversions shared by seven models side by side: first touch, last touch, linear, time decay, position based (40-20-40), a Markov chain (the drop in the chance of converting if a channel did not exist) and exact Shapley values; every model shares out exactly the conversions it was given. Notes point out where the models disagree and remind that attribution splits credit but does not measure what a channel causes.

* **Inventory and demand planning** (dashboard section "Inventory", `kikolens abc-xyz`, `newsvendor`, `eoq`, `reorder`, `stock-simulate` and `inventory-plan`, and `kl.abc_xyz`, `kl.newsvendor`, `kl.eoq`, `kl.reorder_policy`, `kl.simulate_stock_policy` and `kl.inventory_plan`): ABC by value and XYZ by demand variation with the Syntetos-Boylan class of each item (smooth, erratic, intermittent, lumpy) and the approach that suits it; the newsvendor quantity from the critical ratio, with normal demand or past sales; the economic order quantity; safety stock and reorder point for a service level with variable demand and delivery time, plus the fill rate; a simulation of the (s, Q) policy that checks the formulas (it found that checking stock once a day orders half a day of demand late, which is why it works in finer steps); and a policy for every item in a table of sales.

* **Process control** (dashboard section "Process control", `kikolens control-chart`, `capability` and `shift-detect`, `kl.control_chart`, `kl.process_capability` and `kl.shift_detection`): control charts for individual values, subgroup means and counts of defectives or defects (p, u and c, with limits that follow each sample size), the six classic signal rules with the false-alarm rate of the ones you switch on (by default two, about 0.6 % of points), limits set from a first stretch that is cleaned of special causes and then frozen for monitoring, capability indices (Cp, Cpk, Pp, Ppk, Cpm) with the share out of specification and warnings for off-centre, drifting or non-normal data, and CUSUM and EWMA for small sustained shifts, which also date when the shift began.

* **Is the A/B test sound?** (new tab in Experiments, `kikolens experiment`, `srm` and `cuped`, `kl.experiment`, `kl.srm` and `kl.cuped`): a sample ratio mismatch test that stops the verdict when the split of users is not what was planned; the effect of every group against the control (Holm-corrected with several variants, proportions for conversions); CUPED, which uses what each row did before the test to narrow the interval (measured against the theoretical gain and checked for coverage by simulation); the effect by week, with a warning for novelty effects that fade; effects by segment compared with the rest of the rows and corrected for looking at many; and the smallest effect the test could have detected, so that "no difference" is not read as "no effect".

* **Geography** (dashboard section "Geography", `kikolens geo-check`, `geo-distances`, `geo-hotspots`, `spatial` and `geo-grid`, `kl.geo_check`, `kl.geo_distances`, `kl.geo_hotspots`, `kl.spatial_autocorrelation` and `kl.geo_grid`): latitude and longitude columns found by name and range, with checks for swapped, out-of-range, missing and 0,0 coordinates; kilometres to a point or to the nearest centre with the share each centre serves; dense zones (DBSCAN on the sphere); Moran's I with a permutation test and Getis-Ord Gi* hot and cold spots corrected for multiple tests; and grid summaries. Maps are drawn without downloading tiles (outlines are opt-in because Plotly fetches them). No new dependencies.

* **Network analysis** (dashboard section "Network", `kikolens network`, `network-path`, `suggest-links` and `network-robustness`, `kl.network`, `kl.network_path`, `kl.suggest_links` and `kl.network_robustness`): from a list of links, the nodes that matter (PageRank, betweenness, connections, clustering), communities by modularity (Louvain), how connected the network is, typical and longest distances, the shortest path between two nodes, the links that are probably missing (Adamic-Adar) and how fast the network falls apart when its best-connected nodes are removed compared with random ones. Written on SciPy sparse matrices without new dependencies, and checked against networkx to 1e-8 in the tests when it is installed.

* **Text mining** (new tabs in the Text section, `kikolens topics`, `near-duplicates`, `similar`, `patterns`, `keyphrases` and `readability`, `kl.topics`, `kl.near_duplicates`, `kl.similar`, `kl.text_patterns`, `kl.keyphrases` and `kl.readability`): topics with NMF or LDA and the number chosen by coherence (with example texts, overlap warnings and, with a date, how each topic's share changes), near-duplicate groups that tolerate typos, texts most like a phrase, emails, links, phones, amounts, percentages, dates, hashtags and mentions (with a pointer to Privacy when personal data shows up), key phrases (RAKE), word pairs that go together (normalised PMI) and reading ease (Flesch, Fernández-Huerta). All local, no new dependencies.

* **Model governance** (dashboard section "Model governance", `kikolens governance ...`, `kl.governance` and `kl.model_card`): saved models get an owner, a risk level and a model card; they move through draft, staging, production and archived, with checks at each step (documentation complete, file unchanged since registration, a passed validation on new data in the last 90 days that is not worse than training beyond what sampling noise explains, not reliably worse than the model in production, and for high risk the approval of someone other than the owner). Only one model per target is in production; promoting another archives the old one and `rollback` brings it back. Overrides need a reason. Every change goes to an audit log chained by hashes, so an edited or removed line is detected. Models can be referred to by stage (`kikolens serve sales@production`).

* **Classical time series** (dashboard section "Classical series", `kikolens stationarity`, `acf`, `stl` and `arima`, `kl.stationarity`, `kl.autocorrelation`, `kl.decompose_stl` and `kl.arima`): ADF and KPSS read together with the number of differences they suggest; autocorrelation with bands and the Ljung-Box test; robust STL decomposition with the strength of trend and season; and seasonal ARIMA, fitted or chosen by AICc over small candidate orders, with forecast ranges, residual diagnostics and a rolling-origin comparison with repeating the last season, which the model must beat to be trusted. No new dependencies.

* **Plugins:** your own analyses and file formats from a Python file with a `register(registry)` function, found in `~/.kikolens/plugins`, in `[plugins] paths` of `kikolens.toml` or in packages with the `kikolens.plugins` entry point. They get a form in the dashboard (section Plugins), `kikolens plugin` and `kikolens plugins`, `kl.run` and `kl.list_plugins`, a place among the Analyst's tools and the API server. Arguments are checked before the plugin runs, a plugin that fails to load never stops KikoLens, and `KIKOLENS_PLUGINS=0` turns them off. See `docs/plugins.md`.
* **API server** (`kikolens api`): datasets in memory, every main analysis from a closed list plus plugin analyses at `POST /run/{function}`, background jobs, served models, an OpenAPI description, CORS on request, a mandatory token off this computer and a data folder as the only way to read files. See `docs/api-server.md`.
* `kl.regress` and `POST /run/regress` now say so when the target column does not exist instead of raising `KeyError`.

* **Panel with cross-filtering and Office export:** click bars or slices, or drag a box on a chart, and the other tiles of the panel filter to the selection (the chart you clicked keeps showing every option); the selection is shown, can be cleared, and travels into the exports. The panel exports to **PowerPoint** (editable native charts, key figures, tables, a slide for what could not be calculated) and **Word** (charts as images, tables), from the dashboard, `kikolens panel --format pptx|docx`, or `kl.panel(df, "file.pptx")`. `--where "region in North,West"` filters from the terminal. Needs the new extra `kikolens[office]`.

* **Marketing mix and price elasticity** (dashboard section "Marketing mix", `kikolens mmm` and `elasticity`, `kl.marketing_mix`, `kl.budget` and `kl.elasticity`): what each channel adds to the outcome with carry-over (adstock), saturation, trend, season and controls, tuned on the last periods rather than the ones already seen, with a check of which channels can be told apart from zero and a warning when channels move together; a budget split from the response curves; and log-log price elasticity with robust errors, one per group, a margin-maximising price, and an instrument to correct for prices that follow demand.

* **Simulation and optimisation** (dashboard section "Simulation", `kikolens simulate`, `optimize` and `lp`, `kl.simulate` and `kl.optimize`): Monte Carlo with seven distributions plus resampling of real columns, correlations through a Gaussian copula, percentiles, chance of passing a threshold, value at risk, a tornado of what moves the result and a convergence chart; linear and integer programming with the worth of relaxing each limit; optimisation of any formula with bounds and restrictions, checked with the safe interpreter.

* **Causal inference** (dashboard section "Causal effect", `kikolens effect`, `uplift`, `discontinuity`, `synth-control` and `iv`, and `kl.effect`, `kl.uplift`, `kl.discontinuity`, `kl.synthetic_control`, `kl.instrumental`): treatment effects with propensity weighting, matching, regression and a doubly robust estimator side by side with the raw difference, covariate balance, overlap checks and an E-value for hidden confounders; uplift per row with a Qini curve; regression discontinuity with a bandwidth sweep and a manipulation test; synthetic control with placebo tests; instrumental variables with a weak-instrument warning.

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
