# Python API and Jupyter

Everything the dashboard does is also available from Python, so you can use KikoLens inside a notebook (Jupyter, VS Code, Colab) or a script. Nothing leaves your machine.

```python
import kikolens as kl

df = kl.load("sales.csv")        # a file, a link, a Google Sheet, a database...
kl.profile(df)                   # quality score, problems and a profile of every column
pilot = kl.autopilot(df)         # a first full analysis
lab = kl.lab(df, "revenue")      # compare models; lab.leaderboard, lab.pipeline, lab.metrics
lab.plot()                       # the main chart of any result
```

Run `kikolens jupyter` to get a quickstart notebook that builds its own sample data and walks through the calls below (it opens JupyterLab if it is installed: `pip install "kikolens[jupyter]"`).

## Results

Every analysis returns a `Result`: a dictionary with attribute access that shows itself nicely in a notebook (a summary, notes and the main table) and keeps everything the analysis computed.

```python
res = kl.forecast(df, "date", "revenue", horizon=14)
res.summary      # the sentence the dashboard shows
res.table        # the main table (here the forecast with its range)
res.plot()       # the main chart, a Plotly figure
res.best         # any other field: res.leaderboard, res.series ...
```

An unknown field raises an error that lists what is available. Text from your data is escaped in the notebook display.

## Functions

| Call | What it does |
| :--- | :--- |
| `load(source, **kwargs)` | Read a dataset (the same sources as the dashboard). |
| `query(df, "select ... from df")` | Read-only SQL; the table is called `df`. Returns a DataFrame. |
| `ask(df, "question")` | A local model (Ollama) writes the SQL and it is run. |
| `profile(df)` | Quality score, problems and column profile. |
| `dictionary(df, descriptions=None)` | A data dictionary: one row per column. |
| `scan_privacy(df)` | Columns that look like personal data. |
| `autopilot(df, target=None)` | Clean, find patterns, model a column and forecast dates. |
| `lab(df, target, **kwargs)` | The model lab; keyword arguments as in the lab (`models`, `folds`, `tune`, `drop`, ...). |
| `forecast(df, date, value, horizon=12, agg="mean")` | Five models tested on the past; the best one forecasts. |
| `anomalies(df, **kwargs)` | Unusual rows and why. |
| `spikes(df, date, value)` | Unusual periods once trend and season are removed. |
| `drift(reference, current)` | Which columns changed between two datasets. |
| `why(df, metric, date, **kwargs)` | Why a figure changed between two periods. |
| `regress(df, target, features=None)` | Linear (number) or logistic (two values) regression, explained. |
| `survival(df, duration, event, group=None)` | Kaplan-Meier curves and a log-rank test. |
| `cohorts(df, user, date)` / `rfm(df, customer, date, amount)` | Retention and customer segments. |
| `distribution(series)` | Which distribution fits a column best. |
| `chart(df, "request")` | A chart described in words (needs Ollama): `res.figure`, `res.table`, `res.spec`. |
| `analyst(df)` | A conversation: `bot = kl.analyst(df); bot.ask("why did revenue fall?")` returns a turn with `.text`, `.figure`, `.table`, `.sql` and the steps taken in `.report`. |
| `classify_text(texts, ["a", "b"])` | Zero-shot classification of texts with the local model. |
| `formula(df, "revenue per unit")` | A formula for a new column described in words, already checked with the safe interpreter (`name`, `expression`, `preview`). |
| `link(left, right, left_on, right_on)` | Fuzzy match by names; `res.merged` has the joined table. |
| `select_features(df, target)` | Which columns to keep: consensus of four methods and a cross-validated curve; `res.recommended`, `res.removable`. |
| `segment(df, "gmm" \| "hierarchical" \| "dbscan")` | Segments with each group described in words; `res.labels` is the group of every row. |
| `ab_sequential(df, group, outcome, control, order=None)` | A sequential A/B test that stays valid when checked after every observation. |
| `granger(df, date, cause, effect)` / `forecast_drivers(df, date, target, drivers, scenario=...)` | Granger test; forecast using other columns, with what-if scenarios. |
| `effect(df, treatment, outcome, covariates=None, method="aipw")` | Causal effect with five estimators side by side; `res.effect`, `res.interval`, `res.table`, `res.balance`, `res.e_value`. |
| `uplift(df, treatment, outcome, covariates=None)` | Effect per row and who to target; `res.cate`, `res.deciles`, `res.qini`. |
| `discontinuity(df, running, outcome, cutoff)` | Effect of crossing a threshold, with a bandwidth sweep. |
| `synthetic_control(df, unit, time, outcome, treated_unit, intervention)` | Effect on one unit against a synthetic control; `res.p_value` comes from placebos. |
| `instrumental(df, outcome, endogenous, instruments, covariates=None)` | Two-stage least squares; `res.first_stage_f` tells whether the instrument is strong. |
| `simulate(inputs, formula, n=10000, correlations=None, threshold=None)` | Monte Carlo; `inputs` is a dictionary of distributions or a DataFrame whose columns are resampled. `res.mean`, `res.percentiles`, `res.p_above`, `res.sensitivity`, `res.samples`. |
| `optimize(formula_or_objective, bounds, constraints)` | A formula as text (bounds `{"x": (0, 10)}`, restrictions `["x + y <= 20"]`) or a linear program if you pass a dictionary of coefficients; `res.solution`, `res.objective`. |
| `marketing_mix(df, date, target, channels, controls=None)` | What each channel contributes; `res.table`, `res.parts`, `res.params`, `res.r2`. |
| `budget(model, total=None, low=0.5, high=2.0)` | A split of the spend per period from a `marketing_mix` result; `res.table`, `res.gain`. |
| `elasticity(df, quantity, price, controls=None, group=None, instrument=None, unit_cost=None)` | Price elasticity (log-log); `res.elasticity`, `res.interval`, `res.groups`, `res.optimal_price`. |
| `panel(df, path=None, definition=None, where=None)` | A panel of metrics, charts and tables. With a `path` ending in `.html`, `.pptx` or `.docx` it writes the file; without one it returns the `Panel` to add tiles to. |
| `recommender(df, user, item, rating=None)` | An item-based recommender: `.recommend(person)`, `.similar_items(item)`, `.popular()`. |
| `synthesize(df, n=None)` | Synthetic data; `res.data` is the table and `res.report` the fidelity and privacy figures. |
| `runs(target=None)` | The history of model lab runs (the dashboard and `kikolens lab` record them). |
| `save_model(lab_result, df, name)` / `score(df, "name")` | Keep a model and predict with it later. Only open model files you trust. |

## Magics

```python
%load_ext kikolens
%kl_profile df            # quality of a DataFrame
%kl_autopilot df revenue  # a first analysis, predicting 'revenue'
%kl_privacy df            # columns that look like personal data
```

```python
%%kl_sql df
select region, sum(revenue) as revenue from df group by region
```

The cell magic takes the name of a DataFrame and runs the body as a read-only query.

## Reproduce a session

The dashboard's *Export* → **Notebook** tab and `kikolens notebook data.csv analysis.ipynb --recipe recipe.json --target revenue` write a notebook that reloads the data, repeats every recipe step and trains a model.
