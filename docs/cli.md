# Command line reference

Every command accepts a local file, an `http(s)` link or a database URI (`sqlite:///data.db`, `postgresql://user:pass@host/db`) wherever it says `SOURCE`.

## Data

| Command | What it does |
| :--- | :--- |
| `kikolens start [--port N]` | Launch the dashboard. |
| `kikolens analyze SOURCE` | Statistics for every column and the data quality summary. `--no-quality` skips the latter. |
| `kikolens sql SOURCE "QUERY"` | Run a read-only SQL query. The data is the table `df`. |
| `kikolens ask SOURCE "QUESTION" [--model NAME]` | A local model writes the SQL, then it runs. Needs Ollama. |
| `kikolens clean SOURCE --output FILE [--save-recipe FILE]` | Apply the safe automatic cleaning and save as `.csv` or `.parquet`. `--save-recipe` also writes the steps as a JSON recipe. |
| `kikolens apply SOURCE RECIPE --output FILE [-d PATH]` | Repeat a recipe on another file. `-d PATH` (or `-d name=PATH`) supplies the extra datasets that join steps need; it can be repeated. Without a name, the file's name is used, which is how the dashboard names uploaded files. |
| `kikolens validate SOURCE [RULES.json] [--save FILE]` | Check the data against validation rules (or the rules suggested by the data). Exit code `2` when a rule is broken. |
| `kikolens export SOURCE OUTPUT [--recipe FILE] [--workbook]` | Convert to `.csv`, `.tsv`, `.xlsx`, `.json`, `.ndjson`, `.parquet` or `.feather`. `--workbook` makes a formatted Excel workbook with quality, statistics and recipe sheets. |
| `kikolens to-db SOURCE DATABASE TABLE [--if-exists fail\|append\|replace]` | Write the data into a table. It never overwrites unless you say `replace`. |
| `kikolens autopilot SOURCE [--target COL] [--horizon N] [--thorough] [--clean FILE] [--recipe FILE] [--pdf FILE]` | A first full analysis: clean, find patterns, model a column and forecast dates. `--clean` and `--recipe` save the cleaned data and the steps; `--pdf` writes a report. |
| `kikolens privacy SOURCE [--scan] [--mask a,b] [--hash a,b] [--drop a,b] [--salt S] [--output FILE] [--recipe FILE] [--k N --quasi a,b]` | Find personal data, mask / hash / remove columns and measure k-anonymity. The salt used by `--hash` is written to the recipe, so keep the recipe private. |
| `kikolens dictionary SOURCE [--output FILE.md\|.csv\|.xlsx]` | Describe every column: meaning, empty share, typical values, ranges and warnings. |
| `kikolens notebook SOURCE OUTPUT.ipynb [--recipe FILE] [--target COL] [--exclude a,b] [--date COL --value COL] [--horizon N]` | Write a Jupyter notebook that reproduces the analysis. |
| `kikolens link LEFT RIGHT LEFT_COLUMN RIGHT_COLUMN [--threshold 0.85] [--output FILE]` | Match records whose names are not spelled the same way. |
| `kikolens run PIPELINE.json` | Run a pipeline file (see below). Exit code `2` when a `fail_on` condition triggers, `1` when something cannot be read. |
| `kikolens jupyter [FOLDER] [--no-open]` | Write a quickstart notebook for the Python API and open it in JupyterLab if it is installed. See the [Python API guide](python-api.md). |
| `kikolens fit SOURCE COLUMN [--only normal,gamma]` | Which distribution describes a numeric column best (AIC, Kolmogorov-Smirnov, normality). |
| `kikolens segment SOURCE [gmm\|hierarchical\|dbscan] [--columns a,b] [--groups 4] [--output FILE]` | Segment rows and describe each group in words; `--output` adds a group column. |
| `kikolens recommend SOURCE USER ITEM [--rating COL] [--for PERSON] [--similar-to ITEM] [--evaluate]` | Item-based recommender from a table of who used what. |
| `kikolens synth SOURCE OUTPUT [--rows N] [--seed N] [--report]` | Generate synthetic data with the same distributions and correlations; `--report` measures fidelity and privacy. |
| `kikolens analyst SOURCE "QUESTION" [--model M] [--no-samples] [--html FILE] [--trail FILE.py]` | Ask a question in your own words: a local model decides whether to draw a chart, run SQL or chain analyses, and every step is shown. `--trail` writes the steps as Python code. Needs Ollama; without it a basic analysis is chosen from keywords. |
| `kikolens chart SOURCE "REQUEST" [--output chart.html\|spec.json]` | Describe a chart in words; the model writes a specification that KikoLens validates and draws. |
| `kikolens classify-text SOURCE COLUMN --labels a,b,c [--output FILE] [--limit 300]` | Classify a text column into labels you choose, without training anything. |
| `kikolens project FILE.kikolens [--output FILE]` | Show what a project contains (notes, recipe) and optionally export its processed data. |

## Models

| Command | What it does |
| :--- | :--- |
| `kikolens predict SOURCE TARGET` | Train AutoML on `TARGET` and print metrics and top features. |
| `kikolens predict SOURCE TARGET --nn [--layers "64, 32"]` | Same, with a neural network. |
| `kikolens lab SOURCE TARGET` | Compare many models with cross-validation, tune the best one and check for data leaks. |
| `kikolens score SOURCE MODEL [--output FILE] [--raw]` | Predict a file with a saved model. `MODEL` is a `.klmodel` path or the name of a saved model. `--raw` repeats the cleaning steps the model was trained with first. |
| `kikolens models` | List the saved models. |
| `kikolens models info NAME [--card FILE]` | Metrics and expected columns of a model; `--card` writes a Markdown model card. |
| `kikolens models compare SOURCE MODEL MODEL [MODEL...]` | Which of several saved models predicts better on data whose real results are known, and whether the gap is reliable. |
| `kikolens models delete NAME` | Remove a saved model. |
| `kikolens forecast SOURCE DATE VALUE [--horizon N] [--freq D] [--agg mean]` | Forecast a numeric column. Frequencies: `h`, `D`, `W`, `MS`, `QS`, `YS`. |
| `kikolens forecast ... --compare` | Compare five models with rolling-origin tests and forecast with the best one. |
| `kikolens forecast ... --changes` | Also list level shifts and trend changes in the series. |
| `kikolens forecast ... --by COLUMN` | One forecast per value of a column (the 20 with the most activity). |
| `kikolens serve MODEL [--port 8000] [--host 127.0.0.1] [--token T]` | Serve a saved model as a small JSON API. `POST /predict` with `{"rows": [{...}]}` returns predictions and warnings; `GET /model` describes the inputs; `GET /health`. It listens only on this computer unless you give another `--host`, which requires `--token` (sent as `Authorization: Bearer T`). |

`predict` and `lab` take `--save PATH` to write a model package to a file and `--register NAME` to keep it with your saved models. `lab` also takes `--folds`, `--no-tune`, `--bayes` (Bayesian optimisation instead of a random search), `--models "Ridge,Random Forest"`, `--balance` and `--exclude col1,col2`. Besides the default models, `--models` accepts `SVM`, `AdaBoost`, `Bagging`, `Voting Ensemble`, `Stacking Ensemble` and, with PyTorch installed, `PyTorch MLP`.

## Analysis

| Command | What it does |
| :--- | :--- |
| `kikolens anomalies SOURCE [--method M] [--share 2] [--columns a,b] [--output FILE]` | Flag the most unusual rows and say why. Methods: `Isolation Forest`, `Local Outlier Factor`, `Robust z-score`, `Mahalanobis distance`. |
| `kikolens screen SOURCE TARGET` | Test every column against `TARGET` and list those reliably related to it, corrected for testing many columns. |
| `kikolens ab SOURCE GROUP OUTCOME --control VALUE [--success VALUE]` | Compare a control and a treatment group on rates or averages. |
| `kikolens drift REFERENCE CURRENT [--output FILE]` | Compare two datasets column by column. Exits with code `2` when the data changed a lot, so it works as an alarm in scripts and CI. |
| `kikolens text SOURCE COLUMN [--group COL] [--ngram 1\|2\|3] [--themes K]` | Key terms of a free-text column, the terms that distinguish each group, and themes. |
| `kikolens panel SOURCE [PANEL.json] [--output FILE] [--save FILE]` | Build a one-page panel of metrics, charts and tables; without a definition, one is suggested from the data. |
| `kikolens report SOURCE [--target COL] [--nn] [--ai] [--output FILE] [--pdf FILE]` | Build the HTML report, and a PDF with `--pdf`. |
| `kikolens survival SOURCE DURATION EVENT [--group COL]` | Kaplan-Meier medians and a log-rank test between groups. |
| `kikolens cohorts SOURCE USER DATE [--period Day\|Week\|Month]` | Retention by cohort. |
| `kikolens rfm SOURCE CUSTOMER DATE AMOUNT [--output FILE]` | Recency, frequency and spend segments. `--output` writes every customer with their segment. |
| `kikolens basket SOURCE TRANSACTION ITEM [--support 0.02] [--confidence 0.3]` | Association rules: what is bought together. |
| `kikolens impact SOURCE DATE VALUE --at YYYY-MM-DD [--control a,b] [--agg mean\|sum]` | Estimate the effect of a change against what would have happened. |
| `kikolens regress SOURCE TARGET [--using a,b] [--positive VALUE] [--output FILE]` | Linear regression for a number and logistic for a two-valued column: coefficients with ranges and p-values, each effect in a sentence, and warnings about collinearity or unreliable errors. |
| `kikolens spikes SOURCE DATE VALUE [--agg mean\|sum] [--sensitivity 3.5] [--output FILE]` | Unusual periods in a series once trend and season are taken out. |
| `kikolens why SOURCE DATE [--metric COL] [--agg sum\|mean\|count] [--before START,END] [--after START,END] [--by a,b]` | Explain why a figure changed between two periods: which values of which columns account for it. The end date is not included; by default it compares the last third of the dates with the one before. |
| `kikolens ab-seq SOURCE GROUP OUTCOME CONTROL [--order COL] [--alpha 0.05]` | Sequential A/B test (mSPRT): valid even if you look after every observation. |
| `kikolens bandit RATES [--visitors N]` | Simulate a Thompson-sampling bandit against an even split. |
| `kikolens granger SOURCE DATE CAUSE EFFECT [--max-lag 7]` | Does the past of one column help predict another? Predictive, not proof of cause. |
| `kikolens effect SOURCE TREATMENT OUTCOME [--covariates a,b] [--method aipw\|regression\|ipw\|matching] [--treated-value V]` | Causal effect of a treatment, adjusting for confounders, with the raw difference, balance and overlap warnings and an E-value. |
| `kikolens uplift SOURCE TREATMENT OUTCOME [--covariates a,b]` | Who responds most to a treatment: effect by decile and a Qini curve. |
| `kikolens discontinuity SOURCE RUNNING OUTCOME CUTOFF [--bandwidth H]` | Effect of crossing a threshold, with a bandwidth sweep and a manipulation test. |
| `kikolens synth-control SOURCE UNIT TIME OUTCOME TREATED_UNIT INTERVENTION` | Effect on one unit against a weighted mix of control units, with placebo tests. |
| `kikolens iv SOURCE OUTCOME ENDOGENOUS --instruments z1,z2 [--covariates a,b]` | Instrumental variables (two-stage least squares) with a first-stage strength check. |
| `kikolens forecast-drivers SOURCE DATE TARGET --drivers a,b [--scenario price=1.1] [--horizon N]` | Forecast a column using other columns too, and test whether they help. |
| `kikolens select-features SOURCE TARGET [--exclude a,b] [--output FILE]` | Which columns are worth keeping: consensus of four methods and a cross-validated curve. |
| `kikolens runs [--target COL] [--best] [--compare id,id] [--delete id]` | The history of model lab runs. |
| `kikolens ab-bayes SOURCE GROUP OUTCOME CONTROL [--success VALUE]` | Bayesian A/B: the probability a variant is better and the cost of choosing wrongly. |
| `kikolens features SOURCE TARGET [--max-new 5] [--apply FILE] [--recipe FILE]` | Search for ratios, products and other new columns that improve a model; `--apply` writes the data with them and `--recipe` the steps. |

## Monitoring

| Command | What it does |
| :--- | :--- |
| `kikolens performance SOURCE MODEL [--date COL] [--period Day\|Week\|Month\|Quarter] [--raw]` | Is a saved model still good? Scores newer data whose real results are known, period by period, against the accuracy it had in testing. Exit code `2` on a warning. |
| `kikolens monitor create NAME SOURCE [--baseline DATA] [--recipe FILE] [--model NAME] [--rules FILE] [--webhook URL] [--drift-level high\|medium] [--quality-drop 10]` | Create a monitor. The baseline (the source as it is now, unless you give another) is stored inside the monitor. |
| `kikolens monitor run NAME [--source FILE]` | One check. Exit code `0` fine or warnings only, `2` alerts, `1` the data could not be read. |
| `kikolens monitor list` / `history NAME [--last N]` | The monitors and the past checks of one. |
| `kikolens monitor watch NAME [--interval 300]` | Keep running and check whenever the watched file changes. |
| `kikolens monitor schedule NAME [--every hourly\|daily\|weekly]` | Print the cron and Windows commands that run the check on a schedule. Nothing is installed. |
| `kikolens monitor delete NAME` | Remove a monitor and its history. |

A check records rows, quality score, drift against the baseline, broken rules and the model's predictions, and raises alerts for: drift at or above the chosen level, a quality drop, a change of 50% or more in rows, columns missing from or added to the data, columns that changed from numbers to text, model inputs outside the training range, and a shift of 20% in the average prediction (or 10 points in the classifier's confidence). If a webhook is configured it receives a short JSON summary, only when a check ends in `alert` or `failed`.

## Examples

```bash
kikolens analyze examples/sales_demo.csv
kikolens sql examples/sales_demo.csv "SELECT region, ROUND(SUM(revenue)) AS total FROM df GROUP BY region ORDER BY total DESC"
kikolens clean examples/sales_demo.csv --output clean.csv --save-recipe recipe.json
kikolens lab clean.csv revenue --folds 5 --register sales-model
kikolens apply examples/sales_new_quarter.csv recipe.json --output new_clean.csv
kikolens score new_clean.csv sales-model --output predictions.csv
kikolens drift clean.csv new_clean.csv --output drift.csv
kikolens anomalies clean.csv --share 1 --output odd_rows.csv
kikolens monitor create sales-feed clean.csv --model sales-model --recipe recipe.json
kikolens monitor run sales-feed --source examples/sales_new_quarter.csv
kikolens text examples/reviews.csv review --group product --themes 3
kikolens forecast clean.csv date revenue --horizon 30 --compare --changes
kikolens export clean.csv analysis.xlsx --workbook
kikolens report examples/sales_demo.csv --target revenue --nn --output report.html
```

## Recipes and model packages

A **recipe** (`.json`) is the ordered list of cleaning and transformation steps. It has no data in it and can be applied to any file with the same columns. A **model package** (`.klmodel`) holds a trained model together with the columns it expects, their training ranges, its classes and its test metrics. Saved models live in `~/.kikolens/models` (change it with `KIKOLENS_HOME`); saving the same name again creates a new version (`-v2`, `-v3`).

`score` prints warnings before it predicts: columns that are missing, values that could not be read, categories never seen in training and numbers far outside the training range. Regression models also write a likely range (`<target>_low`, `<target>_high`), and classifiers write `confidence` and one probability column per class.

Model packages are pickles: only open files you created or trust.

## Pipeline files

`kikolens run` reads a JSON file that says what to do with a dataset and where to leave the results; paths are relative to the file. Only `source` is required.

```json
{
  "source": "today.csv",
  "recipe": "recipe.json",
  "rules": "rules.json",
  "baseline": "training.csv",
  "model": "sales-model",
  "min_quality": 80,
  "outputs": {"data": "out/clean.csv", "predictions": "out/scored.csv", "report": "out/report.html", "log": "out/log.json"},
  "fail_on": ["rules", "drift", "quality", "input"]
}
```

The stages run in this order: read, apply the recipe, score the quality, validate the rules, compare with the baseline (which also goes through the recipe), score with the model, write the outputs. A problem is only fatal if its name is in `fail_on` (`rules`: a rule is broken; `drift`: the data drifted a lot; `quality`: the score is below `min_quality`; `input`: the model received columns or values it did not see in training); otherwise it is shown as a warning. The `log` output is a JSON record of the run. Typical use is a daily job: `kikolens run pipeline.json || notify-someone`.

## SQL safety

Only a single read-only statement runs. These are rejected: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `ATTACH`, `COPY`, `PRAGMA`, several statements separated by `;`, and file-reading functions such as `read_csv(...)`. Comments and text inside quotes are ignored when checking.

## Environment

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running. |
| `OLLAMA_MODEL` | `llama3.2` | Model for `ask`, summaries and report commentary. |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for the model. |
| `KIKOLENS_HOME` | `~/.kikolens` | Folder for saved models (`models/`) and monitors (`monitors/`). |

## Exit codes

`0` success. `1` a file cannot be loaded, a column does not exist, a query is rejected, a model cannot be trained or found, or a recipe fails; the reason is printed. `2` from `drift` when the data changed a lot, from `validate` when a rule is broken, and from `monitor run` when a check raises an alert.
