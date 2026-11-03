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
| `kikolens models delete NAME` | Remove a saved model. |
| `kikolens forecast SOURCE DATE VALUE [--horizon N] [--freq D]` | Forecast a numeric column. Frequencies: `h`, `D`, `W`, `MS`, `QS`, `YS`. |

`predict` and `lab` take `--save PATH` to write a model package to a file and `--register NAME` to keep it with your saved models. `lab` also takes `--folds`, `--no-tune`, `--models "Ridge,Random Forest"`, `--balance` and `--exclude col1,col2`.

## Analysis

| Command | What it does |
| :--- | :--- |
| `kikolens anomalies SOURCE [--method M] [--share 2] [--columns a,b] [--output FILE]` | Flag the most unusual rows and say why. Methods: `Isolation Forest`, `Local Outlier Factor`, `Robust z-score`, `Mahalanobis distance`. |
| `kikolens screen SOURCE TARGET` | Test every column against `TARGET` and list those reliably related to it, corrected for testing many columns. |
| `kikolens ab SOURCE GROUP OUTCOME --control VALUE [--success VALUE]` | Compare a control and a treatment group on rates or averages. |
| `kikolens drift REFERENCE CURRENT [--output FILE]` | Compare two datasets column by column. Exits with code `2` when the data changed a lot, so it works as an alarm in scripts and CI. |
| `kikolens report SOURCE [--target COL] [--nn] [--ai] [--output FILE]` | Build the HTML report. |

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
kikolens report examples/sales_demo.csv --target revenue --nn --output report.html
```

## Recipes and model packages

A **recipe** (`.json`) is the ordered list of cleaning and transformation steps. It has no data in it and can be applied to any file with the same columns. A **model package** (`.klmodel`) holds a trained model together with the columns it expects, their training ranges, its classes and its test metrics. Saved models live in `~/.kikolens/models` (change it with `KIKOLENS_HOME`); saving the same name again creates a new version (`-v2`, `-v3`).

`score` prints warnings before it predicts: columns that are missing, values that could not be read, categories never seen in training and numbers far outside the training range. Regression models also write a likely range (`<target>_low`, `<target>_high`), and classifiers write `confidence` and one probability column per class.

Model packages are pickles: only open files you created or trust.

## SQL safety

Only a single read-only statement runs. These are rejected: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `ATTACH`, `COPY`, `PRAGMA`, several statements separated by `;`, and file-reading functions such as `read_csv(...)`. Comments and text inside quotes are ignored when checking.

## Environment

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running. |
| `OLLAMA_MODEL` | `llama3.2` | Model for `ask`, summaries and report commentary. |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for the model. |
| `KIKOLENS_HOME` | `~/.kikolens` | Folder for saved models. |

## Exit codes

`0` success. `1` a file cannot be loaded, a column does not exist, a query is rejected, a model cannot be trained or found, or a recipe fails; the reason is printed. `2` only from `drift`, when the data changed a lot.
