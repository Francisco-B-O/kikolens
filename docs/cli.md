# Command line reference

Every command accepts a local file, an `http(s)` link or a database URI (`sqlite:///data.db`, `postgresql://user:pass@host/db`) wherever it says `SOURCE`.

| Command | What it does |
| :--- | :--- |
| `kikolens start [--port N]` | Launch the dashboard. |
| `kikolens analyze SOURCE` | Statistics for every column and the data quality summary. `--no-quality` skips the latter. |
| `kikolens sql SOURCE "QUERY"` | Run a read-only SQL query. The data is the table `df`. |
| `kikolens ask SOURCE "QUESTION" [--model NAME]` | A local model writes the SQL, then it runs. Needs Ollama. |
| `kikolens predict SOURCE TARGET` | Train AutoML on `TARGET` and print metrics and top features. |
| `kikolens predict SOURCE TARGET --nn [--layers "64, 32"]` | Same, with a neural network. |
| `kikolens forecast SOURCE DATE VALUE [--horizon N] [--freq D]` | Forecast a numeric column. Frequencies: `h`, `D`, `W`, `MS`, `QS`, `YS`. |
| `kikolens clean SOURCE --output FILE` | Apply the safe automatic cleaning and save as `.csv` or `.parquet`. |
| `kikolens report SOURCE [--target COL] [--nn] [--ai] [--output FILE]` | Build the HTML report. |

`predict` also takes `--save model.pkl` to keep the trained model.

## Examples

```bash
kikolens analyze examples/sales_demo.csv
kikolens sql examples/sales_demo.csv "SELECT region, ROUND(SUM(revenue)) AS total FROM df GROUP BY region ORDER BY total DESC"
kikolens predict examples/sales_demo.csv revenue --nn --layers "64, 32"
kikolens forecast examples/sales_demo.csv date revenue --horizon 30
kikolens clean examples/sales_demo.csv --output clean.parquet
kikolens report examples/sales_demo.csv --target revenue --nn --output report.html
```

## SQL safety

Only a single read-only statement runs. These are rejected: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `ATTACH`, `COPY`, `PRAGMA`, several statements separated by `;`, and file-reading functions such as `read_csv(...)`. Comments and text inside quotes are ignored when checking.

## Environment

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running. |
| `OLLAMA_MODEL` | `llama3.2` | Model for `ask`, summaries and report commentary. |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for the model. |

Exit code is `1` when a file cannot be loaded, a column does not exist, a query is rejected or a model cannot be trained; the reason is printed.
