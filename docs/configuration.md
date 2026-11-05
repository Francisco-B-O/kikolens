# Configuration

KikoLens works without any configuration. These are the ways to change its behaviour, from the strongest to the weakest.

1. **Environment variables**
2. **`kikolens.toml`** in the folder you run KikoLens from
3. **`config.toml`** in your KikoLens folder (`~/.kikolens`, or the folder in `KIKOLENS_HOME`)
4. The defaults

A file that is not valid TOML, a value of the wrong type, or a key that is not known never stops KikoLens from starting: it is ignored,
and `kikolens doctor` lists the problem.

```toml
[ollama]
host = "http://localhost:11434"   # where Ollama is listening
model = "llama3.2"                # the model used to write SQL, summaries and commentary
timeout = 120                     # seconds to wait for an answer

[dashboard]
port = 8501                       # `kikolens start` uses this unless you pass --port
language = "es"                   # "en" or "es": the language the dashboard opens in

```

## Environment variables

| Variable | Meaning |
| :--- | :--- |
| `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` | The same as the `[ollama]` keys. |
| `KIKOLENS_HOME` | Where saved models, monitors, run history and `config.toml` live. Default `~/.kikolens`. |

## Check everything

```bash
kikolens doctor             # versions, optional extras, the user folder, Ollama and the configuration
kikolens doctor --offline   # skip the Ollama check
kikolens doctor --json      # machine-readable
```

It exits with code `1` if something required is missing or too old, so it can be used in a script or a container health check.

## Measure speed

```bash
kikolens benchmark --rows 20000
kikolens benchmark --rows 100000 --only "model lab,forecast" --repeat 3 --output bench.json
```

Times profiling, cleaning, SQL, drift, anomaly detection, a forecast, a three-model lab run, the data story and synthetic data on generated
data. Useful to spot a regression and to know what to expect with a big file.

## Optional extras

| Extra | Adds |
| :--- | :--- |
| `kikolens[torch]` | The PyTorch network in the model lab. |
| `kikolens[jupyter]` | JupyterLab, for `kikolens jupyter`. |
| `kikolens[cloud]` | `s3://` and `gs://` sources. |
| `kikolens[postgres]`, `kikolens[mysql]` | Those databases. |
| `kikolens[docs]` | Playwright, to regenerate the screenshots. |

## Shell completion

`kikolens --install-completion` installs tab completion for your shell.
