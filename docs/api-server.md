# API server

`kikolens api` serves the analysis over HTTP and JSON, so other programs (a web app, a spreadsheet macro, a scheduler, another language) can use it. It runs on your machine, keeps everything in memory and sends nothing anywhere. It needs nothing beyond KikoLens.

```bash
kikolens api                                    # http://127.0.0.1:8000
kikolens api --token s3cret --host 0.0.0.0      # reachable from the network: a token is mandatory
kikolens api --data-dir ./data --model my-model --cors http://localhost:3000
```

Open `/openapi.json` for the machine-readable description and `/functions` for the analyses and their parameters. The older `kikolens serve MODEL` is still there for serving one model only.

## Security

* It listens only on this computer unless you pass another `--host`, and then `--token` is required (`Authorization: Bearer TOKEN`; `/health` is open).
* Clients can only run a fixed list of analyses plus your plugin analyses. They cannot run code, read arbitrary paths or run arbitrary SQL (queries are read-only and cannot open files).
* Files can be loaded only by name from the folder given with `--data-dir`; without it, clients send rows or csv text. Names that escape the folder are refused.
* At most 20 datasets of a million rows are held at once, request bodies are limited to 25 MB, and unexpected errors are reported by type only, never with paths or stack traces.
* `--model` opens a saved model file, which is code: serve only models you trust. Only the models you name are served.

## Endpoints

| Call | What it does |
| :--- | :--- |
| `GET /health` | `{"status": "ok"}`. |
| `GET /functions` | The analyses: name, description, parameters and source (built-in or plugin). |
| `POST /datasets` | Create a dataset from `{"rows": [{...}]}`, `{"csv": "a,b\n1,2"}` or `{"file": "sales.csv"}`; optional `"name"`. Returns its `id`. |
| `GET /datasets`, `GET /datasets/{id}` | List them, or see columns, types, missing values and the first rows. |
| `DELETE /datasets/{id}` | Forget a dataset. |
| `POST /run/{function}` | `{"dataset": id, "args": {...}}`. Runs the analysis and returns `{"result": {"summary": ..., "table": {"columns", "rows", "total_rows"}, ...}, "omitted": [...]}`. Add `?figures=1` for charts as Plotly JSON. Tables are cut to 200 rows. |
| `POST /jobs`, `GET /jobs/{id}` | The same in the background for slow work (`lab`, `autopilot`): `{"function", "dataset", "args"}` returns an id; poll until `status` is `done` or `failed`. |
| `GET /models`, `POST /models/{name}/predict` | The models named at startup; send `{"rows": [...]}` and get predictions with ranges and input warnings. |

## Example

```python
import requests
base = "http://127.0.0.1:8000"
rows = [{"price": 10, "units": 100}, {"price": 12, "units": 90}]  # ...more rows
dataset = requests.post(f"{base}/datasets", json={"rows": rows}).json()["id"]
out = requests.post(f"{base}/run/regress", json={"dataset": dataset, "args": {"target": "units"}}).json()
print(out["result"]["summary"])
```

Status codes: `400` for an invalid request or arguments (the message says why), `401` for a missing or wrong token, `404` for an unknown dataset, function, job or model, `409` when too many datasets are held, `413` for a body that is too large.
