# KikoLens

Private, local-first data analysis. Load a dataset, let the autopilot take a first pass, clean it, explore it, test hypotheses, compare many
models and understand them, forecast, study customers and the effect of a change, watch a data source over time, and keep what you built:
a recipe you can replay, a model you can score new data with, a panel and a report. Nothing leaves your machine; the optional
language-model features use [Ollama](https://ollama.com/) running locally.

![Home page](images/home-light.png)

## Start here

```bash
pip install "kikolens @ git+https://github.com/Francisco-B-O/kikolens.git"
kikolens doctor       # check the installation
kikolens start        # open the dashboard, then pick "Or try an example" or load your own file
```

| If you want to... | Read |
| :--- | :--- |
| Click through the app | [Dashboard guide](dashboard.md) |
| Script it or run it on a schedule | [Command line](cli.md) and [pipelines](cli.md#pipeline-files) |
| Use it from a notebook | [Python API and Jupyter](python-api.md) |
| Follow a complete example | [Workflows](workflows.md) |
| Know how a result is computed and when not to trust it | [How it works](models.md) |
| Change settings (Ollama, port, language) | [Configuration](configuration.md) |
| Contribute | [Development](development.md) |

## What is in the box

* **Data:** home page, data story, chart gallery, quality and cleaning (undoable, as replayable recipes), privacy tools, record matching, transformations, an autopilot and an analyst that chats with your data.
* **Analysis:** statistical tests with effect sizes, sequential and Bayesian A/B tests, anomalies, drift, segmentation, "why did it change", causal effect and survival.
* **Models:** a model lab (cross-validation, tuning, explanations, error analysis, conformal ranges), regression, neural networks, time series, saved models, model health and monitoring.
* **Learn:** a concept lab of simulations and a bilingual glossary with reading guides.
* **Share:** panels, HTML and PDF reports, Excel, notebooks and single-file projects.
* **Advanced:** text, customers, impact, recommendations, surveys, networks, geography, model governance and synthetic data.

![Model lab](images/lab-leaderboard-light.png)
