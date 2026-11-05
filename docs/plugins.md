# Plugins

A plugin adds your own analyses and file formats to KikoLens without touching its code. What you register shows up in the dashboard (section **Plugins**), in `kikolens plugin`, in Python as `kl.run`, in the [API server](api-server.md) and as a tool the [Analyst](dashboard.md) can choose.

!!! warning "A plugin is code"
    Loading a plugin runs its Python with your permissions. KikoLens only loads plugins from places you control: the `plugins` folder in your KikoLens folder, the paths you list in `kikolens.toml`, and Python packages you installed. It never loads one that arrives inside a data file, a project or a request. Only use plugins you wrote or have read. `KIKOLENS_PLUGINS=0` turns them all off.

## Writing one

Create a file in `~/.kikolens/plugins/` (or the `plugins` folder under `KIKOLENS_HOME`) with a `register` function:

```python
import pandas as pd


def register(registry):
    @registry.analysis(
        "margin_by", "Average margin by group.",
        params={
            "group": {"type": "column"},
            "price": "column",
            "cost": "column",
            "top": {"type": "integer", "default": 3, "min": 1, "max": 10},
            "order": {"type": "choice", "choices": ["high", "low"], "default": "high"},
        },
    )
    def margin_by(df, group, price, cost, top, order):
        table = df.assign(margin=df[price] - df[cost]).groupby(group, as_index=False)["margin"].mean()
        table = table.sort_values("margin", ascending=(order == "low")).head(top)
        return {"summary": f"{table.iloc[0][group]} leads on margin.", "table": table}

    registry.loader(".pipe", lambda path: pd.read_csv(path, sep="|"))
```

**Analyses.** The function receives a copy of the data and the checked arguments, and returns a dictionary (or a DataFrame, which becomes its `table`). Keys KikoLens understands: `summary` (one or two sentences; the default is the description), `table` and any other DataFrame, `figure` (a Plotly figure), `notes` (a list of warnings) and plain numbers or text, which the dashboard shows as figures. Raise `ValueError` with a clear message when the data cannot be analysed; any other exception is reported as "the plugin failed".

**Parameters.** Each one is a type or a dictionary with a `type`, an optional `label`, and a `default` (a parameter without a default is required):

| Type | Meaning | Extras |
| :--- | :--- | :--- |
| `column` | One column of the data (case is forgiven). | |
| `columns` | A list of columns. | |
| `number`, `integer` | A finite number, or a whole one. | `min`, `max` |
| `choice` | One of a fixed list. | `choices` (required) |
| `bool` | True or false. | |
| `text` | Free text of at most 500 characters. | The Analyst cannot choose analyses that take free text. |

Names must be lowercase letters, digits and underscores, and cannot reuse the name of a built-in function. Arguments are validated before your function runs, so it can trust them.

**Formats.** `registry.loader(".ext", function)` teaches `kl.load`, `kikolens` commands and the file uploader of the dashboard to read a new extension; the function receives a path and returns a DataFrame.

## Using it

```bash
kikolens plugins                                     # what is loaded, and anything that failed
kikolens plugin margin_by sales.csv --arg group=shop --arg price=price --arg cost=cost --arg top=2
```

```python
import kikolens as kl
res = kl.run("margin_by", df, group="shop", price="price", cost="cost")
res.summary, res.table
kl.list_plugins()          # analyses with parameters, formats, load errors
```

## Where plugins are read from

1. `~/.kikolens/plugins/*.py` (files starting with `_` are skipped).
2. Each path under `[plugins] paths` in `kikolens.toml`, a file or a folder:

    ```toml
    [plugins]
    paths = ["~/work/kikolens-plugins", "/shared/team_plugin.py"]
    ```

3. Installed packages that declare the `kikolens.plugins` entry point, whose target is a `register(registry)` function or a module with one:

    ```toml
    [project.entry-points."kikolens.plugins"]
    my_company = "my_company.kikolens_plugin"
    ```

A plugin that fails to load, has no `register`, or reuses a taken name is listed by `kikolens plugins` and in the dashboard; the others keep working and KikoLens always starts.
