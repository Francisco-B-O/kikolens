# Contributing to KikoLens

Thanks for wanting to help. This page is the short version of how the project works.

## Set up

```bash
git clone https://github.com/Francisco-B-O/kikolens.git
cd kikolens
pip install -e ".[dev]" ruff
pre-commit install        # optional: runs ruff and basic checks on every commit
kikolens doctor           # checks that the installation is healthy
```

Python 3.12 or newer is required. Optional extras: `[torch]`, `[jupyter]`, `[cloud]`, `[postgres]`, `[mysql]`, `[docs]`.

## Branches and pull requests

The project follows GitFlow.

* `main` holds released versions and `develop` the work in progress. Nothing is pushed straight to either.
* Work on a branch named `feature/...` or `fix/...` created from `develop`, and open a pull request into `develop`.
* A pull request needs the **Tests**, **Lint**, **Build package** and **Documentation** checks green before it is merged.
* Commit messages are short and say what changed: `feat(forecast): ...`, `fix(export): ...`, `docs: ...`.

## Checks to run before you push

```bash
ruff check kikolens tests      # lint: unused imports and variables, undefined names, syntax errors
pytest tests/ -q               # the whole suite (about three minutes)
mkdocs build --strict          # only if you touched docs (pip install mkdocs-material)
```

The suite also runs against pandas 2 and pandas 3, on Linux and Windows, in CI. If you can, try both pandas versions locally: they differ in
three ways that have caused bugs here (copy-on-write makes `.to_numpy()` read-only, datetimes default to microseconds, and strings have their
own `str` dtype, so use `kikolens.utils.dtypes.is_text` instead of `dtype == object`).

## How the code is organised

| Folder | What lives there |
| :--- | :--- |
| `kikolens/analyze/` | The analyses. Pure functions that take a DataFrame and return a dictionary with a `summary` sentence, tables and Plotly figures. No Streamlit here. |
| `kikolens/dashboard/` | The Streamlit app. `views/` has one file per section, `es.py` the Spanish strings, `viz.py` the HTML/SVG pieces. |
| `kikolens/cli/main.py` | The command line (Typer). |
| `kikolens/pipeline/` | Recipes, projects and the pipeline runner. |
| `kikolens/api.py` | The Python API for notebooks. |
| `tests/` | One test file per module, plus dashboard tests with Streamlit's `AppTest`. |

## Adding an analysis

1. Write the function in `kikolens/analyze/yourthing.py`. Validate inputs and raise `ValueError` with a message a person can act on.
   Return a dictionary with at least a `summary`.
2. Write tests that check it does the right thing on data where you know the answer (a planted effect, a known formula), and that
   it refuses bad input. Tests that only check "it runs" are not enough.
3. Add the dashboard view in `kikolens/dashboard/views/`, register it in `app.py`, and add its texts to `es.py`
   (a test fails if a section title is untranslated).
4. Add the command to `kikolens/cli/main.py` and, if it is useful in a notebook, a function in `kikolens/api.py`.
5. Document it in `docs/` (dashboard, cli, models) and add a line to `CHANGELOG.md` under *Unreleased*.

## Principles

* **Local first.** Nothing leaves the machine unless the user asks. No telemetry.
* **Honest results.** Say what a result does not show. Prefer an interval to a single number, and a warning to a silent guess.
* **Safe by default.** No `eval` or `exec` on user input, SQL is validated as read-only, files that can run code (model packages) are only opened on request.
* **Everything undoable and repeatable.** Cleaning and transformation steps go into the recipe.

## Reporting problems

Open an issue with the output of `kikolens doctor --offline`, what you did, what you expected and what happened. For security problems
see [SECURITY.md](SECURITY.md).
