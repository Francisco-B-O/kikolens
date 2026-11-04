import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from kikolens.config import streamlit_flags
from kikolens.utils.loaders import load_data

app = typer.Typer(help="KikoLens: private, local-first data analysis.", no_args_is_help=True)
console = Console()


def _load(source: str) -> pd.DataFrame:
    try:
        return load_data(source)
    except Exception as e:
        console.print(f"[red]Could not load {source}:[/red] {e}")
        raise typer.Exit(code=1)


def _print_frame(df: pd.DataFrame, title: str = "", max_rows: int = 30) -> None:
    table = Table(title=title or None, show_lines=False)
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for _, row in df.head(max_rows).iterrows():
        table.add_row(*["" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row])
    console.print(table)
    if len(df) > max_rows:
        console.print(f"[dim]Showing {max_rows} of {len(df)} rows.[/dim]")


def _check_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        console.print(f"[red]Column '{target}' not found.[/red] Available: {', '.join(map(str, df.columns))}")
        raise typer.Exit(code=1)


def _keep_model(res: dict, df: pd.DataFrame, save: Optional[Path], register: Optional[str]) -> None:
    """Guarda el modelo como paquete reutilizable: en una ruta (--save) y/o en el registro (--register)."""
    from kikolens.analyze.predict import build_bundle, save_bundle

    if not save and not register:
        return
    bundle = build_bundle(res, df, name=register)
    if save:
        console.print(f"Model saved to {save_bundle(bundle, path=save)}")
    if register:
        console.print(f"Model registered as {save_bundle(bundle).name}. Use it with: kikolens score data.csv {register}")


@app.command()
def start(port: int = typer.Option(8501, help="Port for the dashboard.")):
    """Launch the interactive dashboard."""
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    if not app_path.exists():
        console.print(f"[red]Dashboard not found at {app_path}[/red]")
        raise typer.Exit(code=1)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
                        *streamlit_flags()], check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
    except subprocess.CalledProcessError as e:
        raise typer.Exit(code=e.returncode)


@app.command()
def analyze(source: str, quality: bool = typer.Option(True, help="Include the data quality summary.")):
    """Descriptive statistics and data quality for a file, URL or database."""
    from kikolens.analyze.quality import profile_quality
    from kikolens.analyze.stats import basic_statistics

    df = _load(source)
    console.print(f"[bold]{source}[/bold]  {len(df):,} rows × {len(df.columns)} columns")
    _print_frame(basic_statistics(df).round(3).reset_index().rename(columns={"index": "column"}), "Statistics")
    if quality:
        q = profile_quality(df)
        console.print(f"\nQuality score: [bold]{q['score']:.0f}/100[/bold]")
        for issue in q["issues"]:
            console.print(f"  [{issue['severity']}] {issue['message']}")


@app.command()
def sql(source: str, query: str):
    """Run a read-only SQL query. The data is exposed as the table `df`."""
    from kikolens.sql.runner import run_sql_query

    try:
        _print_frame(run_sql_query(_load(source), query))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def ask(source: str, question: str, model: Optional[str] = typer.Option(None, help="Ollama model.")):
    """Ask a question in plain language; a local model writes and runs the SQL."""
    from kikolens.sql.generator import generate_sql_from_text
    from kikolens.sql.runner import run_sql_query

    df = _load(source)
    query = generate_sql_from_text(df, question, model_name=model)
    console.print(f"[dim]{query}[/dim]")
    if query.startswith("-- Error"):
        raise typer.Exit(code=1)
    try:
        _print_frame(run_sql_query(df, query))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def predict(
    source: str,
    target: str,
    nn: bool = typer.Option(False, "--nn", help="Train a neural network instead of AutoML."),
    layers: str = typer.Option("64, 32", help="Hidden layer sizes for --nn."),
    save: Optional[Path] = typer.Option(None, help="Save the model package to this path (.klmodel)."),
    register: Optional[str] = typer.Option(None, help="Also keep it in your saved models under this name."),
):
    """Train a predictive model on a target column."""
    df = _load(source)
    _check_target(df, target)
    try:
        if nn:
            from kikolens.analyze.neural import run_neural_network
            result = run_neural_network(df, target, hidden_layers=layers)
        else:
            from kikolens.analyze.automl import run_automl_analysis
            result = run_automl_analysis(df, target)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    console.print(result["narrative"])
    _print_frame(pd.DataFrame([result["metrics"]]), "Test metrics")
    top = pd.DataFrame(list(result["feature_importance"].items())[:10], columns=["feature", "importance"])
    _print_frame(top, "Top features")
    _keep_model(result, df, save, register)


@app.command()
def lab(
    source: str,
    target: str,
    folds: int = typer.Option(5, help="Cross-validation folds (3-10)."),
    tune: bool = typer.Option(True, help="Tune the best model."),
    models: Optional[str] = typer.Option(None, help="Comma-separated model names; default is all."),
    balance: bool = typer.Option(False, "--balance", help="Weight rarer classes more (classification)."),
    exclude: Optional[str] = typer.Option(None, help="Comma-separated columns to leave out."),
    save: Optional[Path] = typer.Option(None, help="Save the best model package to this path (.klmodel)."),
    register: Optional[str] = typer.Option(None, help="Also keep it in your saved models under this name."),
):
    """Compare many models with cross-validation, tune the best and check for data leaks."""
    from kikolens.analyze.modellab import run_model_lab

    df = _load(source)
    _check_target(df, target)
    names = [m.strip() for m in models.split(",")] if models else None
    drop = [c.strip() for c in exclude.split(",")] if exclude else []
    try:
        with console.status("Comparing models..."):
            res = run_model_lab(df, target, models=names, folds=folds, tune=tune, balance=balance, drop=drop)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    board = res["leaderboard"]
    _print_frame(board[[c for c in board.columns if not c.endswith("std") and c != "total time (s)"]].round(4), "Leaderboard")
    console.print(res["narrative"])
    for leak in res["leakage"]:
        console.print(f"[red]Possible data leak:[/red] {leak['message']}")
    _print_frame(pd.DataFrame(list(res["feature_importance"].items())[:10], columns=["feature", "importance"]).round(4),
                 "Top features")
    _keep_model(res, df, save, register)


@app.command()
def score(
    source: str,
    model: str = typer.Argument(..., help="A model file (.klmodel) or the name of a saved model."),
    output: Path = typer.Option(Path("kikolens_output/predictions.csv"), help="Where to write the predictions."),
    raw: bool = typer.Option(False, "--raw", help="The file is uncleaned: repeat the cleaning steps used in training first."),
):
    """Predict new data with a saved model. Opens a model file, so use only models you trust."""
    from kikolens.analyze.predict import input_warnings, load_bundle, predict_frame, resolve_model

    df = _load(source)
    try:
        bundle = load_bundle(resolve_model(model))
        result, report = predict_frame(bundle, df, apply_recipe=raw)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    for severity, message in input_warnings(report):
        color = {"high": "red", "medium": "yellow"}.get(severity, "dim")
        console.print(f"[{color}]{message}[/{color}]")
    _write_frame(result, output)
    console.print(f"{len(result):,} predictions of '{bundle['target']}' saved to {output}")


models_app = typer.Typer(help="Manage the models you have saved.", no_args_is_help=False)
app.add_typer(models_app, name="models")


@models_app.callback(invoke_without_command=True)
def models_list(ctx: typer.Context):
    """List saved models (run `kikolens models info NAME` for details)."""
    from kikolens.analyze.predict import list_models, models_dir

    if ctx.invoked_subcommand:
        return
    frame = list_models()
    if frame.empty:
        console.print("No saved models yet. Create one with: kikolens lab data.csv target --register my-model")
        return
    _print_frame(frame[["file", "target", "algorithm", "score"]], f"Models in {models_dir()}")


@models_app.command("info")
def models_info(name: str, card: Optional[Path] = typer.Option(None, help="Write the model card (Markdown) to this file.")):
    """Show what a saved model predicts, how well, and which columns it expects."""
    from kikolens.analyze.predict import load_bundle, metadata, model_card, resolve_model

    try:
        bundle = load_bundle(resolve_model(name))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    meta = metadata(bundle)
    console.print(f"[bold]{meta['name']}[/bold]  {meta['label']}  ·  predicts {meta['target']} ({meta['task']})  ·  "
                  f"{meta['rows_trained']:,} training rows")
    _print_frame(pd.DataFrame([meta["metrics"]]), "Test metrics")
    _print_frame(pd.DataFrame(list(meta["inputs"].items()), columns=["input column", "type"]), "Expected inputs")
    if card:
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(model_card(bundle), encoding="utf-8")
        console.print(f"Model card written to {card}")


@models_app.command("delete")
def models_delete(name: str):
    """Delete a saved model."""
    from kikolens.analyze.predict import delete_model

    try:
        console.print(f"Deleted {delete_model(name).name}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def anomalies(
    source: str,
    method: str = typer.Option("Isolation Forest", help="Isolation Forest, Local Outlier Factor, Robust z-score or Mahalanobis distance."),
    share: float = typer.Option(2.0, help="Percentage of rows to flag."),
    columns: Optional[str] = typer.Option(None, help="Comma-separated numeric columns; default is all."),
    output: Optional[Path] = typer.Option(None, help="Write the flagged rows to this .csv or .parquet."),
):
    """Find the rows that do not look like the rest, and say why."""
    from kikolens.analyze.anomalies import detect_anomalies

    df = _load(source)
    cols = [c.strip() for c in columns.split(",")] if columns else None
    try:
        result = detect_anomalies(df, cols, method, share / 100)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(result["summary"])
    _print_frame(result["table"].reset_index(names="row").round(4), "Most unusual rows", max_rows=20)
    if output:
        _write_frame(result["table"].reset_index(names="row"), output)
        console.print(f"Flagged rows saved to {output}")


@app.command()
def drift(reference: str, current: str, output: Optional[Path] = typer.Option(None, help="Write the report to a .csv.")):
    """Compare two datasets and list the columns whose distribution changed."""
    from kikolens.analyze.drift import compare_datasets

    try:
        result = compare_datasets(_load(reference), _load(current))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold]{result['summary']}[/bold]")
    for name, cols in (("Only in the reference", result["only_in_reference"]), ("Only in the current", result["only_in_current"])):
        if cols:
            console.print(f"[yellow]{name}:[/yellow] {', '.join(map(str, cols))}")
    table = result["table"]
    _print_frame(table[["column", "type", "verdict", "psi", "p_value", "change"]].round(4), "Drift by column", max_rows=40)
    if output:
        _write_frame(table, output)
        console.print(f"Report saved to {output}")
    if result["level"] == "high":
        raise typer.Exit(code=2)  # permite usar el comando como alarma en scripts y CI


@app.command()
def screen(source: str, target: str, alpha: float = typer.Option(0.05, help="Significance level.")):
    """Test every column against a target and show which ones are reliably related to it."""
    from kikolens.analyze.stattests import screen_associations

    df = _load(source)
    _check_target(df, target)
    try:
        table = screen_associations(df, target, alpha)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    related = int(table["related"].sum())
    console.print(f"{related} of {len(table)} columns are related to [bold]{target}[/bold] "
                  "(corrected for testing many columns).")
    _print_frame(table.round(5), "Associations", max_rows=40)


@app.command()
def ab(
    source: str, group: str, outcome: str,
    control: str = typer.Option(..., help="The value of the group column that is the control."),
    success: Optional[str] = typer.Option(None, help="For a two-valued outcome, the value that counts as success."),
):
    """Compare a control and a treatment group on an outcome (rates or averages)."""
    from kikolens.analyze.stattests import ab_test

    df = _load(source)
    for col in (group, outcome):
        _check_target(df, col)
    try:
        result = ab_test(df, group, outcome, control, success)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(result["conclusion"])
    if result["needed_per_group"]:
        console.print(f"[dim]Rows per group to reliably detect a gap this size: {result['needed_per_group']:,}[/dim]")


@app.command()
def panel(
    source: str,
    definition: Optional[Path] = typer.Argument(None, help="A panel definition (.json). Without it a panel is suggested from the data."),
    output: Path = typer.Option(Path("kikolens_output/panel.html"), help="Where to write the web page."),
    save: Optional[Path] = typer.Option(None, help="Also write the panel definition to this .json."),
):
    """Build a one-page panel (metrics, charts, tables) from a file."""
    from kikolens.analyze.panel import Panel, panel_html, suggest_panel
    from kikolens.analyze.quality import infer_types

    df, _ = infer_types(_load(source))
    try:
        loaded = Panel.from_json(definition.read_text(encoding="utf-8")) if definition else suggest_panel(df, Path(str(source)).stem)
        _, results = loaded.evaluate(df)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    for tile, _, error in results:
        if error:
            console.print(f"[yellow]{tile.get('title') or tile['type']}: {error}[/yellow]")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(panel_html(df, loaded), encoding="utf-8")
    console.print(f"Panel with {len(loaded.tiles)} tile(s) written to [bold]{output}[/bold]")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(loaded.to_json(), encoding="utf-8")
        console.print(f"Definition saved to {save}")


monitor_app = typer.Typer(help="Watch a data source over time and get alerts.", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")


def _monitor(name: str):
    from kikolens.analyze.monitor import load_monitor

    try:
        return load_monitor(name)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@monitor_app.command("create")
def monitor_create(
    name: str,
    source: str,
    baseline: Optional[str] = typer.Option(None, help="Data to compare against; default is the source as it is now."),
    recipe: Optional[Path] = typer.Option(None, help="Recipe (.json) to apply to every check."),
    model: Optional[str] = typer.Option(None, help="Saved model to score each check with."),
    webhook: Optional[str] = typer.Option(None, help="http(s) address to notify when a check raises an alert."),
    rules: Optional[Path] = typer.Option(None, help="Validation rules (.json) to check on every run."),
    drift_level: str = typer.Option("high", help="Alert when drift reaches this level: medium or high."),
    quality_drop: float = typer.Option(10.0, help="Alert when the quality score falls by this many points."),
):
    """Create a monitor for SOURCE using a baseline."""
    from kikolens.analyze.monitor import create_monitor
    from kikolens.pipeline.recipe import Recipe

    try:
        loaded = Recipe.from_json(recipe.read_text(encoding="utf-8")) if recipe else None
        mon = create_monitor(name, source, _load(baseline) if baseline else None, loaded, model,
                             {"drift_level": drift_level, "quality_drop": quality_drop}, webhook,
                             rules=rules.read_text(encoding="utf-8") if rules else None)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Monitor [bold]{mon.name}[/bold] created. Baseline quality {mon.baseline_quality:.0f}/100. "
                  f"Run it with: kikolens monitor run \"{mon.name}\"")


@monitor_app.command("run")
def monitor_run(name: str, source: Optional[str] = typer.Option(None, help="Check this file instead of the watched source.")):
    """Run one check. Exit code 0 when fine, 2 on alerts, 1 when the data could not be read."""
    from kikolens.analyze.monitor import run_check

    mon = _monitor(name)
    record = run_check(mon, _load(source) if source else None)
    colour = {"ok": "green", "warning": "yellow", "alert": "red", "failed": "red"}[record["status"]]
    console.print(f"[{colour}]{record['status'].upper()}[/{colour}] {mon.name}: {record.get('rows', 0):,} rows")
    for a in record["alerts"]:
        console.print(f"  [{a['severity']}] {a['message']}")
    raise typer.Exit(code={"ok": 0, "warning": 0, "alert": 2, "failed": 1}[record["status"]])


@monitor_app.command("list")
def monitor_list():
    """List the monitors."""
    from kikolens.analyze.monitor import list_monitors

    frame = list_monitors()
    if frame.empty:
        console.print("No monitors yet. Create one with: kikolens monitor create NAME SOURCE")
        return
    _print_frame(frame, "Monitors")


@monitor_app.command("history")
def monitor_history(name: str, last: int = typer.Option(20, help="How many checks to show.")):
    """Show the past checks of a monitor."""
    from kikolens.analyze.monitor import history

    _monitor(name)
    frame = history(name)
    if frame.empty:
        console.print("Nothing checked yet.")
        return
    _print_frame(frame.tail(last).round(3), f"History of {name}", max_rows=last)


@monitor_app.command("watch")
def monitor_watch(name: str, interval: float = typer.Option(300.0, help="Seconds between looks at the file.")):
    """Keep running and check every time the watched file changes. Stop with Ctrl+C."""
    from kikolens.analyze.monitor import watch

    mon = _monitor(name)
    console.print(f"Watching {mon.source} every {interval:g}s. Ctrl+C to stop.")

    def show(record):
        console.print(f"{record['time']} {record['status'].upper()} ({len(record['alerts'])} alert(s))")

    try:
        watch(mon, interval, on_run=show)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@monitor_app.command("schedule")
def monitor_schedule(name: str, every: str = typer.Option("daily", help="hourly, daily or weekly.")):
    """Print the cron and Windows commands that run the check on a schedule (nothing is installed)."""
    from kikolens.analyze.monitor import schedule_hint

    try:
        console.print(schedule_hint(_monitor(name), every), markup=False)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@monitor_app.command("delete")
def monitor_delete(name: str):
    """Delete a monitor and its history."""
    from kikolens.analyze.monitor import delete_monitor

    _monitor(name)
    delete_monitor(name)
    console.print(f"Deleted {name}")


@app.command()
def text(
    source: str,
    column: str,
    group: Optional[str] = typer.Option(None, help="Also show the terms that distinguish each group of this column."),
    ngram: int = typer.Option(1, help="1 words, 2 pairs of words, 3 triples."),
    themes: int = typer.Option(0, help="Group the texts into this many themes (0 skips it)."),
    top: int = typer.Option(15, help="Terms to list."),
):
    """Key terms, groups and themes of a free-text column."""
    from kikolens.analyze import text as tx

    df = _load(source)
    _check_target(df, column)
    try:
        info = tx.text_stats(df[column])
        console.print(f"{info['rows'] - info['empty']:,} texts · {info['mean_words']:.1f} words each · language {info['language']}")
        _print_frame(tx.top_terms(df[column], top, ngram).round(4), "Most common terms")
        if group:
            _check_target(df, group)
            _print_frame(tx.terms_by_group(df, column, group).round(3), f"Terms that distinguish each {group}", max_rows=40)
        if themes:
            _print_frame(tx.cluster_texts(df[column], themes)["summary"].round(3), "Themes")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def validate(
    source: str,
    rules: Optional[Path] = typer.Argument(None, help="A rules file (.json). Without it the rules suggested by the data are used."),
    save: Optional[Path] = typer.Option(None, help="Write the rules used to this .json."),
):
    """Check a dataset against validation rules. Exit code 2 when a rule is broken."""
    from kikolens.analyze.quality import infer_types
    from kikolens.analyze.rules import RuleSet, describe, suggest_rules

    df, _ = infer_types(_load(source))
    try:
        rule_set = RuleSet.from_json(rules.read_text(encoding="utf-8")) if rules else suggest_rules(df)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    result = rule_set.validate(df)
    _print_frame(result["table"].drop(columns="id").round(2), "Rules", max_rows=100)
    console.print(f"{result['summary']} {result['rows_with_problems']:,} row(s) have at least one problem.")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(rule_set.to_json(), encoding="utf-8")
        console.print(f"Rules saved to {save}")
    if not result["passed"]:
        raise typer.Exit(code=2)


@app.command()
def export(
    source: str,
    output: Path = typer.Argument(..., help="Where to write: .csv, .tsv, .xlsx, .json, .ndjson, .parquet or .feather."),
    recipe: Optional[Path] = typer.Option(None, help="Apply this recipe before exporting."),
    workbook: bool = typer.Option(False, "--workbook", help="For .xlsx: add quality, statistics and recipe sheets."),
):
    """Convert a dataset to another format, optionally as an analysis workbook."""
    from kikolens.pipeline.recipe import Recipe

    df = _load(source)
    loaded = None
    try:
        if recipe:
            loaded = Recipe.from_json(recipe.read_text(encoding="utf-8"))
            df, _ = loaded.apply(df)
        if workbook:
            if output.suffix.lower() != ".xlsx":
                raise ValueError("--workbook needs an output ending in .xlsx")
            from kikolens.report.workbook import build_analysis_workbook

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(build_analysis_workbook(df, Path(str(source)).stem, loaded))
        else:
            from kikolens.utils.writers import write_frame

            write_frame(df, output)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"{len(df):,} rows × {len(df.columns)} columns written to [bold]{output}[/bold]")


@app.command("to-db")
def to_db(
    source: str,
    database: str = typer.Argument(..., help="Database address, e.g. sqlite:///results.db or postgresql://user:pass@host/db"),
    table: str = typer.Argument(..., help="Table to write."),
    if_exists: str = typer.Option("fail", help="fail (default), append or replace."),
):
    """Write a dataset into a database table. It never overwrites unless you say `--if-exists replace`."""
    from kikolens.utils.writers import write_database

    df = _load(source)
    try:
        n = write_database(df, database, table, if_exists)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"{n:,} rows written to table [bold]{table}[/bold].")


@app.command()
def forecast(
    source: str, date: str, value: str,
    horizon: int = typer.Option(12, help="Periods to forecast."),
    freq: Optional[str] = typer.Option(None, help="Resample frequency: h, D, W, MS, QS, YS."),
    compare: bool = typer.Option(False, "--compare", help="Compare several models with rolling-origin tests and use the best."),
    changes: bool = typer.Option(False, "--changes", help="List level shifts and trend changes in the series."),
    by: Optional[str] = typer.Option(None, "--by", help="Forecast each value of this column separately."),
    agg: str = typer.Option("mean", help="How to combine rows in the same period: mean, sum, median, min, max, count."),
):
    """Forecast a numeric column over time."""
    from kikolens.analyze import forecasting as fx
    from kikolens.analyze import timeseries as ts

    df = _load(source)
    try:
        if by:
            _check_target(df, by)
            res = fx.forecast_by_group(df, date, value, by, horizon, freq, agg)
            _print_frame(res["summary"].round(3), f"Forecast by {by}", max_rows=40)
            for _, row in res["skipped"].iterrows():
                console.print(f"[yellow]Skipped {row['group']}: {row['reason']}[/yellow]")
            _print_frame(res["forecast"].round(3), "Forecast", max_rows=40)
            return
        series = ts.prepare_series(df, date, value, freq=freq, agg=agg)
        if compare:
            best = fx.forecast_models(series, horizon)
            console.print(best["summary"])
            _print_frame(best["leaderboard"].round(4), "Model comparison")
            _print_frame(best["forecast"].round(4).rename_axis("date").reset_index(), "Forecast")
        else:
            result = ts.forecast(series, horizon=horizon)
            summary = ts.trend_summary(series)
            console.print(f"Trend: [bold]{summary['direction']}[/bold]  ·  seasonal period: {result['period'] or 'none'}")
            _print_frame(pd.DataFrame([result["metrics"]]), "Backtest")
            _print_frame(result["forecast"].rename_axis("date").reset_index(), "Forecast")
        if changes:
            points = fx.detect_changepoints(series)
            if points:
                _print_frame(pd.DataFrame(points).round(3), "Changes in the series")
            else:
                console.print("No level shifts or trend changes found.")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def clean(
    source: str,
    output: Path = typer.Option(..., help="Where to write the cleaned file (.csv or .parquet)."),
    save_recipe: Optional[Path] = typer.Option(None, help="Also save the cleaning steps as a recipe (.json)."),
):
    """Apply safe automatic cleaning and save the result."""
    from kikolens.analyze.quality import auto_clean_steps, profile_quality
    from kikolens.pipeline.recipe import Recipe

    df = _load(source)
    before = profile_quality(df)["score"]
    cleaned, steps = auto_clean_steps(df)
    for _, _, message in steps:
        console.print(f"  • {message}")
    _write_frame(cleaned, output)
    if save_recipe:
        recipe = Recipe()
        for op, params, message in steps:
            recipe.add(op, params, message)
        save_recipe.parent.mkdir(parents=True, exist_ok=True)
        save_recipe.write_text(recipe.to_json(), encoding="utf-8")
        console.print(f"Recipe saved to {save_recipe}")
    console.print(f"Quality score {before:.0f} → {profile_quality(cleaned)['score']:.0f}. Saved to {output}")


def _write_frame(df: pd.DataFrame, output: Path) -> None:
    from kikolens.utils.writers import write_frame

    try:
        write_frame(df, output)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def apply(
    source: str,
    recipe: Path = typer.Argument(..., help="Recipe (.json) saved from the dashboard or `clean --save-recipe`."),
    output: Path = typer.Option(..., help="Where to write the result (.csv or .parquet)."),
    dataset: Optional[list[str]] = typer.Option(None, "--dataset", "-d", help="Extra dataset for join steps: a path, or name=path."),
):
    """Repeat a saved recipe on a new file."""
    from kikolens.pipeline.recipe import Recipe

    try:
        loaded = Recipe.from_json(recipe.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not read the recipe:[/red] {e}")
        raise typer.Exit(code=1)
    extras = {}
    for item in dataset or []:
        name, sep, path = item.partition("=")
        if not sep:  # sin nombre: se usa el del archivo, que es como el dashboard registra los datasets que se suben
            name, path = Path(item).name, item
        extras[name] = _load(path)
    try:
        result, log = loaded.apply(_load(source), extras)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    for line in log:
        console.print(f"  • {line}")
    _write_frame(result, output)
    console.print(f"{len(result):,} rows × {len(result.columns)} columns saved to {output}")


@app.command()
def project(file: Path, output: Optional[Path] = typer.Option(None, help="Write the processed data (.csv or .parquet).")):
    """Show what a .kikolens project contains, and optionally export its processed data."""
    from kikolens.pipeline.project import load_project

    try:
        content = load_project(file.read_bytes())
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    info = content["manifest"]
    console.print(f"[bold]{info['name']}[/bold]  saved {info['created']}  ·  {info['rows']:,} rows × {len(info['columns'])} columns")
    if content["notes"]:
        console.print(f"Notes: {content['notes']}")
    steps = content["recipe"].describe()
    if len(steps):
        _print_frame(steps, "Recipe")
    if output:
        try:
            result, _ = content["recipe"].apply(content["original"], content["datasets"])
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)
        _write_frame(result, output)
        console.print(f"Processed data saved to {output}")


@app.command()
def report(
    source: str,
    output: Path = typer.Option(Path("kikolens_output/report.html"), help="Report path."),
    target: Optional[str] = typer.Option(None, help="Include AutoML for this column."),
    nn: bool = typer.Option(False, "--nn", help="Also train a neural network for the target."),
    ai: bool = typer.Option(False, "--ai", help="Add commentary from the local LLM."),
    pdf: Optional[Path] = typer.Option(None, "--pdf", help="Also write a PDF report to this path."),
):
    """Build a self-contained HTML report, and optionally a PDF."""
    from kikolens.report.builder import save_report

    df = _load(source)
    models = {}
    try:
        if target:
            _check_target(df, target)
            from kikolens.analyze.automl import run_automl_analysis
            models["AutoML"] = run_automl_analysis(df, target)
            if nn:
                from kikolens.analyze.neural import run_neural_network
                models["Neural network"] = run_neural_network(df, target)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    ai_text = None
    if ai:
        from kikolens.ai.insights import get_ai_insights
        ai_text = get_ai_insights(df)
    path = save_report(df, output, name=Path(str(source)).stem or "dataset", models=models, ai_text=ai_text)
    console.print(f"Report written to [bold]{path}[/bold]")
    if pdf:
        from kikolens.report.pdf import build_pdf
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(build_pdf(df, name=Path(str(source)).stem or "dataset"))
        console.print(f"PDF written to [bold]{pdf}[/bold]")


@app.command()
def autopilot(
    source: str,
    target: Optional[str] = typer.Option(None, help="Column to predict; guessed when omitted."),
    horizon: int = typer.Option(12, help="Periods to forecast when the data has dates."),
    thorough: bool = typer.Option(False, "--thorough", help="Compare every model and tune the best (slower)."),
    clean: Optional[Path] = typer.Option(None, help="Write the cleaned data to this file."),
    recipe: Optional[Path] = typer.Option(None, help="Write the cleaning steps as a recipe (JSON)."),
    pdf: Optional[Path] = typer.Option(None, "--pdf", help="Write a PDF report."),
):
    """Analyze a dataset end to end: clean, find patterns, model a column and forecast dates."""
    from kikolens.analyze.autopilot import next_steps, run_autopilot

    df = _load(source)
    if target:
        _check_target(df, target)
    try:
        with console.status("Working..."):
            result = run_autopilot(df, target, forecast_horizon=horizon, fast=not thorough)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(result["narrative"])
    for line in result["steps"]:
        console.print(f"  - {line}")
    for line in result["problems"]:
        console.print(f"[yellow]  - {line}[/yellow]")
    if result["model"]:
        _print_frame(result["model"]["leaderboard"][["model", result["model"]["primary_metric"]]], "Models")
    console.print("[bold]Next[/bold]")
    for line in next_steps(result):
        console.print(f"  - {line}")
    if clean:
        clean.parent.mkdir(parents=True, exist_ok=True)
        result["clean"].to_csv(clean, index=False)
        console.print(f"Cleaned data written to {clean}")
    if recipe:
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(result["recipe"].to_json(), encoding="utf-8")
        console.print(f"Recipe written to {recipe}")
    if pdf:
        from kikolens.report.pdf import build_pdf
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(build_pdf(result["clean"], name=Path(str(source)).stem or "dataset", autopilot=result))
        console.print(f"PDF written to {pdf}")


def _fail(error: Exception):
    console.print(f"[red]{error}[/red]")
    raise typer.Exit(code=1)


@app.command()
def survival(source: str, duration: str, event: str, group: Optional[str] = typer.Option(None, help="Compare these groups.")):
    """Kaplan-Meier survival curves and a log-rank test between groups."""
    from kikolens.analyze.survival import survival_by_group

    df = _load(source)
    try:
        res = survival_by_group(df, duration, event, group)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


@app.command()
def cohorts(source: str, user: str, date: str, period: str = typer.Option("Month", help="Day, Week or Month.")):
    """Retention by cohort: the share of each group of new users still active later."""
    from kikolens.analyze.customers import cohort_retention

    df = _load(source)
    try:
        res = cohort_retention(df, user, date, period)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame((res["matrix"] * 100).round(0).reset_index().rename(columns={"cohort": "cohort"}), "Retention (%)")


@app.command()
def rfm(source: str, customer: str, date: str, amount: str, output: Optional[Path] = typer.Option(None, help="Write every customer with their segment.")):
    """Segment customers by recency, frequency and spend."""
    from kikolens.analyze.customers import rfm as run_rfm

    df = _load(source)
    try:
        res = run_rfm(df, customer, date, amount)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["segments"].round(2), "Segments")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        res["customers"].to_csv(output, index=False)
        console.print(f"Customers written to {output}")


@app.command()
def basket(source: str, transaction: str, item: str, support: float = typer.Option(0.02, help="Minimum share of baskets."),
           confidence: float = typer.Option(0.3, help="Minimum confidence.")):
    """Association rules: what is bought together."""
    from kikolens.analyze.customers import association_rules

    df = _load(source)
    try:
        res = association_rules(df, transaction, item, support, confidence)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["rules"].round(3), "Rules")


@app.command()
def impact(
    source: str, date: str, value: str, at: str = typer.Option(..., help="When the change happened (YYYY-MM-DD)."),
    control: Optional[str] = typer.Option(None, help="Comma-separated control columns the change should not have touched."),
    agg: str = typer.Option("mean", help="mean or sum, for rows that share a period."),
):
    """Estimate the effect of a change by comparing with what would have happened."""
    from kikolens.analyze import timeseries as ts
    from kikolens.analyze.causal import causal_impact

    df = _load(source)
    try:
        series = ts.prepare_series(df, date, value, agg=agg)
        frame = None
        if control:
            frame = pd.concat([ts.prepare_series(df, date, c.strip(), agg=agg).rename(c.strip()) for c in control.split(",")], axis=1)
        res = causal_impact(series, pd.Timestamp(at), frame)
    except (ValueError, KeyError) as e:
        _fail(e)
    console.print(res["summary"])


@app.command()
def features(
    source: str, target: str, apply: Optional[Path] = typer.Option(None, help="Write the data with the chosen columns added."),
    recipe: Optional[Path] = typer.Option(None, help="Write the added columns as a recipe (JSON)."),
    max_new: int = typer.Option(5, help="Most new columns to add."),
):
    """Search for ratios, products and other new columns that improve a model."""
    from kikolens.analyze.autofeatures import apply_suggestions, suggest_features
    from kikolens.pipeline.recipe import Recipe

    df = _load(source)
    _check_target(df, target)
    try:
        with console.status("Trying ideas..."):
            res = suggest_features(df, target, max_new=max_new)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if res["suggestions"]:
        _print_frame(pd.DataFrame(res["suggestions"])[["name", "expression", "gain"]].round(4), "New columns")
        new, steps = apply_suggestions(df, res["suggestions"])
        if apply:
            apply.parent.mkdir(parents=True, exist_ok=True)
            new.to_csv(apply, index=False)
            console.print(f"Data written to {apply}")
        if recipe:
            r = Recipe()
            for op, params, message in steps:
                r.add(op, params, message)
            recipe.write_text(r.to_json(), encoding="utf-8")
            console.print(f"Recipe written to {recipe}")


@app.command("ab-bayes")
def ab_bayes(source: str, group: str, outcome: str, control: str, success: Optional[str] = typer.Option(None, help="Value that counts as success.")):
    """Bayesian A/B test: the chance a variant is better and what a wrong choice would cost."""
    from kikolens.analyze.bayes import bayesian_ab

    df = _load(source)
    try:
        res = bayesian_ab(df, group, outcome, control, success=success)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


def cli():
    app()


if __name__ == "__main__":
    cli()
