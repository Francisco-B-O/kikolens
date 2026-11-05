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
def start(port: Optional[int] = typer.Option(None, help="Port for the dashboard (default 8501, or [dashboard] port in kikolens.toml).")):
    """Launch the interactive dashboard."""
    from kikolens.config import dashboard_port

    port = port or dashboard_port()
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
    bayes: bool = typer.Option(False, "--bayes", help="Tune with Bayesian optimisation instead of a random search."),
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
            res = run_model_lab(df, target, models=names, folds=folds, tune=tune, balance=balance, drop=drop, tuner="bayes" if bayes else "random")
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
    try:
        from kikolens.analyze import runs as rn
        rn.log_lab(res, str(source), {"folds": folds, "tune": tune, "tuner": "bayes" if bayes else "random", "balance": balance})
    except OSError:
        pass


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


@app.command()
def serve(
    model: str = typer.Argument(..., help="A model file (.klmodel) or the name of a saved model."),
    port: int = typer.Option(8000, help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", help="Address to listen on. Anything but this computer needs --token."),
    token: Optional[str] = typer.Option(None, help="Require this bearer token on every request."),
):
    """Serve a saved model as a small JSON API (POST /predict). Opens a model file, so use only models you trust."""
    from kikolens.analyze.predict import load_bundle, resolve_model
    from kikolens.serve import make_server

    try:
        bundle = load_bundle(resolve_model(model))
        server = make_server(bundle, host, port, token)
    except (ValueError, OSError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Serving '{bundle['target']}' at http://{host}:{server.server_address[1]}  (POST /predict, GET /model). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


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


@models_app.command("compare")
def models_compare(
    source: str, models: list[str] = typer.Argument(..., help="Two or more saved model names or .klmodel files."),
    raw: bool = typer.Option(False, "--raw", help="The file is uncleaned: repeat the cleaning steps used in training first."),
):
    """Which of several saved models predicts better on data whose real results are known, and is the gap reliable?"""
    from kikolens.analyze.champion import compare_saved_models
    from kikolens.analyze.predict import load_bundle, resolve_model

    df = _load(source)
    try:
        bundles = {Path(m).stem if Path(m).suffix else m: load_bundle(resolve_model(m)) for m in models}
        res = compare_saved_models(bundles, df, raw)
    except ValueError as e:
        _fail(e)
    _print_frame(res["table"].round(4), "Models")
    console.print(res["summary"])


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
    output: Optional[Path] = typer.Option(None, help="Where to write the file (default: kikolens_output/panel.html, .pptx or .docx)."),
    save: Optional[Path] = typer.Option(None, help="Also write the panel definition to this .json."),
    format: str = typer.Option("html", "--format", help="html, pptx (PowerPoint) or docx (Word). The last two need kikolens[office]."),
    where: list[str] = typer.Option([], "--where", help='Keep only matching rows in every tile, such as "region in North,West". Repeat it.'),
):
    """Build a one-page panel (metrics, charts, tables) from a file, as a web page, a PowerPoint or a Word document."""
    from kikolens.analyze.panel import Panel, panel_html, parse_filter, suggest_panel
    from kikolens.analyze.quality import infer_types

    if format not in ("html", "pptx", "docx"):
        _fail(ValueError("The format must be html, pptx or docx."))
    df, _ = infer_types(_load(source))
    try:
        loaded = Panel.from_json(definition.read_text(encoding="utf-8")) if definition else suggest_panel(df, Path(str(source)).stem)
        loaded.filters += [parse_filter(w) for w in where]
        _, results = loaded.evaluate(df)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    for tile, _, error in results:
        if error:
            console.print(f"[yellow]{tile.get('title') or tile['type']}: {error}[/yellow]")
    output = output or Path(f"kikolens_output/panel.{format}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if format == "html":
        output.write_text(panel_html(df, loaded), encoding="utf-8")
    else:
        from kikolens.report import office
        try:
            output.write_bytes((office.panel_pptx if format == "pptx" else office.panel_docx)(df, loaded))
        except ImportError as e:
            _fail(e)
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
    from kikolens.analyze.rules import RuleSet, suggest_rules

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


def _period(text: Optional[str], name: str):
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        _fail(ValueError(f"{name} must be two dates separated by a comma, such as 2024-03-01,2024-05-01 (the end is not included)."))
    return parts[0], parts[1]


@app.command()
def why(
    source: str, date: str,
    metric: Optional[str] = typer.Option(None, help="Numeric column to measure (not needed with --agg count)."),
    agg: str = typer.Option("sum", help="sum, mean or count."),
    before: Optional[str] = typer.Option(None, help="Earlier period as START,END. Default: the middle third of the dates."),
    after: Optional[str] = typer.Option(None, help="Later period as START,END. Default: the last third of the dates."),
    by: Optional[str] = typer.Option(None, help="Comma-separated columns to break the change down by."),
):
    """Explain why a figure changed between two periods: which values of which columns account for it."""
    from kikolens.analyze.drivers import why_changed

    df = _load(source)
    try:
        res = why_changed(df, metric, date, _period(before, "--before"), _period(after, "--after"),
                          [c.strip() for c in by.split(",")] if by else None, agg)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["ranking"].round(3), "Which column explains it")
    top = res["ranking"]["dimension"].iloc[0]
    _print_frame(res["tables"][top].round(3), f"By {top}")


@app.command()
def regress(
    source: str, target: str,
    using: Optional[str] = typer.Option(None, help="Comma-separated columns to explain the target with; default all others."),
    positive: Optional[str] = typer.Option(None, help="For a two-valued target, the value whose chance is modelled."),
    output: Optional[Path] = typer.Option(None, help="Write the coefficient table to this CSV."),
):
    """Explain what drives a column: linear regression for a number, logistic for two values, with ranges and p-values."""
    from kikolens.analyze.regression import logit, ols

    df = _load(source)
    _check_target(df, target)
    features = [c.strip() for c in using.split(",")] if using else None
    try:
        two = df[target].nunique(dropna=True) == 2
        res = logit(df, target, features, positive) if two else ols(df, target, features)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for line in res["sentences"]:
        console.print(f"  - {line}")
    for line in res["flags"]:
        console.print(f"[yellow]  ! {line}[/yellow]")
    keep = ["term", "coefficient", "std error", "p-value", "low", "high"] + (["odds ratio"] if two else [])
    _print_frame(res["table"][keep].round(4), "Coefficients")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        res["table"].to_csv(output, index=False)
        console.print(f"Table written to {output}")


@app.command()
def spikes(
    source: str, date: str, value: str,
    agg: str = typer.Option("mean", help="mean or sum, for rows that share a period."),
    sensitivity: float = typer.Option(3.5, help="Robust z-score threshold; lower finds more."),
    output: Optional[Path] = typer.Option(None, help="Write the unusual periods to this CSV."),
):
    """Find unusual periods in a series, once trend and season are taken out."""
    from kikolens.analyze.tsanomaly import from_frame

    df = _load(source)
    try:
        res = from_frame(df, date, value, agg=agg, threshold=sensitivity)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if len(res["anomalies"]):
        _print_frame(res["anomalies"].round(3), "Unusual periods")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        res["anomalies"].to_csv(output, index=False)
        console.print(f"Written to {output}")


@app.command()
def performance(
    source: str,
    model: str = typer.Argument(..., help="A model file (.klmodel) or the name of a saved model."),
    date: Optional[str] = typer.Option(None, help="Date column, to score period by period."),
    period: str = typer.Option("Month", help="Day, Week, Month or Quarter."),
    raw: bool = typer.Option(False, "--raw", help="The file is uncleaned: repeat the cleaning steps used in training first."),
):
    """Is a saved model still good? Compare its real accuracy on newer data with what it had in testing. Exit code 2 on a warning."""
    from kikolens.analyze.perfwatch import performance_over_time
    from kikolens.analyze.predict import load_bundle, resolve_model

    df = _load(source)
    try:
        bundle = load_bundle(resolve_model(model))
        res = performance_over_time(bundle, df, date, period, apply_recipe=raw)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].drop(columns=["enough rows"]).round(4), "By period")
    for line in res["alerts"]:
        console.print(f"[yellow]  ! {line}[/yellow]")
    if res["status"] == "alert":
        raise typer.Exit(code=2)


def _names(text: Optional[str]) -> list:
    return [c.strip() for c in text.split(",") if c.strip()] if text else []


@app.command()
def privacy(
    source: str,
    scan: bool = typer.Option(False, "--scan", help="List the columns that look like personal data."),
    mask: Optional[str] = typer.Option(None, help="Comma-separated columns to mask, keeping the last 2 characters."),
    hash: Optional[str] = typer.Option(None, help="Comma-separated columns to replace with salted SHA-256 fingerprints."),
    drop: Optional[str] = typer.Option(None, help="Comma-separated columns to remove."),
    salt: str = typer.Option("", help="Salt for --hash. It is stored in the recipe, so keep the recipe private."),
    output: Optional[Path] = typer.Option(None, help="Write the anonymised data (.csv or .parquet)."),
    recipe: Optional[Path] = typer.Option(None, help="Write the steps as a recipe (JSON)."),
    k: Optional[int] = typer.Option(None, help="Check k-anonymity: the smallest group a person can hide in."),
    quasi: Optional[str] = typer.Option(None, help="Columns that describe a person, for --k (default: guessed from names)."),
):
    """Find personal data, mask or hash it, and measure how identifiable people are. Everything stays on this computer."""
    from kikolens.analyze import privacy as pv
    from kikolens.pipeline.recipe import Recipe

    df = _load(source)
    if scan or not (mask or hash or drop or k):
        found = pv.scan(df)
        console.print("No personal data found." if found.empty else f"{len(found)} column(s) look like personal data.")
        if not found.empty:
            _print_frame(found.round(2), "Personal data")
    plan = {**{c: "mask" for c in _names(mask)}, **{c: "hash" for c in _names(hash)}, **{c: "drop" for c in _names(drop)}}
    result = df
    if plan:
        for column in plan:
            _check_target(df, column)
        try:
            result, steps = pv.anonymize(df, plan, salt)
        except ValueError as e:
            _fail(e)
        for _, _, message in steps:
            console.print(f"  - {message}")
        if output:
            _write_frame(result, output)
            console.print(f"Anonymised data written to {output}")
        if recipe:
            r = Recipe()
            for op, params, message in steps:
                r.add(op, params, message)
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(r.to_json(), encoding="utf-8")
            console.print(f"Recipe written to {recipe}")
    if k:
        columns = _names(quasi) or pv.suggest_quasi_identifiers(result)
        try:
            check = pv.k_anonymity(result, columns, k)
        except ValueError as e:
            _fail(e)
        console.print(f"Columns used: {', '.join(columns)}")
        console.print(check["summary"])
        _print_frame(check["smallest_groups"], "The smallest groups")


@app.command()
def notebook(
    source: str, output: Path = typer.Argument(..., help="Where to write the notebook (.ipynb)."),
    recipe: Optional[Path] = typer.Option(None, help="Include these cleaning steps (a recipe JSON)."),
    target: Optional[str] = typer.Option(None, help="Add a model for this column."),
    exclude: Optional[str] = typer.Option(None, help="Comma-separated columns to leave out of the model."),
    date: Optional[str] = typer.Option(None, help="With --value, add a forecast."),
    value: Optional[str] = typer.Option(None, help="Numeric column to forecast."),
    horizon: int = typer.Option(12, help="Periods to forecast."),
):
    """Write a Jupyter notebook that reproduces the analysis."""
    from kikolens.pipeline.recipe import Recipe
    from kikolens.report.notebook import build_notebook

    if output.suffix.lower() != ".ipynb":
        console.print("[red]The output must end in .ipynb[/red]")
        raise typer.Exit(code=1)
    try:
        loaded = Recipe.from_json(recipe.read_text(encoding="utf-8")) if recipe else None
    except (OSError, ValueError) as e:
        _fail(e)
    if target or date or value:
        df = _load(source)
        for column in filter(None, [target, date, value]):
            _check_target(df, column)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_notebook(source, loaded, target, _names(exclude), date, value, horizon), encoding="utf-8")
    console.print(f"Notebook written to {output}")


@app.command()
def dictionary(source: str, output: Optional[Path] = typer.Option(None, help="Write the dictionary (.md, .csv or .xlsx).")):
    """Describe every column: meaning, empty share, typical values, ranges and warnings."""
    from kikolens.analyze.dictionary import build_dictionary, to_markdown

    df = _load(source)
    table = build_dictionary(df)
    if not output:
        _print_frame(table[["column", "meaning", "empty %", "distinct", "typical values", "notes"]], "Data dictionary", max_rows=100)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".md":
        output.write_text(to_markdown(table, Path(str(source)).stem, len(df)), encoding="utf-8")
    elif suffix in (".csv", ".xlsx"):
        _write_frame(table, output)
    else:
        console.print("[red]The output must end in .md, .csv or .xlsx[/red]")
        raise typer.Exit(code=1)
    console.print(f"Dictionary written to {output}")


@app.command()
def link(
    left: str, right: str, left_on: str = typer.Argument(..., help="Name column in the first file."),
    right_on: str = typer.Argument(..., help="Name column in the second file."),
    threshold: float = typer.Option(0.85, help="Minimum similarity, 0.5 to 1."),
    output: Optional[Path] = typer.Option(None, help="Write the first file with the matching columns of the second."),
):
    """Match records whose names are not spelled the same way (accents, order, typos)."""
    from kikolens.analyze.linkage import link_records, merge_linked

    a, b = _load(left), _load(right)
    try:
        res = link_records(a, b, left_on, right_on, threshold)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    inexact = res["matches"][~res["matches"]["exact"]]
    if len(inexact):
        _print_frame(inexact.drop(columns=["left row", "right row", "exact"]), "Matches that are not exact", max_rows=20)
    if output:
        _write_frame(merge_linked(a, b, res["matches"]), output)
        console.print(f"Linked data written to {output}")


@app.command()
def run(pipeline: Path = typer.Argument(..., help="A pipeline file (.json). See docs/cli.md.")):
    """Run a pipeline file: recipe, validation, drift against a baseline and scoring. Exit code 2 when a fail_on condition triggers."""
    from kikolens.pipeline.runner import run_pipeline

    try:
        result = run_pipeline(pipeline)
    except (ValueError, OSError) as e:
        _fail(e)
    colors = {"ok": "green", "warning": "yellow", "alert": "red"}
    for step in result["steps"]:
        console.print(f"[{colors[step['status']]}]{step['status']:>7}[/{colors[step['status']]}]  {step['step']}: {step['message']}")
    for path in result["written"]:
        console.print(f"  wrote {path}")
    if result["status"] == "alert":
        console.print(f"[red]{len(result['alerts'])} condition(s) triggered.[/red]")
        raise typer.Exit(code=2)


@app.command()
def jupyter(
    folder: Path = typer.Argument(Path("."), help="Folder where the quickstart notebook is written."),
    no_open: bool = typer.Option(False, "--no-open", help="Only write the notebook; do not start JupyterLab."),
):
    """Write a quickstart notebook for the Python API and open it in JupyterLab (if installed)."""
    import shutil

    from kikolens.report.quickstart import quickstart_notebook

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "kikolens_quickstart.ipynb"
    path.write_text(quickstart_notebook(), encoding="utf-8")
    console.print(f"Quickstart notebook written to {path}")
    if no_open:
        return
    lab = shutil.which("jupyter")
    if lab is None:
        console.print('JupyterLab is not installed. Install it with:  pip install "kikolens[jupyter]"  then run:  jupyter lab '
                      f'"{path}"', markup=False)
        return
    console.print("Starting JupyterLab (Ctrl+C to stop)...")
    try:
        subprocess.run([lab, "lab", str(path)], check=False)
    except KeyboardInterrupt:
        pass


@app.command()
def fit(
    source: str, column: str,
    only: Optional[str] = typer.Option(None, help="Comma-separated distributions to try; default all."),
):
    """Which distribution describes a numeric column best (AIC, Kolmogorov-Smirnov, normality)."""
    from kikolens.analyze.distributions import fit_distributions

    df = _load(source)
    _check_target(df, column)
    try:
        res = fit_distributions(df[column], _names(only) or None)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "Fits, best first")


@app.command("ab-seq")
def ab_seq(
    source: str, group: str, outcome: str, control: str,
    order: Optional[str] = typer.Option(None, help="Column that gives the order of arrival (a date); default: the row order."),
    alpha: float = typer.Option(0.05, help="Significance level."),
):
    """Sequential A/B test (mSPRT): valid even if you look at the data after every observation."""
    from kikolens.analyze.sequential import from_frame

    df = _load(source)
    try:
        res = from_frame(df, group, outcome, control, order, alpha=alpha)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


@app.command()
def bandit(
    rates: str = typer.Argument(..., help="True conversion rates of the variants, comma-separated, such as 0.04,0.08."),
    visitors: int = typer.Option(5000, help="Visitors to simulate."),
    seed: int = typer.Option(0, help="Random seed."),
):
    """Simulate a Thompson-sampling bandit against an even split."""
    from kikolens.analyze.sequential import thompson

    try:
        res = thompson([float(r) for r in rates.split(",")], visitors, seed)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


@app.command()
def granger(
    source: str, date: str, cause: str, effect: str,
    max_lag: int = typer.Option(7, help="Most periods back to try."),
    agg: str = typer.Option("mean", help="mean or sum, for rows that share a period."),
):
    """Does the past of one column help predict another (Granger)? Predictive, not proof of cause."""
    from kikolens.analyze.multivar import granger as run_granger

    df = _load(source)
    try:
        res = run_granger(df, date, cause, effect, max_lag, agg)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "By lag")


@app.command("forecast-drivers")
def forecast_drivers(
    source: str, date: str, target: str,
    drivers: str = typer.Option(..., help="Comma-separated columns that help explain the target."),
    horizon: int = typer.Option(12, help="Periods to forecast."),
    scenario: Optional[str] = typer.Option(None, help="Future driver values as factors, such as price=1.1,promo=2."),
    output: Optional[Path] = typer.Option(None, help="Write the forecast to this CSV."),
):
    """Forecast a column using other columns too, and check whether they really help."""
    from kikolens.analyze.multivar import forecast_with_drivers

    df = _load(source)
    try:
        factors = {k.strip(): float(v) for k, v in (pair.split("=") for pair in scenario.split(","))} if scenario else None
        res = forecast_with_drivers(df, date, target, _names(drivers), horizon, scenario=factors)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["forecast"].reset_index().rename(columns={"index": "date"}).round(3), "Forecast")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        res["forecast"].to_csv(output)
        console.print(f"Forecast written to {output}")


@app.command("select-features")
def select_features_cmd(
    source: str, target: str,
    exclude: Optional[str] = typer.Option(None, help="Comma-separated columns to leave out."),
    output: Optional[Path] = typer.Option(None, help="Write the data keeping only the recommended columns and the target."),
):
    """Which columns are worth keeping for predicting a target: consensus of four methods and a cross-validated curve."""
    from kikolens.analyze.featureselect import select_features

    df = _load(source)
    _check_target(df, target)
    try:
        with console.status("Scoring the columns..."):
            res = select_features(df, target, drop=_names(exclude))
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "Consensus ranking")
    if output:
        _write_frame(df[res["recommended"] + [target]], output)
        console.print(f"Data with the recommended columns written to {output}")


@app.command()
def segment(
    source: str,
    method: str = typer.Argument("gmm", help="gmm (finds the number of groups), hierarchical or dbscan."),
    columns: Optional[str] = typer.Option(None, help="Comma-separated numeric columns; default all."),
    groups: int = typer.Option(4, help="Groups for hierarchical."),
    output: Optional[Path] = typer.Option(None, help="Write the data with a group column."),
):
    """Segment rows with a Gaussian mixture, hierarchical clustering or DBSCAN, and describe each group in words."""
    from kikolens.analyze import segments as sg

    df = _load(source)
    chosen = _names(columns) or None
    try:
        if method == "gmm":
            res = sg.gmm_auto(df, chosen)
        elif method == "hierarchical":
            res = sg.hierarchical(df, chosen, groups)
        elif method == "dbscan":
            res = sg.dbscan_auto(df, chosen)
        else:
            raise ValueError("The method must be gmm, hierarchical or dbscan.")
    except ValueError as e:
        _fail(e)
    console.print(f"{res['method']}: {res['summary']}")
    _print_frame(res["profiles"][["group", "rows", "share", "described as"]].round(3), "Groups")
    if output:
        _write_frame(df.assign(group=res["labels"]), output)
        console.print(f"Data with a group column written to {output}")


@app.command()
def recommend(
    source: str, user: str, item: str,
    rating: Optional[str] = typer.Option(None, help="Optional rating column; without it every interaction counts the same."),
    for_user: Optional[str] = typer.Option(None, "--for", help="Recommend for this person."),
    similar_to: Optional[str] = typer.Option(None, "--similar-to", help="List the items most like this one."),
    top: int = typer.Option(10, help="How many to list."),
    evaluate: bool = typer.Option(False, "--evaluate", help="Check it honestly against recommending the most popular items."),
):
    """Item-based recommender from a table of who used what."""
    from kikolens.analyze import recommend as rc

    df = _load(source)
    try:
        if evaluate:
            console.print(rc.evaluate(df, user, item, rating, k=top)["summary"])
        model = rc.fit(df, user, item, rating)
        if for_user is not None:
            _print_frame(model.recommend(for_user, top), f"For {for_user}")
        if similar_to is not None:
            _print_frame(model.similar_items(similar_to, top), f"Most like {similar_to}")
        if not (evaluate or for_user or similar_to):
            _print_frame(model.popular(top), "Most popular")
    except ValueError as e:
        _fail(e)


@app.command()
def synth(
    source: str, output: Path = typer.Argument(..., help="Where to write the synthetic data (.csv, .parquet...)."),
    rows: Optional[int] = typer.Option(None, help="Rows to generate; default as many as the original."),
    seed: int = typer.Option(0, help="Random seed."),
    report: bool = typer.Option(False, "--report", help="Also measure fidelity and privacy."),
):
    """Generate synthetic data that looks like the real data (same distributions and correlations) without being real people."""
    from kikolens.analyze import synthetic as sy

    df = _load(source)
    try:
        res = sy.synthesize(df, rows, seed)
        _write_frame(res["data"], output)
        console.print(f"{res['summary']} Written to {output}")
        if report:
            console.print(sy.report(df, res["data"])["summary"])
    except ValueError as e:
        _fail(e)


@app.command()
def runs(
    target: Optional[str] = typer.Option(None, help="Only runs predicting this column."),
    best: bool = typer.Option(False, "--best", help="Show only the best run for --target."),
    compare: Optional[str] = typer.Option(None, help="Comma-separated run ids to compare side by side."),
    delete: Optional[str] = typer.Option(None, help="Delete the run with this id."),
):
    """The history of model lab runs: every training is recorded so you can compare attempts."""
    from kikolens.analyze import runs as rn

    try:
        if delete:
            rn.delete_run(delete)
            console.print(f"Deleted run {delete}.")
            return
        if compare:
            table = rn.compare(_names(compare))
            _print_frame(table[table["differs"]].drop(columns="differs").reset_index().rename(columns={"index": "what"}), "What differs", max_rows=60)
            return
        if best:
            if not target:
                raise ValueError("--best needs --target.")
            record = rn.best_run(target)
            console.print(f"Best run for {target}: {record['id']} ({record['name']}, {record['created']})")
            console.print({k: round(v, 4) for k, v in record["metrics"].items()})
            return
    except ValueError as e:
        _fail(e)
    frame = rn.list_runs(target)
    if frame.empty:
        console.print("No runs yet. Every `kikolens lab` and every dashboard lab training is recorded here.")
        return
    _print_frame(frame.drop(columns=["task"]).round(4), "Runs, newest first")


@app.command()
def doctor(
    offline: bool = typer.Option(False, "--offline", help="Skip the check that asks Ollama whether it is running."),
    as_json: bool = typer.Option(False, "--json", help="Print the checks as JSON, for scripts."),
):
    """Check the installation: versions, optional extras, the user folder, Ollama and the configuration. Exit code 1 if something required is missing."""
    import json

    from rich.markup import escape

    from kikolens import doctor as dr

    checks = dr.run_checks(network=not offline)
    if as_json:
        console.print_json(json.dumps(checks))
    else:
        table = Table(title="KikoLens doctor")
        for column in ("check", "status", "detail"):
            table.add_column(column, overflow="fold")
        colours = {"ok": "green", "warning": "yellow", "error": "red"}
        for c in checks:
            table.add_row(c["name"], f"[{colours[c['status']]}]{c['status']}[/{colours[c['status']]}]", escape(c["detail"]))
        console.print(table)
        console.print(dr.summary(checks))
    if any(c["status"] == "error" for c in checks):
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    rows: int = typer.Option(20000, help="Rows of synthetic data."),
    repeat: int = typer.Option(1, help="Repetitions of each operation; the best time is kept."),
    only: Optional[str] = typer.Option(None, help="Comma-separated words: run only the operations whose name contains one."),
    output: Optional[Path] = typer.Option(None, help="Write the results as JSON."),
):
    """Time the main operations on synthetic data, to spot regressions and know what to expect with big files."""
    import json

    from kikolens.benchmark import run_benchmark

    try:
        with console.status("Measuring..."):
            res = run_benchmark(rows, repeat, _names(only) or None)
    except ValueError as e:
        _fail(e)
    _print_frame(res["table"], f"{rows:,} rows on {res['machine']}, Python {res['python']}")
    console.print(res["summary"])
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({k: v for k, v in res.items() if k != "table"} | {"results": res["table"].to_dict("records")}, indent=2), encoding="utf-8")
        console.print(f"Results written to {output}")


@app.command()
def analyst(
    source: str, question: str = typer.Argument(..., help="What you want to know, in your own words."),
    model: Optional[str] = typer.Option(None, help="Ollama model to use (default from the configuration)."),
    no_samples: bool = typer.Option(False, "--no-samples", help="Send the model only column names, types and ranges, never values."),
    html: Optional[Path] = typer.Option(None, help="If the answer has a chart, write it to this HTML file."),
    trail: Optional[Path] = typer.Option(None, help="Write the steps as Python code (kikolens API calls) to this file."),
):
    """Ask a question about a dataset. A local model decides whether to draw a chart, run SQL or chain analyses; every step is shown."""
    from kikolens.ai.analyst import Analyst
    from kikolens.ai.client import LLM, ollama_transport

    df = _load(source)
    bot = Analyst(df, LLM(ollama_transport(model=model)) if model else LLM(), samples=not no_samples)
    try:
        turn = bot.ask(question)
    except ValueError as e:
        _fail(e)
    if turn.report is not None and not turn.report.used_model:
        for note in turn.report.notes:
            console.print(f"[yellow]{note}[/yellow]")
    console.print(turn.text)
    if turn.sql:
        console.print(f"SQL: {turn.sql}")
    if turn.table is not None and turn.action in ("sql", "chart"):
        _print_frame(turn.table.head(20))
    if turn.report is not None:
        for i, step in enumerate(turn.report.steps, 1):
            console.print(f"  {i}. {step.tool}({step.args}): {step.summary or step.error}")
    if html and turn.figure is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        turn.figure.write_html(str(html), include_plotlyjs="cdn")
        console.print(f"Chart written to {html}")
    if trail and turn.report is not None:
        trail.parent.mkdir(parents=True, exist_ok=True)
        trail.write_text(turn.report.python(str(source)), encoding="utf-8")
        console.print(f"Steps written to {trail}")


@app.command()
def chart(
    source: str, request: str = typer.Argument(..., help="The chart you want, in your own words."),
    output: Path = typer.Option(Path("kikolens_output/chart.html"), help="Where to write the chart (.html, or .json for the specification)."),
    model: Optional[str] = typer.Option(None, help="Ollama model to use."),
):
    """Describe a chart in words; a local model writes a specification that KikoLens checks and draws."""
    import json

    from kikolens.ai.analyst import ask_chart
    from kikolens.ai.client import LLM, LLMError, ollama_transport

    df = _load(source)
    try:
        fig, table, spec = ask_chart(df, request, LLM(ollama_transport(model=model)) if model else LLM())
    except (ValueError, LLMError) as e:
        _fail(e)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    else:
        fig.write_html(str(output), include_plotlyjs="cdn")
    console.print(f"{spec['kind']} chart written to {output} ({len(table):,} rows drawn)")


@app.command("classify-text")
def classify_text(
    source: str, column: str,
    labels: str = typer.Option(..., help="Comma-separated labels, such as shipping,quality,price."),
    output: Optional[Path] = typer.Option(None, help="Write the texts with their label."),
    limit: int = typer.Option(300, help="Most texts to classify."),
    model: Optional[str] = typer.Option(None, help="Ollama model to use."),
):
    """Classify a text column into labels you choose, without training anything (a local model reads each text)."""
    from kikolens.ai.analyst import classify_texts
    from kikolens.ai.client import LLM, LLMError, ollama_transport

    df = _load(source)
    _check_target(df, column)
    try:
        with console.status("Reading the texts..."):
            result = classify_texts(df[column], _names(labels), LLM(ollama_transport(model=model)) if model else LLM(), limit=limit)
    except (ValueError, LLMError) as e:
        _fail(e)
    counts = result["label"].value_counts()
    console.print(", ".join(f"{label}: {n}" for label, n in counts.items()))
    if output:
        _write_frame(result.reset_index(drop=True), output)
        console.print(f"Classified texts written to {output}")


def cli():
    app()


@app.command()
def effect(
    source: str, treatment: str, outcome: str,
    covariates: Optional[str] = typer.Option(None, help="Comma-separated columns to adjust for (default: all the others)."),
    method: str = typer.Option("aipw", help="aipw, regression, ipw or matching."),
    treated_value: Optional[str] = typer.Option(None, help="Value of the treatment column that counts as treated."),
):
    """Causal effect of a treatment on an outcome, adjusting for confounders (propensity, matching, doubly robust)."""
    from kikolens.analyze.causality import estimate_effect

    df = _load(source)
    try:
        res = estimate_effect(df, treatment, outcome, _names(covariates) or None, treated_value=_typed(df, treatment, treated_value), method=method)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "Estimates")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def uplift(
    source: str, treatment: str, outcome: str,
    covariates: Optional[str] = typer.Option(None, help="Comma-separated columns that may change the effect."),
    treated_value: Optional[str] = typer.Option(None, help="Value of the treatment column that counts as treated."),
):
    """Who responds most to a treatment: effect per row, Qini curve and deciles."""
    from kikolens.analyze.causality import uplift as run

    df = _load(source)
    try:
        res = run(df, treatment, outcome, _names(covariates) or None, treated_value=_typed(df, treatment, treated_value))
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["deciles"].round(4), "By decile of predicted uplift")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def discontinuity(
    source: str, running: str, outcome: str, cutoff: float,
    bandwidth: Optional[float] = typer.Option(None, help="Distance from the cutoff to use (default: automatic)."),
):
    """Effect of crossing a threshold (regression discontinuity)."""
    from kikolens.analyze.causality import regression_discontinuity

    df = _load(source)
    try:
        res = regression_discontinuity(df, running, outcome, cutoff, bandwidth)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "By bandwidth")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("synth-control")
def synth_control(source: str, unit: str, time: str, outcome: str, treated_unit: str, intervention: str):
    """Effect on one unit against a weighted mix of control units (synthetic control, with placebo tests)."""
    from kikolens.analyze.causality import synthetic_control

    df = _load(source)
    try:
        res = synthetic_control(df, unit, time, outcome, _typed(df, unit, treated_unit), intervention)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["weights"].rename("weight").reset_index().rename(columns={"index": unit}).round(4), "Control weights")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def iv(
    source: str, outcome: str, endogenous: str,
    instruments: str = typer.Option(..., help="Comma-separated instrument columns."),
    covariates: Optional[str] = typer.Option(None, help="Comma-separated controls."),
):
    """Instrumental variables (two-stage least squares) when the treatment is confounded."""
    from kikolens.analyze.causality import iv_2sls

    df = _load(source)
    try:
        res = iv_2sls(df, outcome, endogenous, _names(instruments), _names(covariates))
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def simulate(
    formula: str,
    data: Optional[str] = typer.Option(None, help="File whose numeric columns are resampled as inputs (their correlation is kept)."),
    input: list[str] = typer.Option([], "--input", help="An input as name=distribution(args), such as price=normal(10,2). Repeat it; it overrides a column of --data."),
    corr: list[str] = typer.Option([], "--corr", help="Correlation between two inputs as a,b=0.6. Repeat it."),
    n: int = typer.Option(10000, help="Number of simulations."),
    threshold: Optional[float] = typer.Option(None, help="Report the chance that the result goes above this value."),
    output: Optional[Path] = typer.Option(None, help="Write every simulation to this CSV."),
    seed: int = typer.Option(0, help="Seed, for repeatable runs."),
):
    """Monte Carlo: spread the uncertainty of several inputs through a formula (price * volume - cost)."""
    from kikolens.analyze import simulation as sim

    try:
        inputs, correlations = {}, {}
        if data:
            inputs, found = sim.inputs_from_data(_load(data))
            if found is not None:
                correlations = {(a, b): found.loc[a, b] for a in found.index for b in found.columns if a < b}
        for text in input:
            name, spec = sim.parse_input(text)
            inputs[name] = spec
            correlations = {k: v for k, v in correlations.items() if name not in k}
        for text in corr:
            pair, _, rho = text.partition("=")
            a, _, b = pair.partition(",")
            correlations[(a.strip(), b.strip())] = float(rho)
        res = sim.monte_carlo(inputs, formula, n, correlations or None, threshold, seed=seed)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["sensitivity"].round(4), "What moves the result")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        res["samples"].to_csv(output, index=False)
        console.print(f"Simulations written to {output}")


@app.command()
def optimize(
    formula: str,
    var: list[str] = typer.Option(..., "--var", help="A variable with its bounds as name=min:max. Repeat it."),
    constraint: list[str] = typer.Option([], "--constraint", help="A restriction such as 'a + b <= 100'. Repeat it."),
    minimize: bool = typer.Option(False, "--minimize", help="Look for the lowest value instead of the highest."),
):
    """Find the values of some variables that make a formula as high (or low) as possible, within limits."""
    from kikolens.analyze import simulation as sim

    try:
        res = sim.optimize_expression(formula, dict(sim.parse_variable(v) for v in var), constraint, maximize=not minimize)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def lp(problem: Path = typer.Argument(..., help="JSON with objective, constraints, bounds, maximize and integer (see docs/models.md).")):
    """Solve a linear or integer program from a JSON file: allocations, mixes, schedules."""
    import json

    from kikolens.analyze import simulation as sim

    try:
        spec = json.loads(problem.read_text(encoding="utf-8"))
        bounds = {k: tuple(v) for k, v in spec.get("bounds", {}).items()}
        res = sim.linear_program(spec["objective"], spec.get("constraints", []), bounds, spec.get("maximize", True), spec.get("integer"))
    except (OSError, ValueError, KeyError) as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["constraints"].round(4), "Constraints")


@app.command()
def mmm(
    source: str, date: str, target: str,
    channels: str = typer.Option(..., help="Comma-separated spend columns."),
    controls: Optional[str] = typer.Option(None, help="Comma-separated columns that also move the target (price, weather...)."),
    budget: Optional[float] = typer.Option(None, help="Also propose how to split this spend per period (default: today's total)."),
    reallocate: bool = typer.Option(False, "--reallocate", help="Propose a split of today's total spend."),
):
    """Marketing mix model: what each channel contributes, with carry-over and saturation."""
    from kikolens.analyze import marketing as mk

    df = _load(source)
    try:
        res = mk.marketing_mix(df, date, target, _names(channels), _names(controls))
        plan = mk.optimize_budget(res, budget) if (budget or reallocate) else None
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "By channel")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if plan:
        console.print(plan["summary"])
        _print_frame(plan["table"].round(3), "Proposed spend per period")


@app.command()
def elasticity(
    source: str, quantity: str, price: str,
    controls: Optional[str] = typer.Option(None, help="Comma-separated columns that also move demand."),
    group: Optional[str] = typer.Option(None, help="Estimate one elasticity per value of this column."),
    instrument: Optional[str] = typer.Option(None, help="Comma-separated columns that move the price but not demand (a cost, a tax)."),
    unit_cost: Optional[float] = typer.Option(None, help="Unit cost, to find the price that maximises margin."),
):
    """Price elasticity of demand (log-log), with the revenue or margin curve."""
    from kikolens.analyze.marketing import price_elasticity

    df = _load(source)
    try:
        res = price_elasticity(df, quantity, price, _names(controls), group, _names(instrument) or None, unit_cost)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if res["groups"] is not None:
        _print_frame(res["groups"].round(3), "By group")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("plugin")
def run_plugin(
    name: str, source: str,
    arg: list[str] = typer.Option([], "--arg", help="An argument as name=value. Repeat it. Lists of columns are comma-separated."),
    output: Optional[Path] = typer.Option(None, help="Write the main table to this file."),
):
    """Run an analysis added by a plugin (see `kikolens plugins`)."""
    from kikolens import plugins

    df = _load(source)
    try:
        found = plugins.analyses().get(name)
        if found is None:
            raise ValueError(f"No plugin analysis called '{name}'." + (f" Available: {', '.join(plugins.analyses())}." if plugins.analyses() else " No plugins are loaded."))
        args = {}
        for text in arg:
            key, sep, value = text.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"Cannot read '{text}'. Write arguments as name=value.")
            key = key.strip()
            args[key] = _names(value) if found.params.get(key, {}).get("type") == "columns" else value
        res = plugins.run(name, df, **args)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    table = next((v for v in res.values() if isinstance(v, pd.DataFrame)), None)
    if table is not None:
        _print_frame(table.round(4))
        if output:
            _write_frame(table, output)
            console.print(f"Table saved to {output}")


@app.command()
def plugins():
    """List the plugins that are loaded: their analyses, the formats they read and anything that failed to load."""
    from kikolens import plugins as registry

    if not registry.enabled():
        console.print("Plugins are turned off (KIKOLENS_PLUGINS=0).")
        return
    found = registry.analyses()
    for item in registry.loaded():
        console.print(f"[bold]{item['plugin']}[/bold]  {item['source']}  ({item['analyses']} analyses, {item['loaders']} loaders)")
    if found:
        _print_frame(pd.DataFrame([{"analysis": a.name, "plugin": a.plugin, "arguments": ", ".join(a.params) or "-", "description": a.description} for a in found.values()]),
                     "Analyses (run with: kikolens plugin NAME SOURCE --arg name=value)")
    if registry.loaders():
        console.print("Formats: " + ", ".join(sorted(registry.loaders())))
    for message in registry.errors():
        console.print(f"[red]{message}[/red]")
    if not found and not registry.loaders() and not registry.errors():
        from kikolens.config import home_dir
        console.print(f"No plugins found. Put a .py file with a register(registry) function in {home_dir() / 'plugins'} or list paths under [plugins] in kikolens.toml.")


@app.command("api")
def api_server_command(
    port: int = typer.Option(8000, help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", help="Address to listen on. Anything but this computer needs --token."),
    token: Optional[str] = typer.Option(None, help="Require this bearer token on every request except /health."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="A folder whose files clients may load by name. Without it, clients can only send rows or csv text."),
    model: list[str] = typer.Option([], "--model", help="A saved model to serve at /models/NAME/predict. Repeat it. Model files are code: use only ones you trust."),
    cors: Optional[str] = typer.Option(None, help="Allow browser pages from this origin (for example http://localhost:3000)."),
):
    """Serve KikoLens as a JSON API: datasets in memory, every analysis, background jobs, models and plugins. See /openapi.json."""
    from kikolens import api_server
    from kikolens.analyze.predict import load_bundle, resolve_model

    try:
        bundles = {}
        for reference in model:
            bundle = load_bundle(resolve_model(reference))
            bundles[bundle.get("name") or reference] = bundle
        server = api_server.make_server(host, port, token, data_dir, bundles, cors)
    except (ValueError, OSError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"KikoLens API at http://{host}:{server.server_address[1]}  (docs: /openapi.json, analyses: /functions). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.store.pool.shutdown(wait=False, cancel_futures=True)


def _classic_series(source: str, date: str, value: str, agg: str):
    from kikolens.analyze import timeseries as ts

    df = _load(source)
    try:
        return ts.prepare_series(df, date, value, agg=agg)
    except ValueError as e:
        _fail(e)


@app.command()
def stationarity(source: str, date: str, value: str, agg: str = typer.Option("mean", help="mean or sum, for rows that share a period."),
                 trend: bool = typer.Option(False, "--trend", help="Allow a deterministic trend under the null (regression ct).")):
    """Is the series stationary? ADF and KPSS together, and how many differences it needs."""
    from kikolens.analyze import classic

    series = _classic_series(source, date, value, agg)
    try:
        res = classic.stationarity(series, "ct" if trend else "c")
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "By number of differences")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def acf(source: str, date: str, value: str, agg: str = typer.Option("mean", help="mean or sum."), nlags: Optional[int] = typer.Option(None, help="How many lags to show.")):
    """Autocorrelation and partial autocorrelation, with the Ljung-Box test."""
    from kikolens.analyze import classic

    series = _classic_series(source, date, value, agg)
    try:
        res = classic.acf_pacf(series, nlags)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].head(16).round(3), "First lags")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def stl(source: str, date: str, value: str, period: Optional[int] = typer.Option(None, help="Observations per season (default: detected)."),
        agg: str = typer.Option("mean", help="mean or sum."), output: Optional[Path] = typer.Option(None, help="Write trend, seasonal and remainder to this file.")):
    """Trend, seasonal and remainder with STL (robust to outliers), with the strength of each."""
    from kikolens.analyze import classic

    series = _classic_series(source, date, value, agg)
    try:
        res = classic.stl(series, period)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        _write_frame(res["components"].reset_index(names="date"), output)
        console.print(f"Components written to {output}")


@app.command()
def arima(
    source: str, date: str, value: str,
    order: Optional[str] = typer.Option(None, help="p,d,q such as 1,1,1. Without it the best model is searched by AICc."),
    seasonal: str = typer.Option("0,0,0", help="P,D,Q of the seasonal part."),
    period: Optional[int] = typer.Option(None, help="Observations per season (default: detected)."),
    horizon: int = typer.Option(12, help="Periods to forecast."),
    agg: str = typer.Option("mean", help="mean or sum."),
    output: Optional[Path] = typer.Option(None, help="Write the forecast to this file."),
):
    """ARIMA with seasonal terms: forecast with ranges, residual diagnostics and a check against repeating the last season."""
    from kikolens.analyze import classic

    series = _classic_series(source, date, value, agg)
    try:
        if order is None:
            res = classic.auto_arima(series, period, horizon)
        else:
            res = classic.arima(series, tuple(int(x) for x in order.split(",")), tuple(int(x) for x in seasonal.split(",")), period, horizon)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["forecast"].reset_index(names="date").round(3), "Forecast")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        _write_frame(res["forecast"].reset_index(names="date"), output)
        console.print(f"Forecast written to {output}")


governance_app = typer.Typer(help="Who owns each saved model, what stage it is in, and what was checked before it went live.", no_args_is_help=False)
app.add_typer(governance_app, name="governance")


def _governed(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, OSError) as e:
        _fail(e)


@governance_app.callback(invoke_without_command=True)
def governance_overview(ctx: typer.Context):
    """Every registered model with its stage, owner, risk, latest validation and whether a review is due."""
    if ctx.invoked_subcommand is not None:
        return
    from kikolens.analyze import governance as gv

    table = gv.overview()
    if table.empty:
        console.print("No models are registered yet. Use: kikolens governance register MODEL --owner NAME")
        return
    _print_frame(table, "Model governance")
    check = gv.verify_audit()
    if not check["ok"]:
        console.print(f"[red]Audit log: {check['message']}[/red]")


@governance_app.command("register")
def governance_register(
    model: str, owner: str = typer.Option(..., help="The person who answers for this model."),
    description: str = typer.Option("", help="What the model is for."), intended_use: str = typer.Option("", help="Where and how it should be used."),
    limitations: str = typer.Option("", help="Where it should not be trusted."), risk: str = typer.Option("medium", help="low, medium or high."),
    actor: Optional[str] = typer.Option(None, help="Who is doing this (default: your user name)."),
):
    """Register a saved model for governance. It starts as a draft."""
    from kikolens.analyze import governance as gv

    record = _governed(gv.register, model, owner, description, intended_use, limitations, risk, actor)
    console.print(f"{record['model']} registered as a draft, owned by {record['owner']} (risk {record['risk']}).")


@governance_app.command("update")
def governance_update(
    model: str, owner: Optional[str] = None, description: Optional[str] = None, intended_use: Optional[str] = None, limitations: Optional[str] = None,
    risk: Optional[str] = None, actor: Optional[str] = typer.Option(None, help="Who is doing this."),
):
    """Change the owner, the card text or the risk of a registered model."""
    from kikolens.analyze import governance as gv

    fields = {k: v for k, v in dict(owner=owner, description=description, intended_use=intended_use, limitations=limitations, risk=risk).items() if v is not None}
    _governed(gv.update_card, model, actor, **fields)
    console.print(f"Updated: {', '.join(fields) or 'nothing'}.")


@governance_app.command("validate")
def governance_validate(
    model: str, data: str = typer.Argument(..., help="Rows the model has not seen, with the real result in the target column."),
    actor: Optional[str] = typer.Option(None, help="Who is doing this."),
    max_drop: float = typer.Option(0.20, help="How much worse than in training is tolerated, as a fraction."),
    raw: bool = typer.Option(False, "--raw", help="The data is uncleaned: repeat the cleaning steps used in training first."),
):
    """Test the model on new rows with known results and record whether it holds up."""
    from kikolens.analyze import governance as gv

    result = _governed(gv.validate, model, _load(data), actor, raw, max_drop)
    colour = "green" if result["passed"] else "red" if result["passed"] is False else "yellow"
    label = "Passed" if result["passed"] else "Not passed" if result["passed"] is False else "Recorded"
    console.print(f"[{colour}]{label}[/{colour}] on {result['rows']:,} rows. {result['note']}")


@governance_app.command("checks")
def governance_checks(model: str, stage: str, data: Optional[str] = typer.Option(None, help="Rows with real results, to compare with the model in production.")):
    """What would stop this model from moving to a stage, without moving it."""
    from kikolens.analyze import governance as gv

    gates = _governed(gv.checks, model, stage, _load(data) if data else None)
    for gate in gates:
        colour = "green" if gate["passed"] else "red"
        console.print(f"[{colour}]{'ok  ' if gate['passed'] else 'FAIL'}[/{colour}] {gate['name']}: {gate['detail']}")


@governance_app.command("promote")
def governance_promote(
    model: str, stage: str, actor: Optional[str] = typer.Option(None, help="Who is doing this."),
    data: Optional[str] = typer.Option(None, help="Rows with real results, to compare with the model in production."),
    reason: str = typer.Option("", help="Why (needed with --force)."), force: bool = typer.Option(False, "--force", help="Move it even if a check fails; recorded as an override."),
    raw: bool = typer.Option(False, "--raw", help="The data is uncleaned."),
):
    """Move a model to another stage if its checks pass."""
    from kikolens.analyze import governance as gv

    out = _governed(gv.promote, model, stage, actor, _load(data) if data else None, reason, force, raw)
    console.print(f"{out['model']} is now in {out['to']} (was {out['from']})." + (f" It replaced {out['replaced']}, now archived." if out["replaced"] else "")
                  + (f" Overrode: {', '.join(out['overridden'])}." if out["overridden"] else ""))


@governance_app.command("approve")
def governance_approve(model: str, actor: str = typer.Option(..., help="Who approves (not the owner)."), comment: str = typer.Option("", help="Optional note.")):
    """Record an approval of this exact version of the model (high-risk models need one from someone other than the owner)."""
    from kikolens.analyze import governance as gv

    _governed(gv.approve, model, actor, comment)
    console.print(f"Approval by {actor} recorded.")


@governance_app.command("rollback")
def governance_rollback(target: str, actor: Optional[str] = typer.Option(None, help="Who is doing this."), reason: str = typer.Option("", help="Why.")):
    """Go back to the model that was in production for TARGET before the current one."""
    from kikolens.analyze import governance as gv

    out = _governed(gv.rollback, target, actor, reason)
    console.print(f"{out['restored']} is back in production" + (f"; {out['demoted']} moved to staging." if out["demoted"] else "."))


@governance_app.command("card")
def governance_card(model: str, output: Optional[Path] = typer.Option(None, help="Write the card (Markdown) to this file.")):
    """The model card: purpose, owner, performance, validations, approvals and history."""
    from kikolens.analyze import governance as gv

    card = _governed(gv.model_card, model)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(card, encoding="utf-8")
        console.print(f"Card written to {output}")
    else:
        console.print(card)


@governance_app.command("audit")
def governance_audit(model: Optional[str] = typer.Argument(None), last: int = typer.Option(25, help="How many entries to show.")):
    """Check the audit log chain and show the latest entries."""
    import json

    from kikolens.analyze import governance as gv

    check = gv.verify_audit()
    colour = "green" if check["ok"] else "red"
    console.print(f"[{colour}]{check['message']}[/{colour}]")
    log = gv.audit_log(model)
    if len(log):
        view = log.tail(last)[["ts", "actor", "action", "model", "details"]].copy()
        view["details"] = view["details"].map(lambda d: json.dumps(d, ensure_ascii=False)[:80])
        _print_frame(view)


def _text_column(source: str, column: str) -> pd.DataFrame:
    df = _load(source)
    if column not in df.columns:
        _fail(ValueError(f"Column '{column}' not found."))
    return df


@app.command()
def topics(
    source: str, column: str,
    n_topics: Optional[int] = typer.Option(None, "--topics", help="How many topics (default: chosen by coherence)."),
    method: str = typer.Option("nmf", help="nmf or lda."), date: Optional[str] = typer.Option(None, help="A date column, to see how topics change over time."),
    language: Optional[str] = typer.Option(None, help="english or spanish (default: detected)."),
):
    """Topics in a text column, with example texts and (with --date) their evolution."""
    from kikolens.analyze import textmining

    df = _text_column(source, column)
    try:
        res = textmining.topics_over_time(df, column, date, n_topics, method=method, language=language) if date else textmining.topics(df[column], n_topics, method=method, language=language)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].drop(columns="examples").round(3), "Topics")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("near-duplicates")
def near_duplicates(source: str, column: str, threshold: float = typer.Option(0.9, help="Similarity from 0.5 to 1.")):
    """Groups of near-identical texts (typos, small edits)."""
    from kikolens.analyze import textmining

    df = _text_column(source, column)
    try:
        res = textmining.find_duplicates(df[column], threshold)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if res["groups"]:
        _print_frame(res["table"].drop(columns="rows").round(3), "Groups")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def similar(source: str, column: str, query: str, n: int = typer.Option(5, help="How many texts to show.")):
    """The texts most similar to a phrase."""
    from kikolens.analyze import textmining

    df = _text_column(source, column)
    try:
        res = textmining.similar_texts(df[column], query, n)
    except ValueError as e:
        _fail(e)
    _print_frame(res.round(3), "Most similar texts (similarity)")


@app.command()
def patterns(source: str, column: str, kinds: Optional[str] = typer.Option(None, help="Comma-separated: email, url, phone, money, percent, date, hashtag, mention.")):
    """Emails, links, phones, amounts, percentages, dates, hashtags and mentions in a text column."""
    from kikolens.analyze import textmining

    df = _text_column(source, column)
    try:
        res = textmining.extract_patterns(df[column], _names(kinds) or None)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(3), "By kind")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def keyphrases(source: str, column: str, n: int = typer.Option(15, help="How many phrases."), language: Optional[str] = typer.Option(None, help="english or spanish.")):
    """Key phrases (RAKE) of a text column."""
    from kikolens.analyze import textmining

    df = _text_column(source, column)
    try:
        res = textmining.keyphrases(df[column], n, language)
    except ValueError as e:
        _fail(e)
    _print_frame(res.round(2), "Key phrases")


@app.command()
def readability(source: str, column: str, language: Optional[str] = typer.Option(None, help="english or spanish.")):
    """How easy the texts are to read (Flesch, or Fernandez-Huerta for Spanish)."""
    from kikolens.analyze import textmining

    df = _text_column(source, column)
    try:
        res = textmining.readability(df[column], language)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    console.print(res["notes"][0])


def _edges(source_file: str, source: str, target: str, weight):
    df = _load(source_file)
    for column in (source, target, weight):
        if column and column not in df.columns:
            _fail(ValueError(f"Column '{column}' not found."))
    return df


@app.command()
def network(
    file: str, source: str, target: str, weight: Optional[str] = typer.Option(None, help="A numeric column with the strength of each link."),
    directed: bool = typer.Option(False, "--directed", help="Links have a direction (from source to target)."), top: int = typer.Option(15, help="How many nodes to list."),
    output: Optional[Path] = typer.Option(None, help="Write every node with its measures to this file."),
):
    """Analyse a network from a list of links: important nodes, communities, connectivity and distances."""
    from kikolens.analyze import network as nw

    df = _edges(file, source, target, weight)
    try:
        res = nw.analyze(df, source, target, weight, directed)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].head(top).round(4), "Most important nodes (by PageRank)")
    _print_frame(res["communities"].round(4), "Communities")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        _write_frame(res["table"], output)
        console.print(f"Nodes written to {output}")


@app.command("network-path")
def network_path(file: str, source: str, target: str, start: str, end: str, directed: bool = typer.Option(False, "--directed")):
    """The path with the fewest steps between two nodes."""
    from kikolens.analyze import network as nw

    df = _edges(file, source, target, None)
    try:
        res = nw.shortest_path(df, source, target, start, end, None, directed)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


@app.command("suggest-links")
def suggest_links(file: str, source: str, target: str, top: int = typer.Option(10, help="How many suggestions.")):
    """Pairs of nodes that are not linked but share many neighbours: the most likely missing links."""
    from kikolens.analyze import network as nw

    df = _edges(file, source, target, None)
    try:
        res = nw.suggest_links(df, source, target, None, top)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(3), "Suggested links")


@app.command("network-robustness")
def network_robustness(file: str, source: str, target: str, steps: int = typer.Option(10, help="How many removal steps up to half the nodes.")):
    """How fast the network falls apart when its best-connected nodes are removed, against removing random ones."""
    from kikolens.analyze import network as nw

    df = _edges(file, source, target, None)
    try:
        res = nw.robustness(df, source, target, None, steps)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(3), "Largest piece as nodes are removed")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


def _geo_frame(source: str, lat: Optional[str], lon: Optional[str]):
    from kikolens.analyze import geo

    df = _load(source)
    if lat and lon:
        return df, lat, lon
    found = geo.detect_coordinates(df)
    if not found:
        _fail(ValueError("No latitude and longitude columns were found; give them with --lat and --lon."))
    return df, found[0][0], found[0][1]


@app.command("geo-check")
def geo_check(file: str, lat: Optional[str] = typer.Option(None, help="Latitude column (default: detected)."), lon: Optional[str] = typer.Option(None, help="Longitude column.")):
    """Which rows have usable coordinates, and what looks wrong (swapped, out of range, 0,0, repeated)."""
    from kikolens.analyze import geo

    df, la, lo = _geo_frame(file, lat, lon)
    try:
        res = geo.check_coordinates(df, la, lo)
    except ValueError as e:
        _fail(e)
    console.print(f"Using '{la}' and '{lo}'. {res['summary']}")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("geo-distances")
def geo_distances(
    file: str, lat: Optional[str] = None, lon: Optional[str] = None,
    point: Optional[str] = typer.Option(None, help="One place as lat,lon such as 40.4168,-3.7038."),
    centers: Optional[str] = typer.Option(None, help="A file with the centres (shops, warehouses)."),
    center_lat: str = typer.Option("lat", help="Latitude column in the centres file."), center_lon: str = typer.Option("lon", help="Longitude column in the centres file."),
    center_name: Optional[str] = typer.Option(None, help="Name column in the centres file."), reach_km: Optional[float] = typer.Option(None, help="Report the share within this distance."),
):
    """Kilometres to a point or to the nearest of a list of centres, and how many points each centre serves."""
    from kikolens.analyze import geo

    df, la, lo = _geo_frame(file, lat, lon)
    if not point and not centers:
        _fail(ValueError("Give --point lat,lon or --centers FILE."))
    try:
        target = tuple(float(x) for x in point.split(",")) if point else _load(centers)
        res = geo.distances(df, la, lo, target, center_lat, center_lon, center_name, reach_km)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(2), "By nearest centre")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("geo-hotspots")
def geo_hotspots(file: str, lat: Optional[str] = None, lon: Optional[str] = None, value: Optional[str] = typer.Option(None, help="A numeric column to average in each zone."),
                 eps_km: float = typer.Option(5.0, help="Largest gap between neighbouring points in a zone."), min_points: int = typer.Option(10, help="Fewest points in a zone.")):
    """Zones with many points close together."""
    from kikolens.analyze import geo

    df, la, lo = _geo_frame(file, lat, lon)
    try:
        res = geo.hotspots(df, la, lo, value, eps_km, min_points)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if len(res["table"]):
        _print_frame(res["table"].round(3), "Zones")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def spatial(file: str, value: str, lat: Optional[str] = None, lon: Optional[str] = None, k: int = typer.Option(8, help="Neighbours per point."),
            permutations: int = typer.Option(999, help="Shuffles for the p-value.")):
    """Do nearby points have similar values (Moran's I), and where are the hot and cold spots (Getis-Ord)?"""
    from kikolens.analyze import geo

    df, la, lo = _geo_frame(file, lat, lon)
    try:
        res = geo.autocorrelation(df, la, lo, value, k, permutations)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if len(res["table"]):
        _print_frame(res["table"].head(15).round(3), "Strongest local spots")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("geo-grid")
def geo_grid(file: str, lat: Optional[str] = None, lon: Optional[str] = None, value: Optional[str] = None, cell_km: float = typer.Option(10.0, help="Cell size in km."),
             agg: str = typer.Option("mean", help="mean, sum, median, max or min."), output: Optional[Path] = typer.Option(None, help="Write the cells to this file.")):
    """Summarise points in square cells of a few kilometres."""
    from kikolens.analyze import geo

    df, la, lo = _geo_frame(file, lat, lon)
    try:
        res = geo.grid_summary(df, la, lo, value, cell_km, agg)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].head(15).round(3), "Busiest cells")
    if output:
        _write_frame(res["table"], output)
        console.print(f"Cells written to {output}")


@app.command()
def experiment(
    file: str, group: str, outcome: str, control: str,
    pre: Optional[str] = typer.Option(None, help="The same measure before the experiment, to reduce noise (CUPED)."),
    date: Optional[str] = typer.Option(None, help="A date column, to see whether the effect holds over time."),
    segments: Optional[str] = typer.Option(None, help="Comma-separated columns to look for different effects in."),
    expected: Optional[str] = typer.Option(None, help="Planned split such as control=50,new=50 (default: equal)."),
):
    """Full analysis of an A/B test: split check, effect of each group against the control, noise reduction, effect over time and by segment."""
    from kikolens.analyze import experiment as ex

    df = _load(file)
    try:
        plan = {k.strip(): float(v) for k, v in (pair.split("=") for pair in expected.split(","))} if expected else None
        res = ex.analyze_experiment(df, group, outcome, control, pre, date, _names(segments) or None, plan)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "Each group against the control")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def srm(file: str, group: str, expected: Optional[str] = typer.Option(None, help="Planned split such as control=50,new=50 (default: equal).")):
    """Did each group get the share of users that was planned? A broken split invalidates the test."""
    from kikolens.analyze import experiment as ex

    df = _load(file)
    try:
        plan = {k.strip(): float(v) for k, v in (pair.split("=") for pair in expected.split(","))} if expected else None
        res = ex.srm(df, group, plan)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "Rows by group")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def cuped(file: str, group: str, outcome: str, pre: str, control: str):
    """Reduce the noise of an experiment with what each row did before it (CUPED)."""
    from kikolens.analyze import experiment as ex

    df = _load(file)
    try:
        res = ex.cuped(df, group, outcome, pre, control)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "Raw and adjusted effect")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("control-chart")
def control_chart(
    file: str, value: str, chart: str = typer.Option("individuals", help="individuals, xbar, p, u or c."), order: Optional[str] = typer.Option(None, help="Column that gives the time order."),
    subgroup: Optional[str] = typer.Option(None, help="For xbar: the column that identifies each subgroup."), size: Optional[str] = typer.Option(None, help="For p and u: examined or units."),
    baseline: Optional[int] = typer.Option(None, help="How many initial points set the limits; the rest are monitored."),
    rules: str = typer.Option("1,4", help="Signal rules, 1 to 6 (more rules find changes sooner and give more false alarms)."),
):
    """Control chart with signal rules: is the process stable, or did something change?"""
    from kikolens.analyze import spc

    df = _load(file)
    try:
        res = spc.control_chart(df, value, chart, order, subgroup, size, baseline, True, tuple(int(r) for r in rules.split(",")))
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if len(res["signals"]):
        shown = res["signals"][["position", "rule", "meaning"]].copy()
        shown["position"] += 1
        _print_frame(shown.head(15), "Signals")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def capability(file: str, value: str, lsl: Optional[float] = typer.Option(None, help="Lower specification limit."), usl: Optional[float] = typer.Option(None, help="Upper specification limit."),
               target: Optional[float] = typer.Option(None, help="Target value, for Cpm.")):
    """Process capability (Cp, Cpk, Pp, Ppk) against the specification limits."""
    from kikolens.analyze import spc

    df = _load(file)
    if value not in df.columns:
        _fail(ValueError(f"Column '{value}' not found."))
    try:
        res = spc.capability(df[value], lsl, usl, target)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(3), "Indices")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("shift-detect")
def shift_detect(file: str, value: str, target: Optional[float] = typer.Option(None, help="The value the process should have (default: the baseline mean)."),
                 sigma: Optional[float] = typer.Option(None, help="Its standard deviation (default: from the baseline)."), baseline: Optional[int] = typer.Option(None, help="Initial points that define normal.")):
    """CUSUM and EWMA: small, sustained shifts that an ordinary control chart takes long to see."""
    from kikolens.analyze import spc

    df = _load(file)
    if value not in df.columns:
        _fail(ValueError(f"Column '{value}' not found."))
    try:
        one, two = spc.cusum(df[value], target, sigma, baseline=baseline), spc.ewma(df[value], target=target, sigma=sigma, baseline=baseline)
    except ValueError as e:
        _fail(e)
    console.print(f"CUSUM: {one['summary']}")
    console.print(f"EWMA: {two['summary']}")


@app.command("abc-xyz")
def abc_xyz(file: str, item: str, quantity: str, date: Optional[str] = typer.Option(None, help="Date column: needed for XYZ and the demand classes."),
            value: Optional[str] = typer.Option(None, help="Column with the value of each sale (default: the quantity)."), freq: str = typer.Option("W", help="D, W or M periods."),
            output: Optional[Path] = typer.Option(None, help="Write the classification to this file.")):
    """Classify items by value (ABC), by how much their demand varies (XYZ) and by pattern (smooth, erratic, intermittent, lumpy)."""
    from kikolens.analyze import inventory

    df = _load(file)
    try:
        res = inventory.classify_items(df, item, quantity, date, value, freq)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if res["matrix"] is not None:
        _print_frame(res["matrix"].reset_index(), "Items by ABC and XYZ")
    _print_frame(res["table"].head(15).round(3), "Highest-value items")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        _write_frame(res["table"], output)
        console.print(f"Classification written to {output}")


@app.command()
def newsvendor(cost: float = typer.Option(..., help="Unit cost."), price: float = typer.Option(..., help="Selling price."), salvage: float = typer.Option(0.0, help="Value of an unsold unit."),
               mean: Optional[float] = typer.Option(None, help="Mean demand."), sd: Optional[float] = typer.Option(None, help="Standard deviation of demand.")):
    """How much to order of something sold once (a season, a perishable), from the critical ratio."""
    from kikolens.analyze import inventory

    try:
        res = inventory.newsvendor(cost, price, salvage, mean, sd)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def eoq(demand: float = typer.Option(..., help="Demand per year."), order_cost: float = typer.Option(..., help="Cost of placing one order."),
        holding_cost: float = typer.Option(..., help="Cost of holding one unit for a year."), unit_cost: float = typer.Option(0.0, help="Cost per unit, to include the purchase in the total.")):
    """Economic order quantity: the order size that minimises ordering plus holding cost."""
    from kikolens.analyze import inventory

    try:
        res = inventory.eoq(demand, order_cost, holding_cost, unit_cost)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


@app.command()
def reorder(mean: float = typer.Option(..., help="Mean demand per period."), sd: float = typer.Option(..., help="Standard deviation of demand per period."),
            lead_time: float = typer.Option(..., help="Delivery time, in the same periods."), lead_time_sd: float = typer.Option(0.0, help="Its standard deviation."),
            service_level: float = typer.Option(0.95, help="Chance of no stockout in a cycle."), order_quantity: Optional[float] = typer.Option(None, help="To also get the fill rate.")):
    """Safety stock and reorder point for a service level."""
    from kikolens.analyze import inventory

    try:
        res = inventory.reorder_policy(mean, sd, lead_time, lead_time_sd, service_level, order_quantity)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("stock-simulate")
def stock_simulate(mean: float = typer.Option(..., help="Mean demand per day."), sd: float = typer.Option(..., help="Standard deviation of daily demand."),
                   lead_time: int = typer.Option(..., help="Delivery time in days."), order_quantity: float = typer.Option(..., help="Units per order."),
                   reorder_point: float = typer.Option(..., help="Order when stock plus open orders falls to this."), lead_time_sd: float = typer.Option(0.0, help="Its standard deviation."),
                   days: int = typer.Option(365, help="Days per run."), runs: int = typer.Option(300, help="Number of simulated years."),
                   lost_sales: bool = typer.Option(False, "--lost-sales", help="Unmet demand is lost instead of waiting.")):
    """Simulate a reorder policy for many years to see the service it really gives."""
    from kikolens.analyze import inventory

    try:
        res = inventory.simulate_policy(mean, sd, lead_time, order_quantity, reorder_point, lead_time_sd, days, runs, lost_sales)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])


@app.command("inventory-plan")
def inventory_plan(file: str, item: str, date: str, quantity: str, lead_time: float = typer.Option(..., help="Delivery time in days."),
                   service_level: float = typer.Option(0.95, help="Chance of no stockout in a cycle."), order_cost: Optional[float] = typer.Option(None, help="Cost of one order, for the EOQ."),
                   holding_cost: Optional[float] = typer.Option(None, help="Cost of holding a unit a year, for the EOQ."), lead_time_sd: float = typer.Option(0.0, help="Delivery time standard deviation in days."),
                   freq: str = typer.Option("D", help="D or W: the period of the demand figures."), output: Optional[Path] = typer.Option(None, help="Write the plan to this file.")):
    """A reorder policy (safety stock, reorder point, order quantity) for every item in a table of sales."""
    from kikolens.analyze import inventory

    df = _load(file)
    try:
        res = inventory.plan(df, item, date, quantity, lead_time, service_level, order_cost, holding_cost, lead_time_sd, freq)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].head(20).round(2), "Policy by item")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        _write_frame(res["table"], output)
        console.print(f"Plan written to {output}")


@app.command()
def attribution(
    file: str, user: str, time: str, channel: str, converted: str,
    value: Optional[str] = typer.Option(None, help="A numeric column with the value of the conversion (on the converting row)."),
    half_life: float = typer.Option(7.0, help="Days after which a contact counts half, for the time-decay model."),
    output: Optional[Path] = typer.Option(None, help="Write the credit of each channel under every model to this file."),
):
    """Share the credit for conversions between channels: first, last, linear, time decay, position, Markov chain and Shapley."""
    from kikolens.analyze import attribution as at

    df = _load(file)
    try:
        res = at.attribute(df, user, time, channel, converted, value, half_life)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["shares"].reset_index(names="channel").round(3), "Share of the credit by model")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if output:
        _write_frame(res["credit"].reset_index(names="channel"), output)
        console.print(f"Credit written to {output}")


def _survey(file: str, items: Optional[str] = None):
    df = _load(file)
    names = _names(items)
    return df, names


@app.command()
def likert(file: str, items: str = typer.Option(..., help="Comma-separated scale questions."), reverse: Optional[str] = typer.Option(None, help="Questions worded the other way round.")):
    """Scale questions: mean, favourable and unfavourable share and the spread of answers."""
    from kikolens.analyze import survey

    df, names = _survey(file, items)
    try:
        res = survey.likert_summary(df, names, None, _names(reverse) or None)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"][["question", "answers", "mean", "favourable", "neutral", "unfavourable"]].round(3), "By question")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def reliability(file: str, items: str = typer.Option(..., help="Comma-separated questions that measure the same thing."), reverse: Optional[str] = typer.Option(None, help="Reversed questions.")):
    """Cronbach's alpha of a group of questions, and what happens if each one is removed."""
    from kikolens.analyze import survey

    df, names = _survey(file, items)
    try:
        res = survey.reliability(df, names, _names(reverse) or None)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(3), "By question")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def nps(file: str, score: str, group: Optional[str] = typer.Option(None, help="Compare these groups.")):
    """Net Promoter Score from a 0 to 10 recommendation question."""
    from kikolens.analyze import survey

    df = _load(file)
    try:
        res = survey.nps(df, score, group)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    if group:
        _print_frame(res["table"].round(3), "By group")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def crosstab(file: str, row: str, column: str):
    """Cross two questions: chi-square, strength of the link and the cells that differ from chance."""
    from kikolens.analyze import survey

    df = _load(file)
    try:
        res = survey.crosstab(df, row, column)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["row_percent"].round(3).reset_index(), "Share of each row")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("key-drivers")
def key_drivers(file: str, outcome: str, items: str = typer.Option(..., help="Comma-separated questions that may drive the overall rating.")):
    """Which questions matter most for an overall rating, and which to improve first."""
    from kikolens.analyze import survey

    df, names = _survey(file, items)
    try:
        res = survey.key_drivers(df, outcome, names)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"][["question", "importance", "mean score", "favourable", "priority"]].round(3), "Drivers")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("survey-groups")
def survey_groups(file: str, group: str, items: str = typer.Option(..., help="Comma-separated questions to compare.")):
    """Which questions score differently between groups (ordinal tests, corrected for looking at many)."""
    from kikolens.analyze import survey

    df, names = _survey(file, items)
    try:
        res = survey.compare_groups(df, names, group)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    _print_frame(res["table"].round(4), "By question")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command("response-quality")
def response_quality(file: str, items: str = typer.Option(..., help="Comma-separated scale questions."), seconds: Optional[str] = typer.Option(None, help="Column with seconds spent.")):
    """Respondents who give the same answer to everything, leave most blank or answer too fast."""
    from kikolens.analyze import survey

    df, names = _survey(file, items)
    try:
        res = survey.response_quality(df, names, seconds)
    except ValueError as e:
        _fail(e)
    console.print(res["summary"])
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@app.command()
def learn(
    concept: Optional[str] = typer.Argument(None, help="A concept to simulate; leave it out to list them."),
    set_: list[str] = typer.Option([], "--set", help="A parameter as name=value. Repeat it."), seed: int = typer.Option(0, help="Seed: the same seed gives the same simulation."),
    figure: Optional[Path] = typer.Option(None, help="Write the main chart to this HTML file."),
):
    """Concept lab: simulations that show why statistics works the way it does (central limit theorem, p-values, confidence, power, overfitting...)."""
    from kikolens.analyze import concepts as cp

    if concept is None:
        _print_frame(pd.DataFrame([{"concept": k, "question": v["question"]} for k, v in cp.CONCEPTS.items()]), "Concepts (run one with: kikolens learn NAME)")
        return
    try:
        params = {}
        for text in set_:
            key, sep, value = text.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"Cannot read '{text}'. Write parameters as name=value.")
            params[key.strip()] = value.strip()
        res = cp.run(concept, seed, **params)
    except ValueError as e:
        _fail(e)
    console.print(f"[bold]{res['title']}[/bold]")
    console.print(res["summary"])
    console.print(f"[bold]Lesson:[/bold] {res['lesson']}")
    console.print(f"[bold]Try:[/bold] {res['try']}")
    for note in res["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if figure:
        figure.parent.mkdir(parents=True, exist_ok=True)
        res["figure"].write_html(str(figure), include_plotlyjs=True)
        console.print(f"Chart written to {figure}")


def _typed(df: pd.DataFrame, column: str, value: Optional[str]):
    """Convierte un valor escrito en la terminal al tipo de la columna (para elegir qué valor cuenta como tratado)."""
    if value is None or column not in df.columns:
        return value
    kind = df[column].dtype
    if pd.api.types.is_bool_dtype(kind):
        return value.lower() in ("1", "true", "yes", "si", "sí")
    if pd.api.types.is_numeric_dtype(kind):
        try:
            return float(value)
        except ValueError:
            return value
    return value


if __name__ == "__main__":
    cli()
