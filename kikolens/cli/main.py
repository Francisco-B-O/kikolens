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
    save: Optional[Path] = typer.Option(None, help="Save the trained model to this .pkl path."),
):
    """Train a predictive model on a target column."""
    import joblib

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
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], save)
        console.print(f"Model saved to {save}")


@app.command()
def lab(
    source: str,
    target: str,
    folds: int = typer.Option(5, help="Cross-validation folds (3-10)."),
    tune: bool = typer.Option(True, help="Tune the best model."),
    models: Optional[str] = typer.Option(None, help="Comma-separated model names; default is all."),
    balance: bool = typer.Option(False, "--balance", help="Weight rarer classes more (classification)."),
    exclude: Optional[str] = typer.Option(None, help="Comma-separated columns to leave out."),
    save: Optional[Path] = typer.Option(None, help="Save the best pipeline to this .pkl path."),
):
    """Compare many models with cross-validation, tune the best and check for data leaks."""
    import joblib

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
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(res["pipeline"], save)
        console.print(f"Pipeline saved to {save}")


@app.command()
def forecast(
    source: str, date: str, value: str,
    horizon: int = typer.Option(12, help="Periods to forecast."),
    freq: Optional[str] = typer.Option(None, help="Resample frequency: h, D, W, MS, QS, YS."),
):
    """Forecast a numeric column over time."""
    from kikolens.analyze import timeseries as ts

    df = _load(source)
    try:
        series = ts.prepare_series(df, date, value, freq=freq)
        result = ts.forecast(series, horizon=horizon)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    summary = ts.trend_summary(series)
    console.print(f"Trend: [bold]{summary['direction']}[/bold]  ·  seasonal period: {result['period'] or 'none'}")
    _print_frame(pd.DataFrame([result["metrics"]]), "Backtest")
    _print_frame(result["forecast"].rename_axis("date").reset_index(), "Forecast")


@app.command()
def clean(source: str, output: Path = typer.Option(..., help="Where to write the cleaned file (.csv or .parquet).")):
    """Apply safe automatic cleaning and save the result."""
    from kikolens.analyze.quality import auto_clean, profile_quality

    df = _load(source)
    before = profile_quality(df)["score"]
    cleaned, log = auto_clean(df)
    for line in log:
        console.print(f"  • {line}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        cleaned.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        cleaned.to_csv(output, index=False)
    else:
        console.print("[red]Output must end in .csv or .parquet[/red]")
        raise typer.Exit(code=1)
    console.print(f"Quality score {before:.0f} → {profile_quality(cleaned)['score']:.0f}. Saved to {output}")


@app.command()
def report(
    source: str,
    output: Path = typer.Option(Path("kikolens_output/report.html"), help="Report path."),
    target: Optional[str] = typer.Option(None, help="Include AutoML for this column."),
    nn: bool = typer.Option(False, "--nn", help="Also train a neural network for the target."),
    ai: bool = typer.Option(False, "--ai", help="Add commentary from the local LLM."),
):
    """Build a self-contained HTML report (print it to PDF from the browser)."""
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


def cli():
    app()


if __name__ == "__main__":
    cli()
