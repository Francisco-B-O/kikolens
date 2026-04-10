import os
import subprocess
import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer(help="KikoLens: Data Analysis Platform.")
console = Console()

@app.command()
def start():
    """
    Start KikoLens.
    """
    try:
        current_dir = Path(__file__).resolve().parent
        app_path = current_dir.parent / "dashboard" / "app.py"
        
        if not app_path.exists():
             console.print(f"[bold red]Error: Dashboard not found at {app_path}[/bold red]")
             raise typer.Exit(code=1)

        console.rule("[bold]KIKOLENS")
        console.print("[green]Starting application...[/green]")
        
        cmd = ["streamlit", "run", str(app_path)]
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Application stopped.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Launch Error: {e}[/bold red]")
        raise typer.Exit(code=1)

def cli():
    app()

if __name__ == '__main__':
    cli()
