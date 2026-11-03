"""Regenera los ejemplos (examples/*.csv, report.html) y las capturas de docs/images.

Uso:  pip install -e ".[docs]" && python -m playwright install chromium   (o usa Edge/Chrome instalado)
      python docs/make_screenshots.py
"""
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "examples" / "sales_demo.csv"
NEW_CSV = ROOT / "examples" / "sales_new_quarter.csv"
REGIONS_CSV = ROOT / "examples" / "regions.csv"
IMAGES = ROOT / "docs" / "images"
PORT = 8598
VIEW = {"width": 1360, "height": 860}


def make_demo_data() -> None:
    rng = np.random.default_rng(11)
    n = 540
    days = np.arange(n)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], n, p=[0.35, 0.25, 0.25, 0.15]),
        "channel": rng.choice(["Online", "Store"], n, p=[0.6, 0.4]),
        "units": rng.integers(20, 160, n).astype(float),
        "unit_price": [f"{v:,.0f}" for v in rng.integers(1200, 4800, n)],
        "discount": rng.choice([0, 0.05, 0.1, 0.15, 0.2], n),
    })
    weekly = 220 * np.sin(2 * np.pi * days / 7)
    yearly = 140 * np.sin(2 * np.pi * days / 365)
    df["revenue"] = (1500 + 1.3 * days + weekly + yearly + rng.normal(0, 70, n)).round(2)
    df["units"] = (df["revenue"] / 14 + rng.normal(0, 6, n)).round(0).clip(lower=1)
    df.loc[rng.choice(n, 28, replace=False), "discount"] = np.nan
    df = pd.concat([df, df.sample(7, random_state=1)], ignore_index=True)
    CSV.parent.mkdir(exist_ok=True)
    df.to_csv(CSV, index=False)
    make_companion_data(df, rng)


def make_companion_data(df: pd.DataFrame, rng) -> None:
    """Un trimestre nuevo (con un cambio de mezcla de canales y precios más altos) para puntuar y comparar, y una
    tabla de regiones para practicar uniones."""
    pd.DataFrame({"region": ["North", "South", "East", "West"],
                  "manager": ["Ana Ruiz", "Ben Okoye", "Carla Mendes", "Dev Patel"],
                  "target": [95000, 70000, 68000, 41000]}).to_csv(REGIONS_CSV, index=False)
    n = 120
    days = np.arange(540, 540 + n)
    new = pd.DataFrame({
        "date": pd.date_range("2024-06-23", periods=n).strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], n, p=[0.2, 0.2, 0.2, 0.4]),
        "channel": rng.choice(["Online", "Store"], n, p=[0.85, 0.15]),
        "unit_price": [f"{v:,.0f}" for v in rng.integers(2400, 6200, n)],
        "discount": rng.choice([0, 0.05, 0.1, 0.15, 0.2], n),
    })
    new["revenue"] = (1500 + 1.3 * days + 220 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 70, n)).round(2)
    new["units"] = (new["revenue"] / 14 + rng.normal(0, 6, n)).round(0).clip(lower=1)
    new.to_csv(NEW_CSV, index=False)


def port_in_use() -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("localhost", PORT)) == 0


def stop_server(proc: subprocess.Popen) -> None:
    """Termina el árbol completo: `kikolens start` lanza streamlit como proceso hijo."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()


def start_server() -> subprocess.Popen:
    if port_in_use():
        raise RuntimeError(f"Port {PORT} is busy; an old server would serve stale code. Stop it first.")
    proc = subprocess.Popen(
        [sys.executable, "-m", "kikolens.cli.main", "start", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT,
        env={**__import__("os").environ, "STREAMLIT_SERVER_HEADLESS": "true"})
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
    stop_server(proc)
    raise RuntimeError("The dashboard did not start.")


def pick(page, label: str, value: str) -> None:
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=re.compile(rf"^{label}")).first
    box.locator('[data-baseweb="select"]').click()
    page.keyboard.type(value)
    page.keyboard.press("Enter")
    time.sleep(1.2)


def scroll_main(page, y: int) -> None:
    page.evaluate(f"document.querySelector('[data-testid=\"stMain\"]').scrollTo(0,{y})")
    time.sleep(1.5)


def capture(browser, mode: str) -> None:
    page = browser.new_page(viewport=VIEW, color_scheme=mode)
    page.goto(f"http://localhost:{PORT}/")
    page.wait_for_selector("text=KikoLens", timeout=30000)
    page.screenshot(path=IMAGES / f"welcome-{mode}.png")
    page.set_input_files('input[type="file"]', str(CSV))
    page.wait_for_selector("text=Overview", timeout=30000)
    time.sleep(3)
    nav = page.locator('[data-testid="stSidebar"]')

    def go(name, area):
        nav.get_by_text(area, exact=True).first.click()
        time.sleep(1.5)
        nav.get_by_text(name, exact=True).last.click()  # "Models" es a la vez un área y una sección
        time.sleep(3)

    go("Overview", "Data")
    page.screenshot(path=IMAGES / f"overview-{mode}.png")

    go("Quality and cleaning", "Data")
    page.screenshot(path=IMAGES / f"quality-{mode}.png")
    page.get_by_role("button", name="Auto-clean").click()   # como haría una persona antes de analizar
    time.sleep(3)

    go("Transform", "Data")
    page.screenshot(path=IMAGES / f"transform-{mode}.png")

    go("Explore", "Data")
    pick(page, "Chart", "Line")
    pick(page, "X", "date")
    pick(page, "Y", "revenue")
    page.screenshot(path=IMAGES / f"explore-{mode}.png")

    go("Experiments", "Analysis")
    pick(page, "Compare this number", "revenue")
    pick(page, "Between the groups of", "region")
    page.screenshot(path=IMAGES / f"experiments-{mode}.png")

    go("Anomalies", "Analysis")
    page.get_by_role("button", name="Find anomalies").click()
    time.sleep(8)
    scroll_main(page, 260)
    page.screenshot(path=IMAGES / f"anomalies-{mode}.png")

    go("Dimensions", "Analysis")
    time.sleep(2)
    scroll_main(page, 120)
    page.screenshot(path=IMAGES / f"dimensions-{mode}.png")

    go("Compare data", "Analysis")
    time.sleep(2)
    page.screenshot(path=IMAGES / f"compare-{mode}.png")

    go("Neural network", "Models")
    pick(page, "Column to predict", "revenue")
    page.get_by_role("button", name="Train network").click()
    time.sleep(14)
    scroll_main(page, 720)
    page.screenshot(path=IMAGES / f"neural-{mode}.png")

    go("Time series", "Models")
    pick(page, "Value", "revenue")
    page.get_by_role("button", name="Analyze").click()
    time.sleep(6)
    scroll_main(page, 200)
    page.screenshot(path=IMAGES / f"timeseries-{mode}.png")

    go("Model lab", "Models")
    pick(page, "Column to predict", "revenue")
    page.get_by_role("button", name="Run the lab").click()
    page.get_by_text("Download leaderboard").wait_for(timeout=300000)
    time.sleep(2)
    page.mouse.move(5, 5)   # sin tooltips encima
    scroll_main(page, 480)
    page.screenshot(path=IMAGES / f"lab-leaderboard-{mode}.png")
    page.get_by_role("tab", name="Diagnostics").click()
    time.sleep(3)
    scroll_main(page, 640)
    page.screenshot(path=IMAGES / f"lab-diagnostics-{mode}.png")

    go("Predict", "Models")
    page.get_by_role("button", name="Predict", exact=True).first.click()
    time.sleep(5)
    scroll_main(page, 220)
    page.screenshot(path=IMAGES / f"predict-{mode}.png")
    page.close()


def main() -> None:
    from kikolens.report.builder import save_report
    from kikolens.utils.loaders import load_data

    make_demo_data()
    IMAGES.mkdir(parents=True, exist_ok=True)
    save_report(load_data(str(CSV)), ROOT / "examples" / "report.html", name="sales_demo", offline=False)

    server = start_server()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:
                browser = p.chromium.launch(channel="msedge")
            for mode in ("light", "dark"):
                capture(browser, mode)
            browser.close()
    finally:
        stop_server(server)
    print(f"Wrote {len(list(IMAGES.glob('*.png')))} images to {IMAGES}")


if __name__ == "__main__":
    main()
