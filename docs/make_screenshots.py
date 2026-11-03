"""Regenera examples/sales_demo.csv, examples/report.html y las capturas de docs/images.

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

    def go(name):
        nav.get_by_text(name, exact=True).first.click()
        time.sleep(3)

    go("Overview")
    page.screenshot(path=IMAGES / f"overview-{mode}.png")

    go("Quality and cleaning")
    page.screenshot(path=IMAGES / f"quality-{mode}.png")

    go("Explore")
    pick(page, "Chart", "Line")
    pick(page, "X", "date")
    pick(page, "Y", "revenue")
    page.screenshot(path=IMAGES / f"explore-{mode}.png")

    go("Neural network")
    pick(page, "Column to predict", "revenue")
    page.get_by_role("button", name="Train network").click()
    time.sleep(14)
    scroll_main(page, 720)
    page.screenshot(path=IMAGES / f"neural-{mode}.png")

    go("Time series")
    pick(page, "Value", "revenue")
    page.get_by_role("button", name="Analyze").click()
    time.sleep(6)
    scroll_main(page, 200)
    page.screenshot(path=IMAGES / f"timeseries-{mode}.png")
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
