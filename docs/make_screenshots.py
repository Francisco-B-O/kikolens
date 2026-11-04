"""Regenera los ejemplos (examples/*.csv, report.html) y las capturas de docs/images.

Uso:  pip install -e ".[docs]" && python -m playwright install chromium   (o usa Edge/Chrome instalado)
      python docs/make_screenshots.py
"""
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
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
REVIEWS_CSV = ROOT / "examples" / "reviews.csv"
ORDERS_CSV = ROOT / "examples" / "orders.csv"
SUBSCRIPTIONS_CSV = ROOT / "examples" / "subscriptions.csv"
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
    make_reviews(rng)
    make_behaviour(rng)


def make_behaviour(rng) -> None:
    """Pedidos de clientes (retención, RFM) y suscripciones con baja (supervivencia)."""
    rows = []
    for customer in range(1, 401):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 180)))
        loyal = rng.random() < 0.4
        for k in range(int(rng.integers(1, 9 if loyal else 3))):
            day = start + pd.Timedelta(days=int(30 * k * rng.uniform(0.7, 1.3)))
            rows.append({"customer_id": customer, "date": day.strftime("%Y-%m-%d"),
                         "amount": round(float(rng.gamma(3, 25 if loyal else 15)), 2),
                         "category": str(rng.choice(["Home", "Sport", "Books", "Garden"]))})
    pd.DataFrame(rows).sort_values("date").to_csv(ORDERS_CSV, index=False)
    n = 500
    plan = rng.choice(["Basic", "Pro"], n, p=[0.6, 0.4])
    lifetime = rng.exponential(np.where(plan == "Pro", 18, 8))
    watched = rng.uniform(2, 24, n)
    pd.DataFrame({
        "tenure_months": np.minimum(lifetime, watched).round(2), "churned": (lifetime <= watched).astype(int), "plan": plan,
        "seats": rng.integers(1, 20, n),
    }).to_csv(SUBSCRIPTIONS_CSV, index=False)


def make_reviews(rng) -> None:
    """Comentarios de clientes con tres temas claros (envío, calidad, precio), una valoración y un canal."""
    themes = {
        "shipping": ["The delivery was late and the package arrived damaged", "Shipping took three weeks and nobody told me why",
                     "The courier left the parcel outside and it got soaked", "Tracking said delivered but I never received it",
                     "Slow delivery, the box was crushed when it finally came", "Customer service ignored my emails about the delay"],
        "quality": ["The battery lasts all week and the screen is beautiful", "Great build quality, it feels solid and premium",
                    "Excellent camera, the photos look amazing in low light", "The device is fast and the display is really sharp",
                    "Very happy with the sound, much better than I expected", "Easy to set up and it works perfectly every day"],
        "price": ["Too expensive for what it offers, I expected more", "The price went up again and discounts are rare",
                  "Overpriced compared with similar models from other brands", "Not worth the cost, the accessories are extra",
                  "Good product but the subscription fee is a rip off", "I asked for a refund because it was not worth that price"],
    }
    extras = {
        "shipping": ["and support never answered", "so I had to chase the courier", "and the tracking page was wrong",
                     "which ruined the gift", "and nobody apologised", "and the box was already open"],
        "quality": ["and I would buy it again", "so my family wants one too", "and the battery is a bonus",
                    "which surprised me", "and the setup took five minutes", "and it looks great on the desk"],
        "price": ["so I am waiting for a sale", "and a competitor is cheaper", "which feels unfair",
                  "and the extras add up", "so I returned it", "and the warranty costs more"],
    }
    openers = ["", "Honestly, ", "To be fair, ", "After two weeks: ", "Update: "]
    base = {"shipping": 2, "quality": 5, "price": 3}
    rows = []
    for i in range(360):
        theme = rng.choice(list(themes), p=[0.3, 0.45, 0.25])
        rating = int(np.clip(base[theme] + rng.integers(-1, 2), 1, 5))
        text = f"{rng.choice(openers)}{rng.choice(themes[theme])} {rng.choice(extras[theme])}"
        rows.append({"review": text[0].upper() + text[1:], "rating": rating,
                     "channel": str(rng.choice(["Online", "Store"], p=[0.7, 0.3])),
                     "product": str(rng.choice(["Phone", "Tablet", "Watch"]))})
    pd.DataFrame(rows).to_csv(REVIEWS_CSV, index=False)


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
        env={**os.environ, "STREAMLIT_SERVER_HEADLESS": "true"})
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
    stop_server(proc)
    raise RuntimeError("The dashboard did not start.")


def prepare_workspace() -> Path:
    """Una carpeta de datos propia, con un modelo guardado y un monitor con historial, para mostrarlos en las capturas."""
    import warnings

    warnings.simplefilter("ignore")
    from kikolens.analyze import monitor as mt
    from kikolens.analyze import predict as pr
    from kikolens.analyze.modellab import run_model_lab
    from kikolens.analyze.quality import auto_clean_steps
    from kikolens.pipeline.recipe import Recipe
    from kikolens.utils.loaders import load_data

    home = Path(tempfile.mkdtemp(prefix="kikolens_shots_"))
    os.environ["KIKOLENS_HOME"] = str(home)
    raw = load_data(str(CSV))
    clean, steps = auto_clean_steps(raw)
    recipe = Recipe()
    for op, params, message in steps:
        recipe.add(op, params, message)
    res = run_model_lab(clean, "revenue", models=["Ridge", "Random Forest", "Gradient Boosting"], folds=3, tune=False,
                        drop=["units"])
    pr.save_bundle(pr.build_bundle(res, clean, name="sales model"))
    mon = mt.create_monitor("sales feed", "(uploaded)", raw, recipe, model="sales-model")
    mt.run_check(mon, raw.sample(frac=0.8, random_state=1))
    mt.run_check(mon, raw.sample(frac=0.9, random_state=2))
    mt.run_check(mon, load_data(str(NEW_CSV)))
    return home


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
        nav.locator('[data-testid="stRadio"]').first.get_by_text(area, exact=True).click()   # el selector de área, no la etiqueta "Data"
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

    go("Autopilot", "Data")
    page.get_by_role("button", name="Analyze everything").click()
    page.get_by_text("Suggested next steps").wait_for(timeout=120000)
    time.sleep(2)
    page.mouse.move(5, 5)
    page.screenshot(path=IMAGES / f"autopilot-{mode}.png")

    go("Impact", "Analysis")
    pick(page, "Measure", "revenue")
    page.get_by_role("button", name="Estimate the effect").click()
    time.sleep(8)
    page.mouse.move(5, 5)
    scroll_main(page, 160)
    page.screenshot(path=IMAGES / f"impact-{mode}.png")

    go("Why it changed", "Analysis")
    pick(page, "Column", "revenue")
    page.get_by_role("button", name="Explain the change").click()
    time.sleep(6)
    page.mouse.move(5, 5)
    scroll_main(page, 120)
    page.screenshot(path=IMAGES / f"why-{mode}.png")

    go("Panel", "Share")
    page.get_by_role("button", name="Suggest a panel").click()
    time.sleep(6)
    page.mouse.move(5, 5)
    page.screenshot(path=IMAGES / f"panel-{mode}.png")

    go("Export", "Share")
    page.get_by_role("tab", name="Excel workbook").click()
    time.sleep(2)
    page.screenshot(path=IMAGES / f"export-{mode}.png")

    go("Monitoring", "Models")
    time.sleep(3)
    scroll_main(page, 120)
    page.screenshot(path=IMAGES / f"monitoring-{mode}.png")

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

    page.get_by_role("tab", name="Errors").click()
    time.sleep(4)
    scroll_main(page, 640)
    page.screenshot(path=IMAGES / f"lab-errors-{mode}.png")

    page.get_by_role("tab", name="Uncertainty").click()
    time.sleep(5)
    page.mouse.move(5, 5)
    scroll_main(page, 640)
    page.screenshot(path=IMAGES / f"lab-uncertainty-{mode}.png")

    go("Predict", "Models")
    page.get_by_role("button", name="Predict", exact=True).first.click()
    time.sleep(5)
    scroll_main(page, 220)
    page.screenshot(path=IMAGES / f"predict-{mode}.png")

    page.set_input_files('input[type="file"]', str(REVIEWS_CSV))   # otro dataset, para la sección de texto
    time.sleep(5)
    go("Text", "Analysis")
    time.sleep(3)
    scroll_main(page, 120)
    page.screenshot(path=IMAGES / f"text-{mode}.png")

    page.set_input_files('input[type="file"]', str(ORDERS_CSV))   # clientes: retención y segmentos
    time.sleep(5)
    go("Customers", "Analysis")
    panel = page.locator('[role="tabpanel"]:visible')
    panel.get_by_role("button", name="Run").click()
    time.sleep(5)
    page.mouse.move(5, 5)
    page.screenshot(path=IMAGES / f"customers-{mode}.png")

    page.set_input_files('input[type="file"]', str(SUBSCRIPTIONS_CSV))   # bajas: supervivencia
    time.sleep(5)
    go("Survival", "Analysis")
    pick(page, "Time until the event or until last seen", "tenure_months")
    pick(page, "Compare groups by", "plan")
    page.get_by_role("button", name="Draw the curves").click()
    time.sleep(6)
    page.mouse.move(5, 5)
    page.screenshot(path=IMAGES / f"survival-{mode}.png")
    page.close()


def capture_spanish(browser) -> None:
    """Una captura con la interfaz en español."""
    page = browser.new_page(viewport=VIEW, color_scheme="light")
    page.goto(f"http://localhost:{PORT}/")
    page.wait_for_selector("text=KikoLens", timeout=30000)
    page.set_input_files('input[type="file"]', str(CSV))
    page.wait_for_selector("text=Overview", timeout=30000)
    time.sleep(3)
    pick(page, "Language / Idioma", "Español")
    time.sleep(3)
    page.mouse.move(5, 5)
    page.screenshot(path=IMAGES / "spanish-light.png")
    page.close()


def main() -> None:
    from kikolens.report.builder import save_report
    from kikolens.utils.loaders import load_data

    make_demo_data()
    workspace = prepare_workspace()
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
            capture_spanish(browser)
            browser.close()
    finally:
        stop_server(server)
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"Wrote {len(list(IMAGES.glob('*.png')))} images to {IMAGES}")


if __name__ == "__main__":
    main()
