"""Datos de ejemplo generados al vuelo (sin archivos) para probar la aplicación: ventas, reseñas, pedidos de clientes y suscripciones.
Son sintéticos y deterministas, con defectos y patrones puestos a propósito."""
import numpy as np
import pandas as pd

DESCRIPTIONS = {
    "sales": "Daily sales with a weekly pattern, a trend, gaps and prices stored as text.",
    "reviews": "Customer comments with a rating: shipping, quality and price themes.",
    "orders": "Orders by customer over six months: retention, segments and funnels.",
    "subscriptions": "Subscribers with a plan and whether they left: survival curves.",
}


def sales(seed: int = 11, n: int = 540) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], n, p=[0.35, 0.25, 0.25, 0.15]),
        "channel": rng.choice(["Online", "Store"], n, p=[0.6, 0.4]),
        "units": rng.integers(20, 160, n).astype(float),
        "unit_price": [f"{v:,.0f}" for v in rng.integers(1200, 4800, n)],
        "discount": rng.choice([0, 0.05, 0.1, 0.15, 0.2], n),
    })
    df["revenue"] = (1500 + 1.3 * days + 220 * np.sin(2 * np.pi * days / 7) + 140 * np.sin(2 * np.pi * days / 365)
                     + rng.normal(0, 70, n)).round(2)
    df["units"] = (df["revenue"] / 14 + rng.normal(0, 6, n)).round(0).clip(lower=1)
    df.loc[rng.choice(n, 28, replace=False), "discount"] = np.nan
    return pd.concat([df, df.sample(7, random_state=1)], ignore_index=True)


def reviews(seed: int = 5, n: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    themes = {
        "shipping": ["The delivery was late and the package arrived damaged", "Shipping took three weeks and nobody told me why",
                     "Tracking said delivered but I never received it", "Slow delivery, the box was crushed when it finally came"],
        "quality": ["The battery lasts all week and the screen is beautiful", "Great build quality, it feels solid and premium",
                    "Excellent camera, the photos look amazing in low light", "Easy to set up and it works perfectly every day"],
        "price": ["Too expensive for what it offers, I expected more", "The price went up again and discounts are rare",
                  "Overpriced compared with similar models from other brands", "Not worth the cost, the accessories are extra"],
    }
    base = {"shipping": 2, "quality": 5, "price": 3}
    rows = []
    for _ in range(n):
        theme = str(rng.choice(list(themes), p=[0.3, 0.45, 0.25]))
        rows.append({"review": str(rng.choice(themes[theme])), "rating": int(np.clip(base[theme] + rng.integers(-1, 2), 1, 5)),
                     "channel": str(rng.choice(["Online", "Store"], p=[0.7, 0.3])), "product": str(rng.choice(["Phone", "Tablet", "Watch"]))})
    return pd.DataFrame(rows)


def orders(seed: int = 3, customers: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for customer in range(1, customers + 1):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 180)))
        loyal = rng.random() < 0.4
        for k in range(int(rng.integers(1, 9 if loyal else 3))):
            day = start + pd.Timedelta(days=int(30 * k * rng.uniform(0.7, 1.3)))
            rows.append({"customer_id": customer, "date": day.strftime("%Y-%m-%d"), "amount": round(float(rng.gamma(3, 25 if loyal else 15)), 2),
                         "category": str(rng.choice(["Home", "Sport", "Books", "Garden"]))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def subscriptions(seed: int = 7, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["Basic", "Pro"], n, p=[0.6, 0.4])
    lifetime = rng.exponential(np.where(plan == "Pro", 18, 8))
    watched = rng.uniform(2, 24, n)
    return pd.DataFrame({"tenure_months": np.minimum(lifetime, watched).round(2), "churned": (lifetime <= watched).astype(int),
                         "plan": plan, "seats": rng.integers(1, 20, n)})


GENERATORS = {"sales": sales, "reviews": reviews, "orders": orders, "subscriptions": subscriptions}


def load_demo(name: str) -> pd.DataFrame:
    if name not in GENERATORS:
        raise ValueError(f"Unknown example '{name}'. Choose from {', '.join(GENERATORS)}.")
    return GENERATORS[name]()
