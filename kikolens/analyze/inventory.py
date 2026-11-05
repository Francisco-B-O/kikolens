"""Planificación de inventario y demanda.

  * `classify_items`: ABC (cuánto valor aporta cada artículo) y XYZ (cuánto varía su demanda), con la clase de demanda de Syntetos-Boylan
    (regular, errática, intermitente, a rachas) y qué política le suele convenir.
  * `newsvendor`: cuánto pedir de algo que se vende una sola vez (temporada, perecedero), con la razón crítica.
  * `eoq`: el tamaño de pedido que minimiza coste de pedir más coste de tener stock.
  * `reorder_policy`: stock de seguridad y punto de pedido para un nivel de servicio, con demanda y plazo de entrega variables.
  * `simulate_policy`: comprueba una política (s, Q) simulando miles de días, para ver el servicio real y no solo el de la fórmula.
  * `plan`: lo anterior para cada artículo de una tabla de ventas.
Las fórmulas suponen demanda aproximadamente normal e independiente entre periodos; en artículos intermitentes la simulación es más fiable."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

ABC_CUTS = (0.80, 0.95)
XYZ_CUTS = (0.5, 1.0)
ADI_CUT, CV2_CUT = 1.32, 0.49


def _positive(**values) -> None:
    for name, v in values.items():
        if v is None or not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a positive number.")


def _loss(z: float) -> float:
    """Función de pérdida estándar de la normal: E[(Z - z)+]."""
    return float(stats.norm.pdf(z) - z * stats.norm.sf(z))


# ── Clasificación ────────────────────────────────────────────────────────────
def demand_class(adi: float, cv2: float) -> str:
    if adi <= ADI_CUT:
        return "smooth" if cv2 <= CV2_CUT else "erratic"
    return "intermittent" if cv2 <= CV2_CUT else "lumpy"


POLICY = {"smooth": "Forecast it and replenish with a safety stock.", "erratic": "Needs a larger safety stock; look at what drives the swings.",
          "intermittent": "Do not forecast the average: use a simulation or order on demand.", "lumpy": "Hard to plan: keep little stock, or make to order."}


def classify_items(df: pd.DataFrame, item: str, quantity: str, date: str | None = None, value: str | None = None, freq: str = "W") -> dict:
    """ABC por valor acumulado (80 % / 15 % / 5 %) y, con fecha, XYZ por coeficiente de variación de la demanda por periodo y clase de Syntetos-Boylan."""
    for c in (item, quantity) + tuple(x for x in (date, value) if x):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    for c in (quantity,) + ((value,) if value else ()):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be numeric.")
    data = df.dropna(subset=[item, quantity] + ([date] if date else []) + ([value] if value else [])).copy()
    if (data[quantity] < 0).any():
        raise ValueError("Quantities cannot be negative.")
    data["_value"] = data[value] if value else data[quantity]
    total = data.groupby(item)["_value"].sum().sort_values(ascending=False)
    if len(total) < 3:
        raise ValueError("At least 3 items are needed.")
    if total.sum() <= 0:
        raise ValueError("The total value is zero.")
    share = total / total.sum()
    cumulative = share.cumsum()
    abc = np.where(cumulative.shift(fill_value=0) < ABC_CUTS[0], "A", np.where(cumulative.shift(fill_value=0) < ABC_CUTS[1], "B", "C"))
    table = pd.DataFrame({"item": total.index, "value": total.values, "share": share.values, "cumulative share": cumulative.values, "ABC": abc})
    notes = []
    if date:
        data[date] = pd.to_datetime(data[date], errors="coerce")
        data = data.dropna(subset=[date])
        periods = data[date].dt.to_period(freq)
        index = pd.period_range(periods.min(), periods.max(), freq=freq)
        if len(index) < 6:
            raise ValueError("At least 6 periods are needed to judge how much demand varies; use a shorter period.")
        wide = data.groupby([periods, item])[quantity].sum().unstack(fill_value=0).reindex(index, fill_value=0)
        mean, sd = wide.mean(), wide.std(ddof=1)
        cv = (sd / mean.replace(0, np.nan)).reindex(table["item"]).to_numpy()
        nonzero = (wide > 0).sum().reindex(table["item"])
        adi = (len(index) / nonzero.replace(0, np.nan)).to_numpy()
        sizes_mean = wide.where(wide > 0).mean().reindex(table["item"])
        sizes_sd = wide.where(wide > 0).std(ddof=1).reindex(table["item"])
        cv2 = ((sizes_sd / sizes_mean) ** 2).fillna(0).to_numpy()
        table["periods with demand"] = nonzero.to_numpy()
        table["average demand per period"] = mean.reindex(table["item"]).to_numpy()
        table["variation (CV)"] = cv
        table["XYZ"] = np.where(cv < XYZ_CUTS[0], "X", np.where(cv < XYZ_CUTS[1], "Y", "Z"))
        table["average gap between orders (ADI)"] = adi
        table["demand class"] = [demand_class(a, c) if np.isfinite(a) else "no demand" for a, c in zip(adi, cv2)]
        table["suggested approach"] = table["demand class"].map(POLICY).fillna("")
        matrix = pd.crosstab(table["ABC"], table["XYZ"]).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        if (table["demand class"].isin(["intermittent", "lumpy"])).mean() > 0.3:
            notes.append(f"{(table['demand class'].isin(['intermittent', 'lumpy'])).mean():.0%} of the items sell on only a few periods: averages and normal-based safety stocks mislead for them.")
        az = int(((table["ABC"] == "A") & (table["XYZ"] == "Z")).sum())
        if az:
            notes.append(f"{az} high-value item(s) with erratic demand (A and Z): the costliest place to be wrong; look at them one by one.")
    else:
        matrix = None
    a_items = int((table["ABC"] == "A").sum())
    return {"table": table, "matrix": matrix, "notes": notes, "items": len(table), "a_items": a_items, "freq": freq,
            "figure": go.Figure(go.Scatter(x=np.arange(1, len(table) + 1) / len(table), y=table["cumulative share"], mode="lines")).update_layout(
                title="Share of the total value covered by the top items (Pareto)", xaxis_title="Share of items", yaxis_title="Share of value", xaxis_tickformat=".0%", yaxis_tickformat=".0%"),
            "summary": (f"{a_items} of {len(table)} items ({a_items / len(table):.0%}) make up {ABC_CUTS[0]:.0%} of the value (class A); "
                        f"{int((table['ABC'] == 'C').sum())} are class C.")}


# ── Vendedor de periódicos ───────────────────────────────────────────────────
def newsvendor(unit_cost: float, price: float, salvage: float = 0.0, mean: float | None = None, sd: float | None = None, samples=None) -> dict:
    """Cuánto pedir de un producto de una sola venta. La razón crítica `(precio - coste) / (precio - residual)` es la probabilidad de cubrir la demanda que conviene tener.
    Con `mean` y `sd` supone demanda normal; con `samples` (ventas de temporadas parecidas) usa la distribución observada."""
    _positive(unit_cost=unit_cost, price=price)
    if salvage < 0 or salvage >= unit_cost:
        raise ValueError("The salvage value must be at least 0 and below the unit cost (otherwise ordering more can never lose).")
    if price <= unit_cost:
        raise ValueError("The price must be above the unit cost.")
    underage, overage = price - unit_cost, unit_cost - salvage
    ratio = underage / (underage + overage)
    if samples is not None:
        d = np.asarray(pd.Series(samples).dropna(), dtype=float)
        if len(d) < 10:
            raise ValueError("At least 10 past demands are needed.")
        dist = "observed"
        q_star = float(np.quantile(d, ratio))
        grid = np.linspace(max(0, d.min() * 0.5), d.max() * 1.3, 120)
        profit = np.array([float(np.mean(price * np.minimum(q, d) + salvage * np.maximum(q - d, 0) - unit_cost * q)) for q in grid])
        sold = float(np.mean(np.minimum(q_star, d)))
        left = float(np.mean(np.maximum(q_star - d, 0)))
        short = float(np.mean(np.maximum(d - q_star, 0)))
        expected = float(np.mean(price * np.minimum(q_star, d) + salvage * np.maximum(q_star - d, 0) - unit_cost * q_star))
        mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
        p_cover = float(np.mean(d <= q_star))
    else:
        if mean is None or sd is None:
            raise ValueError("Give the mean and standard deviation of demand, or past demands in samples.")
        _positive(mean=mean)
        if sd < 0:
            raise ValueError("The standard deviation cannot be negative.")
        dist, mean_d, sd_d = "normal", float(mean), float(sd)
        z = float(stats.norm.ppf(ratio))
        q_star = mean_d + z * sd_d
        grid = np.linspace(max(0, mean_d - 3.5 * max(sd_d, 1e-9)), mean_d + 3.5 * max(sd_d, 1e-9), 120)

        def exp_short(q):
            zq = (q - mean_d) / sd_d if sd_d > 0 else (np.inf if q >= mean_d else -np.inf)
            return sd_d * _loss(zq) if np.isfinite(zq) else max(mean_d - q, 0)

        def exp_profit(q):
            short_q = exp_short(q)
            sold_q = mean_d - short_q
            return price * sold_q + salvage * (q - sold_q) - unit_cost * q

        profit = np.array([exp_profit(q) for q in grid])
        short = exp_short(q_star)
        sold = mean_d - short
        left = q_star - sold
        expected = exp_profit(q_star)
        p_cover = ratio
    notes = []
    if dist == "normal" and mean_d > 0 and sd_d / mean_d > 0.5:
        notes.append("Demand varies by more than half its mean: a normal distribution would allow negative demand, so check it against past seasons (use samples=).")
    if q_star < 0:
        q_star = 0.0
    fig = go.Figure(go.Scatter(x=grid, y=profit, mode="lines", name="Expected profit"))
    fig.add_vline(x=q_star, line_dash="dash", annotation_text="best order")
    fig.update_layout(title="Expected profit by quantity ordered", xaxis_title="Quantity ordered", yaxis_title="Expected profit")
    return {"quantity": float(q_star), "critical_ratio": float(ratio), "expected_profit": float(expected), "expected_sold": float(sold), "expected_left_over": float(left),
            "expected_short": float(short), "probability_cover": float(p_cover), "distribution": dist, "mean_demand": mean_d, "sd_demand": sd_d, "notes": notes, "figure": fig,
            "table": pd.DataFrame({"quantity": grid, "expected profit": profit}),
            "summary": (f"Order {q_star:,.1f}: it covers demand {p_cover:.0%} of the time (critical ratio {ratio:.2f}), with an expected profit of {expected:,.1f}, "
                        f"{left:,.1f} left over and {short:,.1f} of unmet demand on average.")}


# ── Lote económico ───────────────────────────────────────────────────────────
def eoq(annual_demand: float, order_cost: float, holding_cost: float, unit_cost: float = 0.0) -> dict:
    """Lote económico de pedido `sqrt(2 D S / H)`. `holding_cost` es el coste de tener una unidad un año."""
    _positive(annual_demand=annual_demand, order_cost=order_cost, holding_cost=holding_cost)
    q = math.sqrt(2 * annual_demand * order_cost / holding_cost)
    orders = annual_demand / q
    ordering, holding = orders * order_cost, q / 2 * holding_cost
    grid = np.linspace(q * 0.2, q * 3, 120)
    total = annual_demand / grid * order_cost + grid / 2 * holding_cost
    fig = go.Figure([go.Scatter(x=grid, y=total, name="Total"), go.Scatter(x=grid, y=annual_demand / grid * order_cost, name="Ordering", line=dict(dash="dot")),
                     go.Scatter(x=grid, y=grid / 2 * holding_cost, name="Holding", line=dict(dash="dot"))])
    fig.add_vline(x=q, line_dash="dash", annotation_text="EOQ")
    fig.update_layout(title="Yearly cost by order size", xaxis_title="Units per order", yaxis_title="Cost per year")
    notes = ["The cost is flat near the optimum: ordering 20% more or less changes the total by about 2%, so round to a practical pack size."]
    return {"quantity": q, "orders_per_year": orders, "days_between_orders": 365 / orders, "ordering_cost": ordering, "holding_cost_total": holding, "total_cost": ordering + holding + annual_demand * unit_cost,
            "notes": notes, "figure": fig,
            "summary": f"Order {q:,.1f} units about {orders:,.1f} times a year (every {365 / orders:,.0f} days); ordering and holding cost {ordering + holding:,.0f} a year."}


# ── Stock de seguridad y punto de pedido ─────────────────────────────────────
def reorder_policy(demand_mean: float, demand_sd: float, lead_time: float, lead_time_sd: float = 0.0, service_level: float = 0.95, order_quantity: float | None = None) -> dict:
    """Stock de seguridad `z * sqrt(L sd_d^2 + d^2 sd_L^2)` y punto de pedido `d L + SS`. `demand_*` por periodo y `lead_time` en los mismos periodos.
    El nivel de servicio es la probabilidad de no quedarse sin stock en un ciclo; con `order_quantity` se calcula también la tasa de servicio (fracción de la demanda servida)."""
    _positive(demand_mean=demand_mean, lead_time=lead_time)
    if demand_sd < 0 or lead_time_sd < 0:
        raise ValueError("Standard deviations cannot be negative.")
    if not 0.5 <= service_level < 1:
        raise ValueError("The service level must be at least 0.5 and below 1.")
    z = float(stats.norm.ppf(service_level))
    sd_lead = math.sqrt(lead_time * demand_sd ** 2 + demand_mean ** 2 * lead_time_sd ** 2)
    safety = z * sd_lead
    rop = demand_mean * lead_time + safety
    fill = None
    if order_quantity:
        _positive(order_quantity=order_quantity)
        fill = float(1 - sd_lead * _loss(z) / order_quantity) if sd_lead > 0 else 1.0
    notes = []
    if lead_time_sd > 0 and demand_mean * lead_time_sd > math.sqrt(lead_time) * demand_sd:
        notes.append("Variation in the delivery time matters more than variation in demand: working with the supplier on reliability saves more stock than a better forecast.")
    if demand_sd > demand_mean:
        notes.append("Demand varies by more than its mean, so it is intermittent or very uneven: a normal-based safety stock is only a rough guide; check it with simulate_policy.")
    if fill is not None and fill < service_level:
        notes.append(f"The fill rate ({fill:.1%}) is lower than the cycle service level ({service_level:.0%}) because the order quantity is small compared with the demand in the lead time.")
    return {"safety_stock": float(safety), "reorder_point": float(rop), "z": z, "lead_time_demand_sd": float(sd_lead), "fill_rate": fill, "service_level": service_level,
            "average_inventory": float((order_quantity or 0) / 2 + safety), "notes": notes,
            "summary": (f"Reorder when stock falls to {rop:,.1f} (average use during the lead time {demand_mean * lead_time:,.1f} plus a safety stock of {safety:,.1f}) "
                        f"to avoid a stockout in {service_level:.0%} of cycles" + (f", serving about {fill:.1%} of demand." if fill is not None else "."))}


def simulate_policy(demand_mean: float, demand_sd: float, lead_time: int, order_quantity: float, reorder_point: float, lead_time_sd: float = 0.0, days: int = 365,
                    runs: int = 300, lost_sales: bool = False, steps_per_day: int = 8, seed: int = 0) -> dict:
    """Simula una política (s, Q) con demanda normal (sin negativos) y entregas con plazo variable: servicio real, stock medio y pedidos al año.
    Cada día se divide en `steps_per_day` pasos para que el pedido salga cuando el stock cruza `s` y no al final del día, que lo retrasaría de media medio día de demanda."""
    _positive(demand_mean=demand_mean, order_quantity=order_quantity)
    if demand_sd < 0 or lead_time_sd < 0 or lead_time < 1:
        raise ValueError("The lead time must be at least 1 day and the standard deviations cannot be negative.")
    if not 30 <= days <= 3650 or not 10 <= runs <= 5000 or not 1 <= steps_per_day <= 48:
        raise ValueError("days must be between 30 and 3650, runs between 10 and 5000 and steps_per_day between 1 and 48.")
    rng = np.random.default_rng(seed)
    ns = int(steps_per_day)
    slots = int(math.ceil((demand_mean * lead_time + 4 * demand_sd * math.sqrt(lead_time)) / order_quantity)) + 3        # pedidos abiertos a la vez que puede haber
    if slots > 200:
        raise ValueError("The order quantity is too small compared with the demand during the lead time (it would need more than 200 open orders at once); use a larger order quantity.")
    stock = np.full(runs, reorder_point + order_quantity, float)
    pipe_t = np.full((runs, slots), np.inf)
    pipe_q = np.zeros((runs, slots))
    short = np.zeros(runs, bool)
    served = demanded = onhand = 0.0
    cycles = stockout_cycles = orders = 0
    daily = np.zeros(runs)
    for step in range(days * ns):
        if step % ns == 0:                                                   # la demanda se sortea por día (sin negativos) y se reparte entre los pasos del día
            daily = np.maximum(0.0, rng.normal(demand_mean, demand_sd, runs))
        demand = daily / ns
        sold = np.where(stock > 0, np.minimum(stock, demand), 0.0)
        served += sold.sum()
        demanded += demand.sum()
        short |= demand > sold + 1e-12
        stock -= sold
        if not lost_sales:
            stock -= demand - sold                                           # pedidos pendientes: el stock puede quedar negativo
        arrive = pipe_t <= step                                              # las entregas llegan al final de su paso, tras la demanda de ese paso
        has = arrive.any(axis=1)
        if has.any():
            stock += (pipe_q * arrive).sum(axis=1)
            pipe_t[arrive], pipe_q[arrive] = np.inf, 0.0
            cycles += int(has.sum())
            stockout_cycles += int((short & has).sum())
            short[has] = False
        position = stock + pipe_q.sum(axis=1)
        need = position <= reorder_point
        while need.any():
            who = np.where(need)[0]
            free = np.isinf(pipe_t[who])
            if not free.any(axis=1).all():
                raise ValueError("The order quantity is too small for this reorder point: more orders would be open at once than the simulation can track.")
            slot = free.argmax(axis=1)
            days_to_arrive = np.maximum(1, np.rint(rng.normal(lead_time, lead_time_sd, len(who)))) if lead_time_sd > 0 else np.full(len(who), lead_time)
            pipe_t[who, slot] = step + days_to_arrive * ns
            pipe_q[who, slot] = order_quantity
            position[who] += order_quantity
            orders += len(who)
            need = position <= reorder_point
        onhand += np.maximum(stock, 0.0).sum() / ns
    fill = served / demanded if demanded else 1.0
    cycle_service = 1 - stockout_cycles / cycles if cycles else 1.0
    return {"fill_rate": float(fill), "cycle_service_level": float(cycle_service), "average_on_hand": float(onhand / (runs * days)), "orders_per_year": float(orders / runs * 365 / days),
            "cycles": cycles, "runs": runs, "days": days,
            "summary": f"Over {runs} simulated years: {cycle_service:.1%} of replenishment cycles ended without a stockout, {fill:.1%} of demand was served and the average stock on hand was "
                       f"{onhand / (runs * days):,.1f} units, with {orders / runs * 365 / days:,.1f} orders a year."}


# ── Plan de todos los artículos ──────────────────────────────────────────────
def plan(df: pd.DataFrame, item: str, date: str, quantity: str, lead_time_days: float, service_level: float = 0.95, order_cost: float | None = None, holding_cost: float | None = None,
         lead_time_sd_days: float = 0.0, freq: str = "D") -> dict:
    """Para cada artículo de una tabla de ventas: demanda media y variación por día (o semana), stock de seguridad, punto de pedido y, si das costes, lote económico."""
    for c in (item, date, quantity):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    if freq not in ("D", "W"):
        raise ValueError("freq must be 'D' (daily) or 'W' (weekly).")
    data = df.dropna(subset=[item, date, quantity]).copy()
    data[date] = pd.to_datetime(data[date], errors="coerce")
    data = data.dropna(subset=[date])
    periods = data[date].dt.to_period(freq)
    index = pd.period_range(periods.min(), periods.max(), freq=freq)
    if len(index) < 14:
        raise ValueError("At least 14 periods of history are needed.")
    wide = data.groupby([periods, item])[quantity].sum().unstack(fill_value=0).reindex(index, fill_value=0)
    per_day = 1 if freq == "D" else 7
    lead = lead_time_days / per_day
    lead_sd = lead_time_sd_days / per_day
    rows = []
    for name in wide.columns:
        d = wide[name]
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        if mean <= 0:
            continue
        pol = reorder_policy(mean, sd, lead, lead_sd, service_level)
        row = {"item": name, f"demand per {'day' if freq == 'D' else 'week'}": mean, "variation (CV)": sd / mean, "safety stock": pol["safety_stock"], "reorder point": pol["reorder_point"]}
        if order_cost and holding_cost:
            e = eoq(mean * 365 / per_day, order_cost, holding_cost)
            row["order quantity (EOQ)"] = e["quantity"]
            row["orders per year"] = e["orders_per_year"]
        rows.append(row)
    if not rows:
        raise ValueError("No item has any demand.")
    table = pd.DataFrame(rows).sort_values("reorder point", ascending=False, ignore_index=True)
    erratic = int((table["variation (CV)"] > 1).sum())
    notes = []
    if erratic:
        notes.append(f"{erratic} item(s) have a coefficient of variation above 1 (very uneven demand): check their policy with simulate_policy before trusting the safety stock.")
    return {"table": table, "notes": notes, "service_level": service_level, "lead_time_days": lead_time_days,
            "summary": f"A reorder policy for {len(table)} item(s) at a {service_level:.0%} service level with a {lead_time_days:g}-day lead time; total safety stock {table['safety stock'].sum():,.0f} units."}
