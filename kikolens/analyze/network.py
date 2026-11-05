"""Análisis de redes a partir de una lista de enlaces (origen, destino, peso opcional): quién es importante, qué grupos hay, cómo de conectada está y qué pasaría si
faltaran los nodos clave. Sin dependencias nuevas: matrices dispersas de SciPy y los algoritmos clásicos escritos aquí (PageRank, Louvain, Brandes).

  * `analyze`: grado, PageRank, intermediación, agrupamiento y comunidades por nodo, y las medidas de la red entera.
  * `shortest_path`: el camino más corto (en saltos) entre dos nodos.
  * `suggest_links`: enlaces que faltan, según los vecinos que ya comparten dos nodos (Adamic-Adar).
  * `robustness`: cuánto se rompe la red al quitar los nodos más conectados, comparado con quitar nodos al azar."""
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse
from scipy.sparse import csgraph

MAX_NODES = 20_000
MAX_EDGES = 500_000
DRAW_NODES = 120


class Graph:
    """Lista de nodos y matriz de adyacencia dispersa (con pesos; los enlaces repetidos se suman)."""

    def __init__(self, nodes: list, adj: sparse.csr_matrix, directed: bool, self_loops: int, dropped: int):
        self.nodes, self.adj, self.directed, self.self_loops, self.dropped = nodes, adj.tocsr(), directed, self_loops, dropped
        self.index = {n: i for i, n in enumerate(nodes)}

    @property
    def n(self) -> int:
        return len(self.nodes)

    def symmetric(self) -> sparse.csr_matrix:
        return self.adj if not self.directed else (self.adj + self.adj.T).tocsr()

    def edges(self) -> int:
        a = self.adj
        return int(a.nnz) if self.directed else int(a.nnz // 2)


def build_graph(df: pd.DataFrame, source: str, target: str, weight: str | None = None, directed: bool = False) -> Graph:
    for c in (source, target) + ((weight,) if weight else ()):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    if source == target:
        raise ValueError("The source and target columns must be different.")
    if weight and not pd.api.types.is_numeric_dtype(df[weight]):
        raise ValueError(f"The weight column '{weight}' must be numeric.")
    edges = df[[source, target] + ([weight] if weight else [])].copy()
    before = len(edges)
    edges = edges.dropna(subset=[source, target])
    dropped = before - len(edges)
    if weight:
        edges[weight] = edges[weight].fillna(0)
        if (edges[weight] < 0).any():
            raise ValueError("Weights cannot be negative.")
        edges = edges[edges[weight] > 0]
    if edges.empty:
        raise ValueError("There are no links to analyse.")
    if len(edges) > MAX_EDGES:
        raise ValueError(f"At most {MAX_EDGES:,} links; filter or aggregate the data first.")
    loops = int((edges[source].astype(str) == edges[target].astype(str)).sum())
    edges = edges[edges[source].astype(str) != edges[target].astype(str)]
    if edges.empty:
        raise ValueError("Every link goes from a node to itself.")
    labels = pd.unique(pd.concat([edges[source], edges[target]], ignore_index=True))
    if len(labels) > MAX_NODES:
        raise ValueError(f"At most {MAX_NODES:,} nodes (found {len(labels):,}); filter the data first.")
    code = {label: i for i, label in enumerate(labels)}
    rows = edges[source].map(code).to_numpy()
    cols = edges[target].map(code).to_numpy()
    w = edges[weight].to_numpy(float) if weight else np.ones(len(edges))
    n = len(labels)
    adj = sparse.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    if not directed:
        adj = (adj + adj.T).tocsr()
        # un enlace A-B y otro B-A son el mismo enlace: quedan sumados, que es lo que quiere decir el peso
    return Graph(list(labels), adj, directed, loops, dropped)


# ── Algoritmos ───────────────────────────────────────────────────────────────
def pagerank(g: Graph, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    n = g.n
    a = g.adj
    out = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros(n), where=out > 0)
    transition = sparse.diags(inv) @ a
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dangling = x[out == 0].sum()
        new = damping * (transition.T @ x + dangling / n) + (1 - damping) / n
        if np.abs(new - x).sum() < tol:
            x = new
            break
        x = new
    return x / x.sum()


def _neighbours(adj: sparse.csr_matrix) -> list:
    return [adj.indices[adj.indptr[i]: adj.indptr[i + 1]] for i in range(adj.shape[0])]


def betweenness(g: Graph, max_pivots: int | None = None, seed: int = 0) -> tuple:
    """Intermediación de Brandes sobre saltos (sin pesos). Con muchos nodos usa un subconjunto de orígenes y reescala; devuelve (valores, exacta)."""
    n = g.n
    adj = g.adj
    neigh = _neighbours(adj)
    budget = max(20, int(3_000_000 / max(adj.nnz, 1)))
    pivots = n if max_pivots is None and n <= budget else min(n, max_pivots or budget)
    exact = pivots == n
    rng = np.random.default_rng(seed)
    sources = np.arange(n) if exact else rng.choice(n, pivots, replace=False)
    score = np.zeros(n)
    for s in sources:
        stack, preds = [], [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in neigh[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                score[w] += delta[w]
    score *= n / len(sources)
    if not g.directed:
        score /= 2
    return score, exact


def clustering(g: Graph) -> np.ndarray:
    """Coeficiente de agrupamiento: qué fracción de los vecinos de un nodo están enlazados entre sí."""
    a = g.symmetric().copy()
    a.data = np.ones_like(a.data)
    deg = np.asarray(a.sum(axis=1)).ravel()
    if a.nnz > 400_000:
        return np.full(g.n, np.nan)
    triangles = np.asarray(a.multiply(a @ a).sum(axis=1)).ravel() / 2
    possible = deg * (deg - 1) / 2
    return np.divide(triangles, possible, out=np.zeros(g.n), where=possible > 0)


def louvain(adj: sparse.csr_matrix, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Comunidades por optimización de la modularidad (Blondel et al., 2008)."""
    n = adj.shape[0]
    neigh = [dict(zip(adj.indices[adj.indptr[i]: adj.indptr[i + 1]], adj.data[adj.indptr[i]: adj.indptr[i + 1]])) for i in range(n)]
    loops = np.zeros(n)
    m2 = float(adj.sum())
    if m2 == 0:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    membership = np.arange(n)
    while True:
        size = len(neigh)
        deg = np.array([sum(d.values()) + loops[i] for i, d in enumerate(neigh)])
        com = list(range(size))
        total = deg.copy()
        improved = False
        while True:
            moved = False
            for i in rng.permutation(size):
                ci = com[i]
                links = {}
                for j, w in neigh[i].items():
                    if j != i:
                        links[com[j]] = links.get(com[j], 0.0) + w
                total[ci] -= deg[i]
                best, best_gain = ci, links.get(ci, 0.0) - resolution * total[ci] * deg[i] / m2
                for c, w in links.items():
                    gain = w - resolution * total[c] * deg[i] / m2
                    if gain > best_gain + 1e-12:
                        best, best_gain = c, gain
                total[best] += deg[i]
                if best != ci:
                    com[i] = best
                    moved = improved = True
            if not moved:
                break
        if not improved:
            break
        labels = {c: k for k, c in enumerate(dict.fromkeys(com))}
        mapping = np.array([labels[c] for c in com])
        membership = mapping[membership]
        k = len(labels)
        new_neigh, new_loops = [dict() for _ in range(k)], np.zeros(k)
        for i, d in enumerate(neigh):
            ci = mapping[i]
            new_loops[ci] += loops[i]
            for j, w in d.items():
                cj = mapping[j]
                if ci == cj:
                    new_loops[ci] += w
                else:
                    new_neigh[ci][cj] = new_neigh[ci].get(cj, 0.0) + w
        neigh, loops = new_neigh, new_loops
        if k == 1:
            break
    return membership


def modularity(adj: sparse.csr_matrix, labels: np.ndarray, resolution: float = 1.0) -> float:
    m2 = float(adj.sum())
    if m2 == 0:
        return 0.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    coo = adj.tocoo()
    same = labels[coo.row] == labels[coo.col]
    inside = float(coo.data[same].sum())
    totals = np.bincount(labels, weights=deg)
    return inside / m2 - resolution * float(np.sum((totals / m2) ** 2))


def _bfs(neigh, source: int, n: int) -> np.ndarray:
    dist = np.full(n, -1)
    dist[source] = 0
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for w in neigh[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                queue.append(w)
    return dist


# ── Análisis completo ────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame, source: str, target: str, weight: str | None = None, directed: bool = False, top: int = 15, seed: int = 0) -> dict:
    """Mide la red: nodos importantes, comunidades, conectividad y distancias."""
    g = build_graph(df, source, target, weight, directed)
    n = g.n
    sym = g.symmetric()
    strength_out = np.asarray(g.adj.sum(axis=1)).ravel()
    strength_in = np.asarray(g.adj.sum(axis=0)).ravel()
    binary = g.adj.copy()
    binary.data = np.ones_like(binary.data)
    degree_out = np.asarray(binary.sum(axis=1)).ravel().astype(int)
    degree_in = np.asarray(binary.sum(axis=0)).ravel().astype(int)
    n_comp, comp = csgraph.connected_components(sym, directed=False)
    sizes = np.bincount(comp)
    giant = int(sizes.argmax())
    labels = louvain(sym, seed=seed)
    # renumera las comunidades de mayor a menor
    order = pd.Series(labels).value_counts().index
    labels = pd.Series(labels).map({c: i + 1 for i, c in enumerate(order)}).to_numpy()
    q = modularity(sym, labels)
    pr = pagerank(g)
    btw, exact = betweenness(g, seed=seed)
    clust = clustering(g)
    table = pd.DataFrame({"node": g.nodes, "connections": degree_out if not directed else degree_in + degree_out, "strength": strength_out + (strength_in if directed else 0),
                          "pagerank": pr, "betweenness": btw, "clustering": clust, "community": labels, "component": comp + 1})
    if directed:
        table["in"], table["out"] = degree_in, degree_out
    table = table.sort_values("pagerank", ascending=False, ignore_index=True)
    # distancias dentro de la componente gigante
    members = np.where(comp == giant)[0]
    neigh = _neighbours(sym)
    rng = np.random.default_rng(seed)
    probes = members if len(members) <= 60 else rng.choice(members, 60, replace=False)
    distances = np.concatenate([_bfs(neigh, int(s), n)[members] for s in probes])
    distances = distances[distances > 0]
    far = int(probes[0])
    for _ in range(2):                                             # barrido doble: una cota inferior buena del diámetro
        d = _bfs(neigh, far, n)
        far = int(np.argmax(np.where(comp == giant, d, -1)))
    diameter = int(_bfs(neigh, far, n).max())
    edges = g.edges()
    density = edges / (n * (n - 1) / (1 if directed else 2)) if n > 1 else 0.0
    hubs = int(np.ceil(0.01 * n)) or 1
    degree_sorted = np.sort((degree_out + (degree_in if directed else 0)))[::-1]
    hub_share = float(degree_sorted[:hubs].sum() / max(degree_sorted.sum(), 1))
    assortativity = _assortativity(sym)
    notes = []
    if sizes[giant] / n < 0.9:
        notes.append(f"The network is split: the largest connected piece holds {sizes[giant] / n:.0%} of the nodes and there are {n_comp:,} pieces.")
    if hub_share > 0.3 and n >= 50:
        notes.append(f"The top 1% of nodes ({hubs}) hold {hub_share:.0%} of all connections: the network depends heavily on a few hubs.")
    bridges = table[(table["betweenness"] > table["betweenness"].quantile(0.95)) & (table["connections"] <= table["connections"].median())]
    if len(bridges):
        notes.append(f"{len(bridges)} node(s) connect parts of the network despite having few links of their own (high betweenness, low degree): {', '.join(map(str, bridges['node'].head(3)))}.")
    if not exact:
        notes.append("Betweenness was estimated from a sample of starting nodes because the network is large.")
    if g.self_loops:
        notes.append(f"{g.self_loops:,} link(s) from a node to itself were ignored.")
    if g.dropped:
        notes.append(f"{g.dropped:,} row(s) with a missing source or target were ignored.")
    if q < 0.3:
        notes.append(f"The communities are weak (modularity {q:.2f}): the groups found are not clearly separated.")
    communities = table.groupby("community").agg(nodes=("node", "size"), pagerank=("pagerank", "sum"),
                                                  leader=("node", "first")).reset_index().sort_values("community")
    return {"table": table, "communities": communities, "nodes": n, "edges": edges, "density": float(density), "components": int(n_comp), "giant_share": float(sizes[giant] / n),
            "avg_distance": float(distances.mean()) if len(distances) else float("nan"), "diameter": diameter, "modularity": float(q), "assortativity": assortativity,
            "hub_share": hub_share, "betweenness_exact": exact, "directed": directed, "notes": notes, "figure": network_figure(g, table, sym), "_graph": g,
            "summary": (f"{n:,} nodes and {edges:,} links; {sizes[giant] / n:.0%} of the nodes are in one connected piece, {len(order)} communities (modularity {q:.2f}), "
                        f"typical distance {float(distances.mean()) if len(distances) else float('nan'):.1f} steps and longest {diameter}. "
                        f"Most important by PageRank: {', '.join(map(str, table['node'].head(3)))}.")}


def _assortativity(sym: sparse.csr_matrix) -> float:
    """¿Se enlazan los nodos muy conectados con otros muy conectados (positivo) o con los poco conectados (negativo)?"""
    coo = sym.tocoo()
    deg = np.diff(sym.indptr).astype(float)
    x, y = deg[coo.row], deg[coo.col]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _layout(adj: sparse.csr_matrix, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """Disposición por fuerzas (Fruchterman-Reingold): los enlazados se acercan y todos se repelen."""
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    dense = (adj.toarray() > 0).astype(float)
    k = np.sqrt(1.0 / max(n, 1))
    t = 0.1
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=2), 1e-4)
        force = (k * k / dist ** 2 - dense * dist / k)[:, :, None] * delta
        move = force.sum(axis=1)
        length = np.maximum(np.linalg.norm(move, axis=1), 1e-9)[:, None]
        pos += move / length * np.minimum(length, t)
        t = max(t * 0.97, 0.002)
    return pos - pos.mean(axis=0)


def network_figure(g: Graph, table: pd.DataFrame, sym: sparse.csr_matrix) -> go.Figure:
    """Dibuja los nodos más importantes (hasta 120) con los enlaces entre ellos; el color es la comunidad y el tamaño, el PageRank."""
    shown = table.head(DRAW_NODES)
    ids = [g.index[node] for node in shown["node"]]
    sub = sym[ids][:, ids]
    pos = _layout(sub)
    coo = sparse.triu(sub, k=1).tocoo()
    xs, ys = [], []
    for i, j in zip(coo.row, coo.col):
        xs += [pos[i, 0], pos[j, 0], None]
        ys += [pos[i, 1], pos[j, 1], None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.6, color="rgba(150,150,150,0.5)"), hoverinfo="skip", showlegend=False))
    size = 8 + 28 * shown["pagerank"].to_numpy() / max(shown["pagerank"].max(), 1e-12)
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers", text=shown["node"].astype(str), hovertemplate="%{text}<extra></extra>",
                             marker=dict(size=size, color=shown["community"], colorscale="Turbo", line=dict(width=0.5, color="white")), showlegend=False))
    fig.update_layout(title=f"The {len(shown)} most important nodes (colour = community, size = PageRank)", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ── Caminos, enlaces que faltan y robustez ───────────────────────────────────
def _need_node(g: Graph, node):
    if node in g.index:
        return g.index[node]
    match = [k for k in g.nodes if str(k) == str(node)]
    if match:
        return g.index[match[0]]
    raise ValueError(f"'{node}' is not a node of this network.")


def shortest_path(df: pd.DataFrame, source: str, target: str, start, end, weight: str | None = None, directed: bool = False) -> dict:
    """El camino con menos saltos entre dos nodos."""
    g = build_graph(df, source, target, weight, directed)
    a, b = _need_node(g, start), _need_node(g, end)
    neigh = _neighbours(g.adj)
    parent = {a: None}
    queue = deque([a])
    while queue and b not in parent:
        v = queue.popleft()
        for w in neigh[v]:
            if w not in parent:
                parent[w] = v
                queue.append(w)
    if b not in parent:
        return {"path": [], "steps": None, "connected": False, "summary": f"There is no path from {start} to {end}: they are in different pieces of the network."}
    path = []
    node = b
    while node is not None:
        path.append(g.nodes[node])
        node = parent[node]
    path.reverse()
    return {"path": path, "steps": len(path) - 1, "connected": True, "summary": f"{start} reaches {end} in {len(path) - 1} step(s): " + " → ".join(map(str, path)) + "."}


def suggest_links(df: pd.DataFrame, source: str, target: str, weight: str | None = None, top: int = 10) -> dict:
    """Pares sin enlace que comparten muchos vecinos poco conectados (Adamic-Adar): candidatos a conocerse, a comprar juntos o a colaborar."""
    g = build_graph(df, source, target, weight, False)
    a = g.adj.copy()
    a.data = np.ones_like(a.data)
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    scores = (a @ sparse.diags(inv) @ a).tocoo()
    common = (a @ a).tocoo()
    shared = {(i, j): c for i, j, c in zip(common.row, common.col, common.data)}
    rows = []
    existing = a.tolil()
    for i, j, s in zip(scores.row, scores.col, scores.data):
        if i < j and existing[i, j] == 0 and s > 0:
            rows.append({"a": g.nodes[i], "b": g.nodes[j], "score": float(s), "shared neighbours": int(shared.get((i, j), 0))})
    if not rows:
        raise ValueError("No pair of unconnected nodes shares a neighbour.")
    table = pd.DataFrame(rows).sort_values(["score", "shared neighbours"], ascending=False, ignore_index=True).head(top)
    return {"table": table, "notes": ["A high score means the two are close in the network, not that a link is certain: use it to decide what to look at."],
            "summary": f"{table.iloc[0]['a']} and {table.iloc[0]['b']} are not linked but share {table.iloc[0]['shared neighbours']} neighbour(s): the most likely missing link."}


def robustness(df: pd.DataFrame, source: str, target: str, weight: str | None = None, steps: int = 10, seed: int = 0) -> dict:
    """Tamaño de la mayor pieza conectada al quitar poco a poco los nodos más conectados y, para comparar, nodos al azar."""
    g = build_graph(df, source, target, weight, False)
    n = g.n
    sym = g.symmetric()
    deg = np.diff(sym.indptr)
    steps = max(2, min(steps, 30))
    fractions = np.linspace(0, 0.5, steps + 1)
    rng = np.random.default_rng(seed)

    def giant_share(removed: np.ndarray) -> float:
        keep = np.ones(n, bool)
        keep[removed] = False
        if keep.sum() == 0:
            return 0.0
        _, labels = csgraph.connected_components(sym[keep][:, keep], directed=False)
        return float(np.bincount(labels).max() / n)

    by_degree = np.argsort(-deg)
    targeted, random_ = [], []
    for f in fractions:
        k = int(round(f * n))
        targeted.append(giant_share(by_degree[:k]))
        random_.append(float(np.mean([giant_share(rng.choice(n, k, replace=False)) for _ in range(3)])))
    table = pd.DataFrame({"nodes removed": fractions, "largest piece (most connected first)": targeted, "largest piece (random)": random_})
    half = next((f for f, v in zip(fractions, targeted) if v < 0.5 * targeted[0]), None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fractions, y=targeted, name="Most connected first"))
    fig.add_trace(go.Scatter(x=fractions, y=random_, name="At random", line=dict(dash="dash")))
    fig.update_layout(title="Share of nodes still in the largest piece as nodes are removed", xaxis_title="Share of nodes removed", yaxis_title="Largest piece", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%")
    fragile = targeted[steps // 5 or 1] < 0.7 * random_[steps // 5 or 1]
    return {"table": table, "figure": fig, "fragile": bool(fragile), "notes": (["Removing the best-connected nodes breaks the network much faster than removing random ones: it depends on its hubs."] if fragile else []),
            "summary": (f"Removing the best-connected nodes halves the largest piece after {half:.0%} of nodes; at random it takes far more." if half is not None else
                        "Even after removing half of the best-connected nodes, the largest piece keeps at least half of its size.")}
