import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import network as nw


def edges_of(rows, a="a", b="b"):
    return pd.DataFrame(rows, columns=[a, b])


def planted(n_per=40, groups=3, p_in=0.25, p_out=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * groups
    truth = np.repeat(range(groups), n_per)
    rows = [(f"n{i}", f"n{j}") for i in range(n) for j in range(i + 1, n) if rng.random() < (p_in if truth[i] == truth[j] else p_out)]
    return edges_of(rows), pd.Series(truth, index=[f"n{i}" for i in range(n)])


# ── Medidas con respuesta conocida ───────────────────────────────────────────
def test_a_path_has_the_textbook_betweenness_and_a_star_has_one_hub():
    path = nw.analyze(edges_of([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]), "a", "b")
    table = path["table"].set_index("node")
    assert table["betweenness"].to_dict() == {"A": 0.0, "B": 3.0, "C": 4.0, "D": 3.0, "E": 0.0} and path["diameter"] == 4 and path["components"] == 1
    assert table["pagerank"].sum() == pytest.approx(1.0)
    star = nw.analyze(edges_of([("hub", f"leaf{i}") for i in range(8)]), "a", "b")
    assert star["table"].iloc[0]["node"] == "hub" and star["table"].iloc[0]["connections"] == 8 and star["assortativity"] < 0
    assert star["table"].set_index("node").loc["hub", "betweenness"] == pytest.approx(28.0)           # 8 elegir 2 pares pasan por el centro


def test_clustering_counts_triangles():
    triangle_plus_tail = edges_of([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])
    table = nw.analyze(triangle_plus_tail, "a", "b")["table"].set_index("node")
    assert table.loc["A", "clustering"] == 1.0 and table.loc["C", "clustering"] == pytest.approx(1 / 3) and table.loc["D", "clustering"] == 0.0


def test_communities_are_recovered_from_a_planted_partition():
    edges, truth = planted()
    res = nw.analyze(edges, "a", "b")
    labels = res["table"].set_index("node")["community"]
    purity = pd.crosstab(truth.reindex(labels.index), labels).max(axis=1).sum() / len(labels)
    assert purity > 0.95 and res["modularity"] > 0.5 and len(res["communities"]) == 3 and res["communities"]["nodes"].sum() == 120
    assert res["table"]["community"].min() == 1 and res["communities"].iloc[0]["nodes"] >= res["communities"].iloc[-1]["nodes"]
    assert not res["notes"] or all("weak" not in n for n in res["notes"])


def test_the_measures_agree_with_networkx_when_it_is_installed():
    nx = pytest.importorskip("networkx")
    edges, _ = planted(30, 2, 0.3, 0.03, seed=3)
    res = nw.analyze(edges, "a", "b")
    G = nx.from_pandas_edgelist(edges, "a", "b")
    table = res["table"].set_index("node")
    expected = {"pagerank": nx.pagerank(G, tol=1e-12), "betweenness": nx.betweenness_centrality(G, normalized=False), "clustering": nx.clustering(G)}
    for column, values in expected.items():
        assert max(abs(values[k] - table.loc[k, column]) for k in values) < 1e-8, column
    assert res["betweenness_exact"] and res["avg_distance"] == pytest.approx(nx.average_shortest_path_length(G), abs=0.05)
    assert res["assortativity"] == pytest.approx(nx.degree_assortativity_coefficient(G), abs=1e-6)
    assert res["modularity"] == pytest.approx(nx.community.modularity(G, [set(table.index[table["community"] == c]) for c in table["community"].unique()]), abs=1e-9)
    directed = edges.sample(frac=0.7, random_state=1)
    d = nw.analyze(directed, "a", "b", directed=True)
    D = nx.from_pandas_edgelist(directed, "a", "b", create_using=nx.DiGraph)
    expected = nx.pagerank(D, tol=1e-12)
    assert max(abs(expected[k] - d["table"].set_index("node").loc[k, "pagerank"]) for k in expected) < 1e-8


def test_weights_and_repeated_links_add_up_and_directed_networks_report_in_and_out():
    rows = pd.DataFrame({"a": ["x", "x", "y", "z"], "b": ["y", "y", "z", "x"], "w": [2.0, 3.0, 1.0, 4.0]})
    undirected = nw.build_graph(rows, "a", "b", "w")
    assert undirected.adj[undirected.index["x"], undirected.index["y"]] == 5.0 and undirected.adj[undirected.index["x"], undirected.index["z"]] == 4.0 and undirected.edges() == 3
    directed = nw.analyze(rows, "a", "b", "w", directed=True)
    assert {"in", "out"} <= set(directed["table"].columns) and directed["table"].set_index("node").loc["x", "out"] == 1 and directed["directed"]


def test_split_networks_and_hub_dependence_are_called_out():
    rows = [(f"a{i}", f"a{(i + 1) % 10}") for i in range(10)] + [(f"b{i}", f"b{(i + 1) % 10}") for i in range(10)] + [("lone1", "lone2")]
    res = nw.analyze(edges_of(rows), "a", "b")
    assert res["components"] == 3 and any("split" in n for n in res["notes"]) and res["giant_share"] == pytest.approx(10 / 22)
    hubs = edges_of([("hub", f"s{i}") for i in range(80)] + [(f"s{i}", f"s{i + 1}") for i in range(0, 78, 3)])
    assert any("depends heavily" in n for n in nw.analyze(hubs, "a", "b")["notes"])


def test_input_problems_are_explained():
    ok = edges_of([("A", "B"), ("B", "C")])
    for kwargs, fragment in (({"target": "a"}, "must be different"), ({"source": "nope"}, "not found"), ({"weight": "a"}, "must be numeric")):
        args = {"source": "a", "target": "b", **kwargs}
        with pytest.raises(ValueError, match=fragment):
            nw.build_graph(ok, args["source"], args["target"], kwargs.get("weight"))
    with pytest.raises(ValueError, match="no links"):
        nw.build_graph(edges_of([(None, "A")]), "a", "b")
    with pytest.raises(ValueError, match="itself"):
        nw.build_graph(edges_of([("A", "A")]), "a", "b")
    with pytest.raises(ValueError, match="negative"):
        nw.build_graph(pd.DataFrame({"a": ["x"], "b": ["y"], "w": [-1.0]}), "a", "b", "w")
    noisy = nw.analyze(edges_of([("A", "B"), ("B", "C"), ("C", "C"), (None, "D")]), "a", "b")
    assert any("itself" in n for n in noisy["notes"]) and any("missing" in n for n in noisy["notes"])


# ── Caminos, enlaces que faltan y robustez ───────────────────────────────────
def test_shortest_path_and_unreachable_nodes():
    rows = edges_of([("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"), ("X", "Y")])
    res = nw.shortest_path(rows, "a", "b", "A", "C")
    assert res["steps"] == 2 and res["path"][0] == "A" and res["path"][-1] == "C" and res["connected"]
    assert nw.shortest_path(rows, "a", "b", "A", "D")["steps"] == 1
    far = nw.shortest_path(rows, "a", "b", "A", "X")
    assert far["connected"] is False and "no path" in far["summary"]
    with pytest.raises(ValueError, match="not a node"):
        nw.shortest_path(rows, "a", "b", "A", "ghost")
    ids = edges_of([(1, 2), (2, 3)])
    assert nw.shortest_path(ids, "a", "b", "1", "3")["steps"] == 2                 # los nodos numéricos se pueden pedir como texto


def test_link_suggestions_find_the_missing_edge_of_a_clique():
    members = ["a", "b", "c", "d", "e"]
    rows = [(x, y) for i, x in enumerate(members) for y in members[i + 1:] if (x, y) != ("a", "e")] + [("e", "far1"), ("far1", "far2")]
    res = nw.suggest_links(edges_of(rows), "a", "b", top=3)
    first = res["table"].iloc[0]
    assert {first["a"], first["b"]} == {"a", "e"} and first["shared neighbours"] == 3 and res["notes"]
    with pytest.raises(ValueError, match="No pair"):
        nw.suggest_links(edges_of([("A", "B"), ("C", "D")]), "a", "b")


def test_robustness_shows_a_hub_network_breaking_faster_than_random_removal():
    star = edges_of([("hub", f"s{i}") for i in range(60)])
    res = nw.robustness(star, "a", "b", steps=10)
    table = res["table"]
    assert table["largest piece (most connected first)"].iloc[0] == 1.0 and table["largest piece (most connected first)"].iloc[1] < 0.2 < table["largest piece (random)"].iloc[1]
    assert res["fragile"] and res["notes"] and "halves" in res["summary"] and len(res["figure"].data) == 2
    ring = edges_of([(f"r{i}", f"r{(i + 1) % 60}") for i in range(60)] + [(f"r{i}", f"r{(i + 7) % 60}") for i in range(60)])
    assert not nw.robustness(ring, "a", "b")["fragile"]


def test_large_networks_use_sampled_betweenness_and_stay_quick():
    rng = np.random.default_rng(5)
    n = 3000
    rows = edges_of([(f"u{i}", f"u{j}") for i, j in zip(rng.integers(0, n, 9000), rng.integers(0, n, 9000))])
    res = nw.analyze(rows, "a", "b")
    assert not res["betweenness_exact"] and any("sample" in note for note in res["notes"]) and res["nodes"] > 2000 and len(res["figure"].data) == 2


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    edges, _ = planted(20, 3)
    assert kl.network(edges, "a", "b")["kind"] == "network" and kl.network(edges, "a", "b")["modularity"] > 0.3
    assert kl.network_path(edges, "a", "b", "n0", "n59")["kind"] == "network path"
    assert kl.suggest_links(edges, "a", "b", top=3)["table"].shape[0] == 3 and kl.network_robustness(edges, "a", "b")["kind"] == "network robustness"
    csv = tmp_path / "e.csv"
    edges.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["network", str(csv), "a", "b"], "communities"), (["network", str(csv), "a", "b", "--output", str(tmp_path / "nodes.csv")], "Nodes written"),
                       (["network-path", str(csv), "a", "b", "n0", "n59"], "n0"), (["suggest-links", str(csv), "a", "b"], "missing link"),
                       (["network-robustness", str(csv), "a", "b"], "largest piece")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in " ".join(out.output.split()), (args, out.output)
    assert (tmp_path / "nodes.csv").exists() and runner.invoke(app, ["network", str(csv), "a", "a"]).exit_code == 1
