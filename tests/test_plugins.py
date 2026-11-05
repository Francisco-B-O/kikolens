import textwrap

import pandas as pd
import pytest

from kikolens import plugins

GOOD = textwrap.dedent('''
    import pandas as pd

    def register(registry):
        @registry.analysis("margin_by", "Average margin by group.", params={
            "group": {"type": "column"}, "price": "column", "cost": "column",
            "top": {"type": "integer", "default": 3, "min": 1, "max": 10},
            "order": {"type": "choice", "choices": ["high", "low"], "default": "high"}})
        def margin_by(df, group, price, cost, top, order):
            table = df.assign(margin=df[price] - df[cost]).groupby(group, as_index=False)["margin"].mean()
            table = table.sort_values("margin", ascending=(order == "low")).head(top)
            return {"summary": f"{table.iloc[0][group]} leads on margin.", "table": table}

        registry.loader(".pipe", lambda path: pd.read_csv(path, sep="|"))
''')


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KIKOLENS_PLUGINS", raising=False)
    (tmp_path / "home" / "plugins").mkdir(parents=True)
    plugins.reset()
    yield tmp_path / "home" / "plugins"
    plugins.reset()


@pytest.fixture
def sales():
    return pd.DataFrame({"shop": list("aabbcc"), "price": [10.0, 12, 9, 9, 20, 22], "cost": [5.0, 6, 8, 8, 10, 11]})


def test_a_plugin_in_the_user_folder_adds_an_analysis_that_runs_with_checked_arguments(folder, sales):
    (folder / "margins.py").write_text(GOOD, encoding="utf-8")
    found = plugins.analyses()
    assert set(found) == {"margin_by"} and found["margin_by"].plugin == "margins" and plugins.errors() == []
    assert plugins.loaded() == [{"plugin": "margins", "source": str(folder / "margins.py"), "analyses": 1, "loaders": 1}]
    out = plugins.run("margin_by", sales, group="shop", price="price", cost="cost", top=2)
    assert out["summary"] == "c leads on margin." and list(out["table"]["shop"]) == ["c", "a"]
    assert plugins.run("margin_by", sales, group="SHOP", price="price", cost="cost", order="low")["table"].iloc[0]["shop"] == "b"      # sin distinguir mayúsculas
    assert sales.shape == (6, 3)                                                                                                          # los datos originales no se tocan


@pytest.mark.parametrize("args,fragment", [
    ({"group": "shop", "price": "price"}, "needs 'cost'"),
    ({"group": "nope", "price": "price", "cost": "cost"}, "does not exist"),
    ({"group": "shop", "price": "price", "cost": "cost", "top": 99}, "between"),
    ({"group": "shop", "price": "price", "cost": "cost", "top": 1.5}, "whole number"),
    ({"group": "shop", "price": "price", "cost": "cost", "top": "x"}, "must be a number"),
    ({"group": "shop", "price": "price", "cost": "cost", "order": "middle"}, "one of high, low"),
    ({"group": "shop", "price": "price", "cost": "cost", "evil": 1}, "does not take"),
])
def test_bad_arguments_are_rejected_before_the_plugin_runs(folder, sales, args, fragment):
    (folder / "margins.py").write_text(GOOD, encoding="utf-8")
    with pytest.raises(ValueError, match=fragment):
        plugins.run("margin_by", sales, **args)
    with pytest.raises(ValueError, match="No plugin analysis called"):
        plugins.run("ghost", sales)


def test_a_broken_plugin_is_reported_and_does_not_stop_the_others(folder):
    (folder / "good.py").write_text(GOOD, encoding="utf-8")
    (folder / "syntax.py").write_text("def register(registry:\n", encoding="utf-8")
    (folder / "norun.py").write_text("x = 1\n", encoding="utf-8")
    (folder / "boom.py").write_text("def register(registry):\n    raise RuntimeError('nope')\n", encoding="utf-8")
    (folder / "clash.py").write_text('def register(registry):\n    registry.analysis("profile", "x")(lambda df: {})\n', encoding="utf-8")
    (folder / "_private.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    assert set(plugins.analyses()) == {"margin_by"}
    errors = "\n".join(plugins.errors())
    assert all(name in errors for name in ("syntax.py", "norun.py", "boom.py", "clash.py")) and "_private" not in errors
    assert "no register" in errors and "RuntimeError: nope" in errors and "built-in" in errors


def test_registration_rules_names_duplicates_and_parameter_specs():
    r = plugins.Registry()
    with pytest.raises(ValueError, match="lowercase"):
        r.analysis("Bad Name")
    with pytest.raises(ValueError, match="type must be"):
        r.analysis("ok_name", params={"a": {"type": "magic"}})
    with pytest.raises(ValueError, match="needs its 'choices'"):
        r.analysis("ok_name", params={"a": {"type": "choice"}})
    r.analysis("one", "first")(lambda df: {})
    with pytest.raises(ValueError, match="already registered"):
        r.analysis("one", "again")(lambda df: {})
    with pytest.raises(ValueError, match="file extension"):
        r.loader("dat", lambda p: None)
    r.loader(".dat", lambda p: None)
    with pytest.raises(ValueError, match="already registered"):
        r.loader([".dat"], lambda p: None)


def test_a_plugin_that_crashes_or_returns_odd_things_is_handled(folder, sales):
    code = textwrap.dedent('''
        import pandas as pd
        def register(registry):
            registry.analysis("crash", "boom")(lambda df: 1 / 0)
            registry.analysis("frame", "a frame")(lambda df: df.head(2))
            registry.analysis("number", "a number")(lambda df: 42)
            registry.analysis("series", "a series")(lambda df: df["price"])
            registry.analysis("bad_value", "raises")(lambda df: (_ for _ in ()).throw(ValueError("be nicer")))
    ''')
    (folder / "odd.py").write_text(code, encoding="utf-8")
    with pytest.raises(ValueError, match="failed: ZeroDivisionError"):
        plugins.run("crash", sales)
    with pytest.raises(ValueError, match="be nicer"):
        plugins.run("bad_value", sales)
    assert len(plugins.run("frame", sales)["table"]) == 2 and plugins.run("number", sales)["value"] == 42
    assert plugins.run("frame", sales)["summary"] == "a frame" and "price" in plugins.run("series", sales)["table"].columns


def test_plugins_can_be_turned_off_and_extra_paths_are_read_from_the_configuration(folder, tmp_path, monkeypatch, sales):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "mine.py").write_text(GOOD, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kikolens.toml").write_text(f'[plugins]\npaths = ["{elsewhere.as_posix()}"]\n', encoding="utf-8")
    assert set(plugins.load_plugins(force=True).analyses) == {"margin_by"}
    monkeypatch.setenv("KIKOLENS_PLUGINS", "0")
    assert plugins.load_plugins(force=True).analyses == {} and not plugins.enabled()


def test_package_entry_points_are_loaded(folder, monkeypatch):
    class Point:
        name, value = "pkg", "pkg.mod:register"
        def load(self):
            return lambda registry: registry.analysis("from_package", "ok")(lambda df: {"summary": "hi"})

    class Broken(Point):
        name = "broken"
        def load(self):
            raise ImportError("missing dependency")

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [Point(), Broken()])
    assert "from_package" in plugins.load_plugins(force=True).analyses
    assert any("broken: ImportError" in e for e in plugins.errors()) and plugins.loaded()[0]["plugin"] == "pkg"


def test_a_plugin_loader_teaches_load_data_a_new_format(folder, tmp_path):
    from kikolens.utils.loaders import load_data
    (folder / "margins.py").write_text(GOOD, encoding="utf-8")
    data = tmp_path / "sales.pipe"
    data.write_text("a|b\n1|2\n3|4\n", encoding="utf-8")
    assert load_data(str(data)).to_dict("list") == {"a": [1, 3], "b": [2, 4]}
    with open(data, "rb") as handle:
        assert load_data(handle).shape == (2, 2)                       # un archivo subido
    with pytest.raises(ValueError, match="Unsupported"):
        load_data(str(tmp_path / "x.unknown"))
    (folder / "wrong.py").write_text('def register(registry):\n    registry.loader(".bad", lambda p: "not a frame")\n', encoding="utf-8")
    plugins.reset()
    bad = tmp_path / "x.bad"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must return a DataFrame"):
        load_data(str(bad))


def test_analyses_become_tools_for_the_analyst_and_appear_in_the_python_api_and_terminal(folder, sales, tmp_path):
    import kikolens as kl
    from kikolens.ai import analyst
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    (folder / "margins.py").write_text(GOOD, encoding="utf-8")
    (folder / "free.py").write_text('def register(registry):\n    registry.analysis("free_text", "needs text", params={"note": {"type": "text"}})(lambda df, note: {})\n',
                                    encoding="utf-8")
    tools = analyst._tools()
    assert "margin_by" in tools and "free_text" not in tools                # el texto libre no se le deja elegir al modelo
    assert tools["margin_by"].params["top"] == "number?" and tools["margin_by"].params["order"] == "text:high|low?"
    clean = analyst.validate_call(sales, "margin_by", {"group": "shop", "price": "price", "cost": "cost", "top": 1})
    assert tools["margin_by"].run(sales, **clean)["summary"] == "c leads on margin."
    assert "margin_by" in analyst.tool_catalog()
    res = kl.run("margin_by", sales, group="shop", price="price", cost="cost")
    assert res["kind"] == "margin_by" and isinstance(res["table"], pd.DataFrame)
    assert {a["name"] for a in kl.list_plugins()["analyses"]} == {"margin_by", "free_text"} and kl.list_plugins()["loaders"] == [".pipe"]
    csv = tmp_path / "s.csv"
    sales.to_csv(csv, index=False)
    runner = CliRunner()
    out = runner.invoke(app, ["plugin", "margin_by", str(csv), "--arg", "group=shop", "--arg", "price=price", "--arg", "cost=cost", "--arg", "top=1"])
    assert out.exit_code == 0 and "c leads on margin." in out.output
    assert runner.invoke(app, ["plugin", "margin_by", str(csv), "--arg", "group=shop"]).exit_code == 1
    assert runner.invoke(app, ["plugin", "margin_by", str(csv), "--arg", "oops"]).exit_code == 1
    listing = runner.invoke(app, ["plugins"])
    assert listing.exit_code == 0 and "margin_by" in listing.output and "free_text" in listing.output


def test_the_plugins_page_shows_a_form_runs_the_analysis_and_reports_problems(folder, sales):
    from streamlit.testing.v1 import AppTest
    from kikolens.dashboard import app as dashboard_app
    from pathlib import Path
    app = str(Path(dashboard_app.__file__))

    def open_page():
        at = AppTest.from_file(app, default_timeout=60)
        at.session_state["df"] = sales
        at.session_state["original_df"] = sales.copy()
        at.session_state["filename"] = "s.csv"
        at.session_state["nav"] = "Plugins"
        return at.run()

    at = open_page()
    assert not at.exception and any("No plugins found" in str(n.value) for n in at.markdown)
    (folder / "margins.py").write_text(GOOD, encoding="utf-8")
    (folder / "boom.py").write_text("raise RuntimeError('broken plugin')\n", encoding="utf-8")
    plugins.reset()
    at = open_page()
    assert not at.exception and any("broken plugin" in str(n.value) for n in at.markdown)
    at.selectbox(key="pg_margin_by_group").set_value("shop")
    at.selectbox(key="pg_margin_by_price").set_value("price")
    at.selectbox(key="pg_margin_by_cost").set_value("cost")
    at.button(key="pg_run").click().run()
    assert not at.exception
    assert at.session_state["pg_res"][1]["summary"] == "c leads on margin."
