import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze.quality import auto_clean_steps
from kikolens.cli.main import app
from kikolens.pipeline.project import load_project, save_project
from kikolens.pipeline.recipe import Recipe, apply_operation

runner = CliRunner()


@pytest.fixture
def messy():
    return pd.DataFrame({
        "name ": [" a", "b ", "b ", "c", "d"], "price": ["1,000", "2,500", "2,500", "30", "400"],
        "const": [1, 1, 1, 1, 1], "qty": [1, 2, 2, 3, 4],
    })


def make_recipe(df):
    recipe = Recipe()
    _, steps = auto_clean_steps(df)
    for op, params, message in steps:
        recipe.add(op, params, message)
    recipe.add("add_column", {"name": "total", "expression": "price * qty"}, "total")
    return recipe


def test_recipe_replays_same_result_on_new_data(messy):
    recipe = make_recipe(messy)
    first, log = recipe.apply(messy)
    assert len(log) == len(recipe) and "total" in first.columns
    assert "const" not in first.columns and len(first) == 4
    again, _ = recipe.apply(messy.copy())
    pd.testing.assert_frame_equal(first, again)


def test_recipe_json_roundtrip(messy):
    recipe = make_recipe(messy)
    restored = Recipe.from_json(recipe.to_json())
    assert [s.op for s in restored.steps] == [s.op for s in recipe.steps]
    pd.testing.assert_frame_equal(restored.apply(messy)[0], recipe.apply(messy)[0])


def test_recipe_params_with_numpy_values_serialize():
    recipe = Recipe()
    recipe.add("sample_rows", {"n": np.int64(3), "seed": np.int64(1)})
    data = json.loads(recipe.to_json())
    assert data["steps"][0]["params"] == {"n": 3, "seed": 1}


def test_recipe_rejects_bad_input():
    with pytest.raises(ValueError, match="valid JSON"):
        Recipe.from_json("{nope")
    with pytest.raises(ValueError, match="not a KikoLens recipe"):
        Recipe.from_json("[]")
    with pytest.raises(ValueError, match="Unknown or malformed"):
        Recipe.from_json(json.dumps({"steps": [{"op": "os.system", "params": {}}]}))
    with pytest.raises(ValueError, match="newer"):
        Recipe.from_json(json.dumps({"kikolens_recipe": 99, "steps": []}))


def test_recipe_names_the_failing_step(messy):
    recipe = Recipe()
    recipe.add("fill_missing", {"column": "ghost", "strategy": "mean"})
    with pytest.raises(ValueError, match="Step 1"):
        recipe.apply(messy)
    with pytest.raises(ValueError, match="Invalid parameters"):
        apply_operation(messy, "drop_columns", wrong=1)


def test_recipe_with_join_requires_the_dataset(messy):
    recipe = Recipe()
    recipe.add("merge_datasets", {"on": ["qty"], "how": "left", "other": "extra"})
    assert recipe.required_datasets == ["extra"]
    with pytest.raises(ValueError, match="extra"):
        recipe.apply(messy)
    out, _ = recipe.apply(messy, {"extra": pd.DataFrame({"qty": [1, 2], "label": ["x", "y"]})})
    assert "label" in out.columns


def test_python_export_is_valid_and_runs(messy, tmp_path):
    recipe = make_recipe(messy)
    csv = tmp_path / "in.csv"
    messy.to_csv(csv, index=False)
    code = recipe.to_python(str(csv))
    compile(code, "pipeline.py", "exec")
    namespace = {"__name__": "generated"}
    exec(code, namespace)
    assert "total" in namespace["run"](str(csv)).columns


def test_project_roundtrip(messy):
    recipe = make_recipe(messy)
    extra = pd.DataFrame({"qty": [1, 2], "label": ["x", "y"]})
    blob = save_project(messy, recipe, name="demo", datasets={"extra": extra}, notes="hello", models={"lab": b"123"})
    project = load_project(blob, load_models=True)
    assert project["notes"] == "hello" and project["manifest"]["name"] == "demo"
    assert project["models"] == {"lab": b"123"}
    pd.testing.assert_frame_equal(project["datasets"]["extra"], extra)
    assert len(project["recipe"]) == len(recipe)
    assert load_project(blob)["models"] == {}  # los modelos no se leen salvo petición explícita


def test_project_handles_mixed_object_columns():
    df = pd.DataFrame({"mixed": [1, "a", None, 2.5], "n": [1, 2, 3, 4]})
    project = load_project(save_project(df, Recipe()))
    assert project["original"]["mixed"].tolist()[1] == "a"


def test_project_rejects_foreign_files():
    with pytest.raises(ValueError, match="not a KikoLens project"):
        load_project(b"plain text")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("hello.txt", "x")
    with pytest.raises(ValueError, match="not a KikoLens project"):
        load_project(buffer.getvalue())


def test_cli_clean_saves_recipe_and_apply_reuses_it(tmp_path, messy):
    source, other = tmp_path / "a.csv", tmp_path / "b.csv"
    messy.to_csv(source, index=False)
    messy.assign(qty=[9, 8, 8, 7, 6]).to_csv(other, index=False)
    recipe_path = tmp_path / "r.json"
    result = runner.invoke(app, ["clean", str(source), "--output", str(tmp_path / "out.csv"),
                                 "--save-recipe", str(recipe_path)])
    assert result.exit_code == 0 and recipe_path.exists()
    result = runner.invoke(app, ["apply", str(other), str(recipe_path), "--output", str(tmp_path / "out2.parquet")])
    assert result.exit_code == 0, result.output
    out = pd.read_parquet(tmp_path / "out2.parquet")
    assert "const" not in out.columns and pd.api.types.is_numeric_dtype(out["price"])


def test_cli_apply_errors(tmp_path, messy):
    source = tmp_path / "a.csv"
    messy.to_csv(source, index=False)
    bad = tmp_path / "bad.json"
    bad.write_text("nope")
    assert runner.invoke(app, ["apply", str(source), str(bad), "--output", str(tmp_path / "o.csv")]).exit_code == 1
    recipe = tmp_path / "r.json"
    recipe.write_text(make_recipe(messy).to_json())
    assert runner.invoke(app, ["apply", str(source), str(recipe), "--output", str(tmp_path / "o.txt")]).exit_code == 1
    assert runner.invoke(app, ["apply", str(source), str(recipe), "--output", str(tmp_path / "o.csv"),
                               "-d", "broken"]).exit_code == 1


def test_cli_project_info_and_export(tmp_path, messy):
    path = tmp_path / "p.kikolens"
    path.write_bytes(save_project(messy, make_recipe(messy), name="demo", notes="remember me"))
    result = runner.invoke(app, ["project", str(path), "--output", str(tmp_path / "x.csv")])
    assert result.exit_code == 0 and "remember me" in result.output
    assert "total" in pd.read_csv(tmp_path / "x.csv").columns
    assert runner.invoke(app, ["project", str(tmp_path / "missing.kikolens")]).exit_code == 1
