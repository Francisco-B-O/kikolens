# Development

The short version is in [CONTRIBUTING.md](https://github.com/Francisco-B-O/kikolens/blob/develop/CONTRIBUTING.md). This page explains how the
checks are organised.

## Continuous integration

Every pull request into `develop` or `main` runs:

| Job | What it checks |
| :--- | :--- |
| **Lint** | `ruff check` for unused imports and variables, undefined names and syntax errors. |
| **Tests** | The whole suite on Linux with Python 3.12 and the latest pandas, with coverage. |
| **Compatibility** | The suite on Linux with pandas 2, on Windows with pandas 3, and, as warnings that do not block, on macOS and with Python 3.13. |
| **Build package** | Builds the wheel, installs it and runs the command line, including `kikolens doctor`. |
| **Documentation** | `mkdocs build --strict`: a broken link or a missing page fails the build. |
| **CodeQL** | Security analysis of the Python code, also weekly. |
| **SonarCloud** | Code quality, from the coverage of the Tests job. |

Dependabot proposes monthly updates of the dependencies and the actions into `develop`.

## Releases

1. Move the *Unreleased* section of `CHANGELOG.md` under the new version and set `__version__` in `kikolens/__init__.py` and `version` in `pyproject.toml`.
2. Merge `develop` into `main` with a pull request.
3. Push the tag, for example `git tag -a v1.0.0 -m "1.0.0" && git push origin v1.0.0`.

The **Release** workflow checks that the tag matches the package version, builds the wheel and the source archive, and creates the GitHub
release with the changelog notes of that version. Publishing to PyPI is deliberately manual, because it cannot be undone.

## Tests

* `tests/test_dashboard.py` and the other `AppTest` based tests drive the real Streamlit script headlessly.
* `tests/conftest.py` points `KIKOLENS_HOME` at a temporary folder so no test writes to a real user folder.
* Tests of the analyses plant a known effect (a coefficient, a changed region, a drifted period) and check that it is found, and plant none and check that it is not.

## Docs and screenshots

```bash
pip install mkdocs-material
mkdocs serve                       # live preview at http://127.0.0.1:8000
pip install -e ".[docs]" && python -m playwright install chromium
python docs/make_screenshots.py    # regenerates the example data and every image (about 25 minutes)
```
