# kikolens/__init__.py
__version__ = "1.0.0"

from . import cli
from . import analyze
from . import ai
from . import utils
from . import sql

# Re-exporting core functions for the Intelligence Engine
from .utils.loaders import load_data
from .ai.insights import get_ai_insights
from .analyze.stats import basic_statistics, correlation_matrix
from .analyze.automl import run_automl_analysis
from .analyze.clustering import run_clustering_analysis
from .analyze.discovery import run_discovery_engine
from .sql.runner import run_sql_query
from .cli.main import cli


# API para cuadernos y scripts (`import kikolens as kl; kl.profile(df)`): se carga solo cuando se usa.
def _api():
    import importlib
    return importlib.import_module("kikolens.api")


def __getattr__(name):
    if name.startswith("_") or name in ("api", "ipython"):
        raise AttributeError(f"module 'kikolens' has no attribute {name!r}")
    api = _api()
    if name in api.FUNCTIONS:
        return getattr(api, name)
    raise AttributeError(f"module 'kikolens' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_api().FUNCTIONS))


def load_ipython_extension(ipython):
    """`%load_ext kikolens` en Jupyter o IPython."""
    from .ipython import register
    register(ipython)
