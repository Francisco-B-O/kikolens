# kikolens/__init__.py
# 💎 KikoLens Intelligence Core

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
