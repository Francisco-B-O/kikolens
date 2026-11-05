"""Plugins: análisis y lectores de datos propios que se suman a KikoLens sin tocar su código.

Un plugin es un archivo `.py` (o un paquete instalado que declare el punto de entrada `kikolens.plugins`) con una función `register(registry)`:

    def register(registry):
        @registry.analysis("margin_by", "Margin by group.", params={"group": {"type": "column"}, "cost": {"type": "column"}, "price": {"type": "column"}})
        def margin_by(df, group, cost, price):
            table = df.assign(margin=df[price] - df[cost]).groupby(group, as_index=False)["margin"].mean()
            return {"summary": f"Best margin: {table.loc[table.margin.idxmax(), group]}", "table": table}

        registry.loader([".dat"], lambda path: pd.read_csv(path, sep="|"))

Lo que se registra aparece en el panel (sección Plugins), en `kikolens run`, en `kl.run`, en el servidor de API y entre las herramientas del analista.

SEGURIDAD: cargar un plugin ejecuta su código con tus permisos. Solo se leen los de `~/.kikolens/plugins`, los de `[plugins] paths` en `kikolens.toml`
y los paquetes que has instalado; nunca los que vienen dentro de un archivo de datos, un proyecto o una petición. `KIKOLENS_PLUGINS=0` los desactiva todos.
Un plugin que falla al cargarse se anota en `errors()` y no impide arrancar."""
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from kikolens import config

NAME = re.compile(r"[a-z][a-z0-9_]{1,40}")
PARAM_TYPES = ("column", "columns", "number", "integer", "text", "choice", "bool")
MAX_TEXT = 500


@dataclass
class Analysis:
    name: str
    fn: object
    description: str
    params: dict = field(default_factory=dict)
    plugin: str = ""

    def tool_params(self) -> dict | None:
        """Los parámetros en el formato compacto de las herramientas del analista, o None si alguno no se puede describir ahí (texto libre)."""
        out = {}
        for key, spec in self.params.items():
            optional = "?" if "default" in spec else ""
            kind = spec["type"]
            if kind in ("column", "columns"):
                out[key] = kind + optional
            elif kind in ("number", "integer"):
                out[key] = "number" + optional
            elif kind == "choice":
                out[key] = "text:" + "|".join(map(str, spec["choices"])) + optional
            elif kind == "bool":
                out[key] = "text:true|false" + optional
            else:
                return None
        return out


@dataclass
class Registry:
    analyses: dict = field(default_factory=dict)
    loaders: dict = field(default_factory=dict)
    current: str = ""

    def analysis(self, name: str, description: str = "", params: dict | None = None):
        """Decorador que registra `fn(df, **parámetros)`. Debe devolver un diccionario (con `summary`, `table`, `figure`...) o un DataFrame."""
        if not NAME.fullmatch(name or ""):
            raise ValueError(f"Analysis name '{name}' must be lowercase letters, digits and underscores, starting with a letter.")
        spec = _check_params(name, params or {})

        def register(fn):
            if name in self.analyses:
                raise ValueError(f"An analysis called '{name}' is already registered (by {self.analyses[name].plugin or 'another plugin'}).")
            if _builtin(name):
                raise ValueError(f"'{name}' is the name of a built-in KikoLens function; choose another.")
            self.analyses[name] = Analysis(name, fn, description or (fn.__doc__ or "").strip().split("\n")[0], spec, self.current)
            return fn

        return register

    def loader(self, extensions, fn=None):
        """Enseña a KikoLens a leer archivos con esas extensiones: `fn(ruta) -> DataFrame`. También sirve como decorador."""
        exts = [extensions] if isinstance(extensions, str) else list(extensions)
        for e in exts:
            if not re.fullmatch(r"\.[a-z0-9]{1,10}", e):
                raise ValueError(f"'{e}' is not a file extension like '.dat'.")

        def register(func):
            for e in exts:
                if e in self.loaders:
                    raise ValueError(f"A loader for '{e}' is already registered.")
                self.loaders[e] = func
            return func

        return register(fn) if fn else register


def _builtin(name: str) -> bool:
    try:
        from kikolens import api
        return name in api.FUNCTIONS
    except ImportError:
        return False


def _check_params(name: str, params: dict) -> dict:
    clean = {}
    for key, spec in params.items():
        spec = {"type": spec} if isinstance(spec, str) else dict(spec)
        if not re.fullmatch(r"[a-z_]\w{0,40}", key):
            raise ValueError(f"{name}: parameter name '{key}' is not valid.")
        if spec.get("type") not in PARAM_TYPES:
            raise ValueError(f"{name}.{key}: type must be one of {', '.join(PARAM_TYPES)}.")
        if spec["type"] == "choice" and not spec.get("choices"):
            raise ValueError(f"{name}.{key}: a 'choice' parameter needs its 'choices'.")
        spec.setdefault("label", key.replace("_", " ").capitalize())
        clean[key] = spec
    return clean


# ── Validación de los argumentos ─────────────────────────────────────────────
def validate_args(analysis: Analysis, df: pd.DataFrame, args: dict) -> dict:
    """Comprueba y convierte los argumentos contra los datos; devuelve los que se pasan a la función o lanza ValueError con la razón."""
    args = dict(args or {})
    unknown = set(args) - set(analysis.params)
    if unknown:
        raise ValueError(f"{analysis.name} does not take {sorted(unknown)}. It takes: {', '.join(analysis.params) or 'nothing'}.")
    clean = {}
    for key, spec in analysis.params.items():
        value = args.get(key)
        if value is None or value == "" or value == []:
            if "default" not in spec:
                raise ValueError(f"{analysis.name} needs '{key}'.")
            clean[key] = spec["default"]
            continue
        kind = spec["type"]
        if kind == "column":
            clean[key] = _column(df, value, key)
        elif kind == "columns":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"'{key}' must be a list of columns.")
            clean[key] = [_column(df, v, key) for v in value]
        elif kind in ("number", "integer"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a number.") from None
            if number != number or number in (float("inf"), float("-inf")):
                raise ValueError(f"'{key}' must be a finite number.")
            if kind == "integer":
                if number != int(number):
                    raise ValueError(f"'{key}' must be a whole number.")
                number = int(number)
            if "min" in spec and number < spec["min"] or "max" in spec and number > spec["max"]:
                raise ValueError(f"'{key}' must be between {spec.get('min', '-inf')} and {spec.get('max', 'inf')}.")
            clean[key] = number
        elif kind == "choice":
            if str(value) not in map(str, spec["choices"]):
                raise ValueError(f"'{key}' must be one of {', '.join(map(str, spec['choices']))}.")
            clean[key] = next(c for c in spec["choices"] if str(c) == str(value))
        elif kind == "bool":
            if isinstance(value, str):
                if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError(f"'{key}' must be true or false.")
                value = value.lower() in ("true", "1", "yes")
            clean[key] = bool(value)
        else:
            text = str(value)
            if len(text) > MAX_TEXT:
                raise ValueError(f"'{key}' is longer than {MAX_TEXT} characters.")
            clean[key] = text
    return clean


def _column(df: pd.DataFrame, value, key: str) -> str:
    if value in df.columns:
        return value
    match = [c for c in df.columns if str(c).lower() == str(value).lower()]
    if match:
        return match[0]
    raise ValueError(f"'{key}': column '{value}' does not exist. Columns: {', '.join(map(str, df.columns))}.")


# ── Descubrimiento ───────────────────────────────────────────────────────────
_registry: Registry | None = None
_errors: list = []
_loaded: list = []


def enabled() -> bool:
    return os.getenv("KIKOLENS_PLUGINS", "1").strip().lower() not in ("0", "false", "no", "off")


def plugin_paths() -> list:
    """Archivos de plugin: la carpeta de usuario y los `[plugins] paths` de la configuración (archivos o carpetas)."""
    configured = config.settings().get("plugins", {}).get("paths", [])
    roots = [config.home_dir() / "plugins"] + [Path(p).expanduser() for p in (configured if isinstance(configured, list) else [configured])]
    found = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            found.append(root)
        elif root.is_dir():
            found += sorted(p for p in root.glob("*.py") if not p.name.startswith("_"))
    return list(dict.fromkeys(found))


def _call_register(module, registry: Registry, label: str) -> None:
    register = getattr(module, "register", None)
    if not callable(register):
        raise ValueError("it has no register(registry) function")
    registry.current = label
    register(registry)


def _load_file(path: Path, registry: Registry) -> None:
    module_name = f"kikolens_plugin_{re.sub(r'[^a-zA-Z0-9_]', '_', path.stem)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("it cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        _call_register(module, registry, path.stem)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


def load_plugins(force: bool = False) -> Registry:
    """Lee los plugins una vez (o de nuevo con `force`). Lo que falla se anota en `errors()` y el resto sigue funcionando."""
    global _registry
    if _registry is not None and not force:
        return _registry
    registry = Registry()
    _errors.clear()
    _loaded.clear()
    if enabled():
        for path in plugin_paths():
            before = (len(registry.analyses), len(registry.loaders))
            try:
                _load_file(path, registry)
                _loaded.append({"plugin": path.stem, "source": str(path), "analyses": len(registry.analyses) - before[0], "loaders": len(registry.loaders) - before[1]})
            except Exception as e:                                    # un plugin ajeno puede fallar de cualquier forma
                _errors.append(f"{path}: {type(e).__name__}: {e}")
        try:
            from importlib.metadata import entry_points
            points = entry_points(group="kikolens.plugins")
        except Exception:
            points = []
        for point in points:
            before = (len(registry.analyses), len(registry.loaders))
            try:
                target = point.load()
                if callable(target) and not hasattr(target, "register"):
                    registry.current = point.name
                    target(registry)
                else:
                    _call_register(target, registry, point.name)
                _loaded.append({"plugin": point.name, "source": f"package {point.value}", "analyses": len(registry.analyses) - before[0],
                                "loaders": len(registry.loaders) - before[1]})
            except Exception as e:
                _errors.append(f"{point.name}: {type(e).__name__}: {e}")
    registry.current = ""
    _registry = registry
    return registry


def analyses() -> dict:
    return load_plugins().analyses


def loaders() -> dict:
    return load_plugins().loaders


def errors() -> list:
    load_plugins()
    return list(_errors)


def loaded() -> list:
    load_plugins()
    return list(_loaded)


def reset() -> None:
    """Olvida lo cargado (para las pruebas o tras cambiar los archivos)."""
    global _registry
    _registry = None


# ── Ejecución ────────────────────────────────────────────────────────────────
def run(name: str, df: pd.DataFrame, **args) -> dict:
    """Ejecuta un análisis de plugin con los argumentos validados. Devuelve siempre un diccionario con `summary`."""
    found = analyses()
    if name not in found:
        raise ValueError(f"No plugin analysis called '{name}'." + (f" Available: {', '.join(found)}." if found else " No plugins are loaded."))
    analysis = found[name]
    clean = validate_args(analysis, df, args)
    try:
        result = analysis.fn(df.copy(), **clean)
    except ValueError:
        raise
    except Exception as e:                                          # el error de un plugin no debe tumbar el panel ni el servidor
        raise ValueError(f"The plugin '{analysis.plugin or name}' failed: {type(e).__name__}: {e}") from None
    if isinstance(result, pd.DataFrame):
        result = {"table": result}
    elif isinstance(result, pd.Series):
        result = {"table": result.to_frame()}
    elif not isinstance(result, dict):
        result = {"value": result}
    result = dict(result)
    result.setdefault("summary", analysis.description or f"{name} finished.")
    return result


def read_with_loader(path: str):
    """Si algún plugin sabe leer esta extensión, devuelve el DataFrame; si no, None."""
    ext = Path(str(path).split("?")[0]).suffix.lower()
    fn = loaders().get(ext)
    if fn is None:
        return None
    try:
        frame = fn(path)
    except Exception as e:
        raise ValueError(f"The loader for '{ext}' failed: {type(e).__name__}: {e}") from None
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"The loader for '{ext}' must return a DataFrame.")
    return frame
