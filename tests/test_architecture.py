"""The layering CLAUDE.md describes, checked rather than trusted.

Every module under ``app/`` is read as source and parsed — nothing is imported,
so nothing here can reach the network or start Streamlit. From the parse this
builds the app-internal import graph and asserts the rules the architecture
rests on:

* the logic layer never reaches Streamlit, ``requests`` or ``yfinance``, even
  transitively, except through the two orchestrators that exist to fetch;
* the data layer never imports the logic or view layers;
* the logic layer never imports the view layer;
* no module reaches into another's private names;
* the DCF arithmetic reaches nothing in ``app.data`` at all.

The graph is transitive on purpose. The failure this guards against is not a
logic module writing ``import streamlit`` — nobody does that — but a logic
module importing a constant from a module that does.
"""

from __future__ import annotations

import ast
from collections import deque
from functools import cache
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"

#: Third-party packages the logic layer must never reach: the UI framework and
#: the two network clients.
FORBIDDEN_IN_LOGIC = frozenset({"streamlit", "requests", "yfinance"})

#: The logic modules that exist to fetch. Each is the single network-touching
#: entry point of its feature — ``load_research`` and ``load_company`` — and may
#: reach the data layer's fetchers. They still may not import Streamlit
#: themselves; see :func:`test_nothing_in_logic_imports_the_ui_or_network_directly`.
ORCHESTRATORS = frozenset({"app.logic.research.dataset", "app.logic.dcf.model"})

#: The DCF arithmetic. CLAUDE.md: no Streamlit, no network, no global state —
#: what makes batch screening an extension rather than a rewrite.
PURE_ENGINE = ("app.logic.dcf.wacc", "app.logic.dcf.valuation", "app.logic.dcf.reverse")


def _module_name(path: Path) -> str:
    parts = path.relative_to(APP.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES: dict[str, Path] = {
    _module_name(path): path
    for path in sorted(APP.rglob("*.py"))
    if "__pycache__" not in path.parts
}


def _is_package(name: str) -> bool:
    return MODULES[name].name == "__init__.py"


def _resolve(node: ast.ImportFrom, importer: str) -> str:
    if not node.level:
        return node.module or ""
    package = importer if _is_package(importer) else importer.rpartition(".")[0]
    for _ in range(node.level - 1):
        package = package.rpartition(".")[0]
    return f"{package}.{node.module}" if node.module else package


def _with_parents(name: str) -> set[str]:
    """``app.logic.research.summary`` runs every package ``__init__`` above it."""
    parts = name.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1)} & MODULES.keys()


@cache
def _tree(name: str) -> ast.Module:
    return ast.parse(MODULES[name].read_text(), filename=str(MODULES[name]))


def _type_checking_only(tree: ast.Module) -> set[int]:
    """Nodes under ``if TYPE_CHECKING:`` — imports that never run."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
            (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
            or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
        ):
            for statement in node.body:
                skipped.update(id(child) for child in ast.walk(statement))
    return skipped


@cache
def _imports(name: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(internal modules, third-party top-level packages)`` one module imports.

    ``from app.x import y`` counts as importing ``app.x.y`` when that is a
    module and ``app.x`` otherwise, so a bare constant import still lands on
    the module that defines it. Imports under ``if TYPE_CHECKING:`` are left
    out: they are annotations, and never load anything.
    """
    internal: set[str] = set()
    external: set[str] = set()
    skipped = _type_checking_only(_tree(name))

    def note(target: str) -> None:
        if target.split(".")[0] == "app":
            internal.update(_with_parents(target))
        elif target:
            external.add(target.split(".")[0])

    for node in ast.walk(_tree(name)):
        if id(node) in skipped:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node, name)
            if base == "__future__":
                continue
            note(base)
            for alias in node.names:
                if f"{base}.{alias.name}" in MODULES:
                    note(f"{base}.{alias.name}")
    internal.discard(name)
    return frozenset(internal), frozenset(external)


def _reach(start: str) -> dict[str, tuple[str, ...]]:
    """Every app module ``start`` loads, each with the chain that loads it.

    Seeded with ``start``'s own parent packages, because importing a module
    runs every ``__init__`` above it before the module itself.
    """
    paths = {name: (start, name) for name in _with_parents(start) - {start}}
    paths[start] = (start,)
    queue = deque(paths)
    while queue:
        current = queue.popleft()
        for nxt in sorted(_imports(current)[0]):
            if nxt not in paths:
                paths[nxt] = paths[current] + (nxt,)
                queue.append(nxt)
    return paths


def _forbidden_chains(start: str, forbidden: frozenset[str]) -> list[str]:
    chains = []
    for module, chain in _reach(start).items():
        for package in sorted(_imports(module)[1] & forbidden):
            chains.append(" → ".join(chain + (package,)))
    return chains


def _in_layer(layer: str) -> list[str]:
    return [name for name in MODULES if name == layer or name.startswith(layer + ".")]


# ---------------------------------------------------------------------------
# The scanner sees what it claims to
# ---------------------------------------------------------------------------


def test_the_scanner_finds_the_graph_it_checks():
    """A scanner that silently found nothing would pass every rule below.

    Same principle as the model's checks: absent inputs are *not evaluated*,
    never a pass.
    """
    assert {"app.logic.dcf.wacc", "app.logic.research.summary"} <= set(_in_layer("app.logic"))
    assert "app.data.sharadar_client" in _reach("app.logic.research.dataset")
    assert "streamlit" in _imports("app.views.research")[1]
    assert "app.logic.research" in _reach("app.logic.research.summary")


# ---------------------------------------------------------------------------
# The logic layer
# ---------------------------------------------------------------------------


def test_nothing_in_logic_imports_the_ui_or_network_directly():
    offenders = {
        name: sorted(_imports(name)[1] & FORBIDDEN_IN_LOGIC)
        for name in _in_layer("app.logic")
        if _imports(name)[1] & FORBIDDEN_IN_LOGIC
    }
    assert not offenders


#: Modules that break the rule today, each with the step that fixes it. Strict,
#: so a fix that lands without removing its entry fails as an unexpected pass.
_KNOWN_IMPURE = {
    "app.logic.portfolio_statistics": "fetches price history from the logic layer",
}


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(reason=_KNOWN_IMPURE[name], strict=True),
        )
        if name in _KNOWN_IMPURE
        else name
        for name in sorted(set(_in_layer("app.logic")) - ORCHESTRATORS)
    ],
)
def test_logic_never_reaches_the_ui_or_network(module):
    """Transitively — the way it actually breaks is through a constant.

    A logic module that needs ``SF1_CAPTIONS`` and imports it from the module
    that also holds the HTTP client loads Streamlit and ``requests`` with it.
    """
    assert not _forbidden_chains(module, FORBIDDEN_IN_LOGIC)


@pytest.mark.parametrize("module", _in_layer("app.logic"))
def test_logic_never_imports_the_view_layer(module):
    internal, _ = _imports(module)
    assert not {
        name for name in internal if name.startswith(("app.views", "app.charts"))
    }


@pytest.mark.parametrize("module", PURE_ENGINE)
def test_the_dcf_arithmetic_reaches_nothing_in_the_data_layer(module):
    assert not [name for name in _reach(module) if name.startswith("app.data")]


# ---------------------------------------------------------------------------
# The data layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", _in_layer("app.data"))
def test_data_never_imports_logic_or_views(module):
    internal, _ = _imports(module)
    assert not {
        name for name in internal if name.startswith(("app.logic", "app.views"))
    }


@pytest.mark.parametrize("module", ("app.provenance", "app.formatting"))
def test_shared_root_modules_import_no_layer(module):
    """Both layers construct these, so they may depend on neither."""
    internal, _ = _imports(module)
    assert not {
        name
        for name in internal
        if name.startswith(("app.data", "app.logic", "app.views"))
    }


# ---------------------------------------------------------------------------
# Private names stay private
# ---------------------------------------------------------------------------


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _private_reaches(name: str) -> list[str]:
    """``from app.x import _y`` and ``alias._y`` where ``alias`` is an app module."""
    found = []
    aliases: dict[str, str] = {}
    tree = _tree(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _resolve(node, name)
            if base.split(".")[0] != "app":
                continue
            for alias in node.names:
                target = f"{base}.{alias.name}"
                if target in MODULES:
                    aliases[alias.asname or alias.name] = target
                elif _private(alias.name):
                    found.append(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in MODULES and alias.asname:
                    aliases[alias.asname] = alias.name
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and _private(node.attr)
        ):
            found.append(f"{aliases[node.value.id]}.{node.attr}")
    return sorted(set(found))


#: Packages whose modules share their internals with each other, behind a
#: public facade. ``app.data.sharadar`` is the Sharadar client split by table:
#: its table modules call ``transport._request_all`` and the uncached
#: ``_fetch_*`` of their siblings, and nothing outside it may — the rest of the
#: app goes through ``app.data.sharadar_client``.
_IMPLEMENTATION_PACKAGES = ("app.data.sharadar",)


def _same_implementation_package(importer: str, reached: str) -> bool:
    return any(
        importer.startswith(package + ".") and reached.startswith(package + ".")
        for package in _IMPLEMENTATION_PACKAGES
    )


@pytest.mark.parametrize("module", sorted(MODULES))
def test_no_module_reaches_into_another_modules_private_names(module):
    """An underscore is a promise that nothing else depends on the name.

    Tests are exempt — they are the one consumer allowed to pin internals.
    """
    assert not [
        reached
        for reached in _private_reaches(module)
        if not _same_implementation_package(module, reached)
    ]
