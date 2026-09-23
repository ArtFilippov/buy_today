"""Build narrow, reviewable overlays from pinned upstream declaration files.

Every declaration in the upstream modules is retained. Only the named signatures
below are refined; this is not a stub generator for arbitrary application code.
Run with --check in CI or --write after reviewing a dependency upgrade.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
from importlib.metadata import distribution
from importlib.util import resolve_name
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PINS = {"matplotlib": "3.11.2", "scikit-learn-stubs": "0.0.3"}
PROPERTY_METHODS = {
    "axes/_axes.pyi": {
        "set_title", "legend", "text", "hist", "barh", "bar_label", "stairs", "scatter", "imshow",
    },
    "axes/_base.pyi": {"grid", "set_xscale", "set_yscale", "set_xticks", "set_yticks", "set_xlabel", "set_ylabel"},
    "figure.pyi": {"text", "colorbar", "suptitle", "savefig"},
}


def _annotate_properties(source: str, methods: set[str]) -> str:
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    insertions: list[int] = []
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in methods:
            found.add(node.name)
            keyword = node.args.kwarg
            if keyword is not None and keyword.annotation is None:
                insertions.append(offsets[keyword.lineno - 1] + keyword.col_offset + len(keyword.arg))
    if found != methods:
        raise ValueError(f"Upstream methods changed: {methods - found}")
    for offset in sorted(insertions, reverse=True):
        source = source[:offset] + ": object" + source[offset:]
    return source


def _matplotlib(relative: str, source: str) -> str:
    if relative in PROPERTY_METHODS:
        source = _annotate_properties(source, PROPERTY_METHODS[relative])
    if relative == "figure.pyi":
        source = source.replace("fname: str | os.PathLike | IO,", "fname: str | os.PathLike[str] | IO[bytes] | IO[str],")
    if relative in {"__init__.pyi", "pyplot.py"}:
        source = source.replace("fname: str | Path | os.PathLike", "fname: str | Path | os.PathLike[str]")
        source = source.replace("fname: str | pathlib.Path | os.PathLike", "fname: str | pathlib.Path | os.PathLike[str]")
        # rc_context only reads this mapping, so its actual mapping contract is
        # covariant in values. RcParams performs validation of individual keys.
        source = source.replace("from collections.abc import Callable, Generator", "from collections.abc import Callable, Generator, Mapping")
        if relative == "__init__.pyi":
            source = source.replace("rc: dict[RcKeyType, Any] | None", "rc: Mapping[RcKeyType, object] | None")
    if relative == "pyplot.py":
        source = _declarations(source)
    return source


class _Declarations(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        node.decorator_list = [decorator for decorator in node.decorator_list
                               if isinstance(decorator, ast.Name) and decorator.id == "overload"]
        return node


def _declarations(source: str) -> str:
    tree = _Declarations().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _validation(source: str) -> str:
    source = source.replace("from collections.abc import Sequence", "from collections.abc import Iterable, Sequence")
    source = source.replace("attributes: tuple[str, ...] | None | Sequence | list[str] | str", "attributes: Sequence[str] | str | None")
    source = source.replace("all_or_any: Callable = ...", "all_or_any: Callable[[Iterable[bool]], bool] = ...")
    # check_is_fitted accepts third-party estimators too. The real function checks
    # for a fit method and raises TypeError for non-estimators at runtime.
    source = source.replace("def check_is_fitted(\n    estimator: BaseEstimator,", "def check_is_fitted(\n    estimator: object,")
    return source


def _absolute_imports(source: str, module: str) -> str:
    parts = module.split(".")[:-1]
    def replace(match: re.Match[str]) -> str:
        dots, suffix = match.groups()
        parent = parts[:len(parts) - len(dots) + 1]
        return "from " + ".".join(parent + ([suffix] if suffix else [])) + " import"
    return re.sub(r"from (\.+)([\w.]*) import", replace, source)


def _sklearn_importers() -> list[str]:
    """Find the smallest relative-import closure preserving estimator identity."""
    installed = distribution("scikit-learn-stubs")
    modules = {
        str(path)[len("sklearn-stubs/"):]: installed.locate_file(path).read_text(encoding="utf-8")
        for path in installed.files or ()
        if str(path).startswith("sklearn-stubs/") and str(path).endswith(".pyi")
    }
    imports: dict[str, set[str]] = {}
    for relative, source in modules.items():
        module = "sklearn." + relative.removesuffix(".pyi").replace("/", ".")
        package = module.rsplit(".", 1)[0]
        dependencies: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level:
                target = resolve_name("." * node.level + (node.module or ""), package)
                dependencies.add(target)
                dependencies.update(target + "." + symbol.name for symbol in node.names)
        imports[relative] = dependencies
    selected = {"base.pyi", "utils/validation.pyi", "metrics/cluster/_unsupervised.pyi"}
    while True:
        names = {"sklearn." + path.removesuffix(".pyi").replace("/", ".").removesuffix(".__init__")
                 for path in selected}
        expanded = selected | {path for path, dependencies in imports.items() if dependencies & names}
        if expanded == selected:
            break
        selected = expanded
    # These two definitions are handwritten, version-reviewed contracts.
    return sorted(selected - {"base.pyi", "metrics/cluster/_unsupervised.pyi"})


def build() -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    provenance: dict[str, object] = {"versions": PINS, "modules": {}}
    modules: dict[str, dict[str, str]] = {}
    for package, paths in {
        "matplotlib": ["__init__.pyi", "axes/__init__.pyi", "pyplot.py", *PROPERTY_METHODS],
        "scikit-learn-stubs": _sklearn_importers(),
    }.items():
        installed = distribution(package)
        if installed.version != PINS[package]:
            raise ValueError(f"Review {package} {installed.version}; expected {PINS[package]}")
        prefix = "matplotlib" if package == "matplotlib" else "sklearn-stubs"
        destination = "matplotlib" if package == "matplotlib" else "sklearn"
        for relative in paths:
            original = installed.locate_file(f"{prefix}/{relative}").read_text(encoding="utf-8")
            if package == "matplotlib":
                refined = _matplotlib(relative, original)
            elif relative == "utils/validation.pyi":
                refined = _validation(original)
            else:
                refined = original
            target = (Path(destination) / relative).with_suffix(".pyi")
            refined = _absolute_imports(refined, str(target.with_suffix("")).replace("/", "."))
            outputs[target] = (
                f"# Based on {package} {installed.version}, {prefix}/{relative}.\n"
                "# Generated by typings/build_overlays.py; see typings/README.md.\n"
                + refined
            )
            modules[str(target)] = {"source": f"{prefix}/{relative}", "sha256": sha256(original.encode()).hexdigest()}
        if package == "matplotlib":
            license_files = [entry for entry in installed.files or () if str(entry).endswith(".dist-info/LICENSE")]
            if len(license_files) != 1:
                raise ValueError("Could not identify the Matplotlib license")
            outputs[Path("licenses/matplotlib.txt")] = installed.locate_file(license_files[0]).read_text(encoding="utf-8")
    provenance["modules"] = modules
    outputs[Path("upstream.json")] = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--write", action="store_true", help="materialize reviewed declarations")
    _ = parser.add_argument("--check", action="store_true", help="verify checked-in declarations")
    arguments = parser.parse_args()
    outputs = build()
    mismatches: list[str] = []
    for relative, content in outputs.items():
        path = ROOT / relative
        if arguments.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(content, encoding="utf-8")
        elif not path.is_file() or path.read_text(encoding="utf-8") != content:
            mismatches.append(str(relative))
    if mismatches:
        raise SystemExit("Outdated overlay files: " + ", ".join(mismatches))
    print(f"{'Built' if arguments.write else 'Verified'} {len(outputs)} overlay/provenance files")


if __name__ == "__main__":
    main()
