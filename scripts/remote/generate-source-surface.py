#!/usr/bin/env python3
"""Generate the deterministic M0 C# port source-surface inventory."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_COMMIT = "400fd31054597515f47125691032c04b1c3ee24e"


def classify(path: Path) -> tuple[str, str]:
    normalized = path.as_posix().lower()
    if "multigpu" in normalized or "all2all" in normalized or "all_to_all" in normalized or "_mgpu" in normalized:
        return "deferred", "Multi-GPU, distributed, NCCL, and all-to-all paths are outside the single-GPU port."
    if (
        "/dsl_kernels/" in normalized
        or "/ltx_kernels/vae/" in normalized
        or "sm100" in normalized
        or "tcgen05" in normalized
    ):
        return "deferred", "B200/datacenter-Blackwell CUTLASS DSL path; an in-scope eager or non-B200 fallback is inventoried separately."
    return "in_scope", "Required by the single-GPU C# port."


def module_name(path: Path) -> str:
    parts = path.parts
    if "src" in parts:
        start = parts.index("src") + 1
        module_parts = list(parts[start:])
        if module_parts[-1] == "__init__.py":
            module_parts.pop()
        else:
            module_parts[-1] = Path(module_parts[-1]).stem
        return ".".join(module_parts)
    if "scripts" in parts:
        package = path.parts[1].replace("-", "_")
        return f"{package}.scripts.{path.stem}"
    return path.with_suffix("").as_posix().replace("/", ".")


def production_python_files() -> list[Path]:
    files = set(ROOT.glob("packages/*/src/**/*.py"))
    files.update(ROOT.glob("packages/*/scripts/*.py"))
    files.update(ROOT.glob("packages/*/setup.py"))
    return sorted(path.relative_to(ROOT) for path in files)


def decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return f"{decorator_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def string_args(call: ast.Call) -> list[str]:
    return [arg.value for arg in call.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]


def has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            if ast.unparse(node.test) in {"__name__ == '__main__'", "'__main__' == __name__"}:
                return True
        except (AttributeError, ValueError):
            pass
    return False


def cli_declarations(path: Path, tree: ast.AST) -> list[dict[str, object]]:
    declarations: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = decorator_name(node.func)
        names: list[str] = []
        kind = ""
        if called.endswith(".add_argument"):
            raw = string_args(node)
            names = raw or ["<positional>"]
            kind = "argparse"
        if not kind:
            continue
        declarations.append(
            {
                "path": path.as_posix(),
                "line": node.lineno,
                "kind": kind,
                "declared_names": sorted(set(names)),
            }
        )
    return declarations


def typer_parameter_flags(tree: ast.AST, path: Path) -> list[dict[str, object]]:
    declarations: list[dict[str, object]] = []
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        parameters = list(function.args.posonlyargs) + list(function.args.args)
        defaults = [None] * (len(parameters) - len(function.args.defaults)) + list(function.args.defaults)
        for parameter, default in zip(parameters, defaults, strict=True):
            if not isinstance(default, ast.Call):
                continue
            called = decorator_name(default.func)
            if called not in {"typer.Option", "typer.Argument"}:
                continue
            explicit = [value for value in string_args(default) if value.startswith("-")]
            if called == "typer.Option":
                names = explicit or ["--" + parameter.arg.replace("_", "-")]
                kind = "typer-option"
            else:
                names = [f"<{parameter.arg}>"]
                kind = "typer-argument"
            declarations.append(
                {
                    "path": path.as_posix(),
                    "line": default.lineno,
                    "kind": kind,
                    "declared_names": names,
                }
            )
    return declarations


def native_inventory() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    suffixes = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
    paths = sorted(
        path.relative_to(ROOT)
        for path in (ROOT / "packages").rglob("*")
        if path.is_file() and path.suffix in suffixes
    )
    sources: list[dict[str, object]] = []
    kernels: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    global_pattern = re.compile(
        r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:[\w:<>,*&]+\s+)+(?P<name>[A-Za-z_]\w*)\s*\(",
        re.MULTILINE,
    )
    binding_pattern = re.compile(r"\bm\.def\(\s*\"(?P<name>[^\"]+)\"")

    def erase_comments(text: str) -> str:
        pattern = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)

        def replacement(match: re.Match[str]) -> str:
            return "".join("\n" if character == "\n" else " " for character in match.group(0))

        return pattern.sub(replacement, text)

    for path in paths:
        status, reason = classify(path)
        sources.append({"path": path.as_posix(), "status": status, "reason": reason})
        text = (ROOT / path).read_text(errors="replace")
        code = erase_comments(text)
        for match in global_pattern.finditer(code):
            kernels.append(
                {
                    "name": match.group("name"),
                    "path": path.as_posix(),
                    "line": code.count("\n", 0, match.start()) + 1,
                    "implementation": "cuda-cpp",
                    "status": status,
                    "reason": reason,
                }
            )
        for match in binding_pattern.finditer(code):
            bindings.append(
                {
                    "name": match.group("name"),
                    "path": path.as_posix(),
                    "line": code.count("\n", 0, match.start()) + 1,
                    "status": status,
                    "reason": reason,
                }
            )

    for path in production_python_files():
        text = (ROOT / path).read_text()
        try:
            tree = ast.parse(text, filename=path.as_posix())
        except SyntaxError:
            continue
        status, reason = classify(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {decorator_name(decorator) for decorator in node.decorator_list}
            implementation = ""
            if any(
                name == "jit" or name.endswith("triton.jit") or name.endswith("triton.autotune")
                for name in decorators
            ):
                implementation = "triton"
            elif any(name.endswith("cute.kernel") for name in decorators):
                implementation = "cutlass-cute-dsl"
            if implementation:
                kernels.append(
                    {
                        "name": node.name,
                        "path": path.as_posix(),
                        "line": node.lineno,
                        "implementation": implementation,
                        "status": status,
                        "reason": reason,
                    }
                )
    kernels.sort(key=lambda item: (str(item["path"]), int(item["line"]), str(item["name"])))
    bindings.sort(key=lambda item: (str(item["path"]), int(item["line"]), str(item["name"])))
    return sources, kernels, bindings


def build_inventory() -> dict[str, object]:
    files = production_python_files()
    modules: list[dict[str, object]] = []
    packages: list[dict[str, object]] = []
    entrypoints: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []
    pipelines: list[dict[str, object]] = []

    for path in files:
        status, reason = classify(path)
        name = module_name(path)
        item = {"name": name, "path": path.as_posix(), "status": status, "reason": reason}
        modules.append(item)
        if path.name == "__init__.py":
            packages.append(item.copy())

        text = (ROOT / path).read_text()
        try:
            tree = ast.parse(text, filename=path.as_posix())
        except SyntaxError as error:
            raise RuntimeError(f"cannot parse {path}:{error.lineno}") from error
        declarations = cli_declarations(path, tree) + typer_parameter_flags(tree, path)
        flags.extend(declarations)
        if has_main_guard(tree):
            local_flags = sorted(
                {
                    declared
                    for declaration in declarations
                    for declared in declaration["declared_names"]
                }
            )
            entrypoints.append(
                {
                    "name": name,
                    "path": path.as_posix(),
                    "invocation": f"python -m {name}" if "/src/" in path.as_posix() else f"python {path.as_posix()}",
                    "locally_declared_flags": local_flags,
                    "status": status,
                    "reason": reason,
                }
            )

        if path.parts[:4] == ("packages", "ltx-pipelines", "src", "ltx_pipelines") and len(path.parts) == 5:
            classes = sorted(
                node.name for node in tree.body if isinstance(node, ast.ClassDef) and "Pipeline" in node.name
            )
            names = classes or ([path.stem] if has_main_guard(tree) else [])
            for class_name in names:
                pipelines.append(
                    {
                        "name": class_name,
                        "module": name,
                        "path": path.as_posix(),
                        "status": status,
                        "reason": reason,
                    }
                )

    native_sources, native_kernels, native_bindings = native_inventory()
    flags.sort(key=lambda item: (str(item["path"]), int(item["line"]), str(item["kind"])))
    counts = {
        "python_packages": len(packages),
        "python_modules": len(modules),
        "cli_entrypoints": len(entrypoints),
        "cli_flag_declarations": len(flags),
        "pipelines": len(pipelines),
        "native_sources": len(native_sources),
        "native_kernels": len(native_kernels),
        "native_bindings": len(native_bindings),
    }
    return {
        "schema_version": 1,
        "source_reference_commit": REFERENCE_COMMIT,
        "scope": "single-GPU LTX-2 C# port on Linux/CUDA; Python remains the numerical oracle",
        "classification_rules": {
            "in_scope": "Single-GPU model, pipeline, trainer, media, storage, CLI, quantization, LoRA, and non-B200 kernel surface.",
            "deferred": "Multi-GPU/distributed/all-to-all, B200-only CUTLASS DSL, and full fine-tuning are deferred by C_SHARP_PORT_PLAN.md.",
        },
        "explicit_deferred_features": [
            {"name": "multi-GPU pipelines and modules", "reason": "Single-GPU port only."},
            {"name": "DDP/FSDP/full fine-tuning", "reason": "Source documents full fine-tuning as a multi-GPU FSDP workload."},
            {"name": "CUDA IPC/NCCL/all-to-all", "reason": "Multi-GPU transport is outside scope."},
            {"name": "B200 SM100/SM10x CUTLASS DSL kernels", "reason": "B200 is explicitly excluded; correctness fallbacks remain in scope."},
        ],
        "counts": counts,
        "python_packages": packages,
        "python_modules": modules,
        "cli_entrypoints": entrypoints,
        "cli_flag_declarations": flags,
        "pipelines": pipelines,
        "native_sources": native_sources,
        "native_kernels": native_kernels,
        "native_bindings": native_bindings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/M0/source-surface.json")
    args = parser.parse_args()
    inventory = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
