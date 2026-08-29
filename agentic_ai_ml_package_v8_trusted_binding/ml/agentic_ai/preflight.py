from __future__ import annotations

import argparse
import importlib.metadata as metadata
import re
from collections import defaultdict
from pathlib import Path

from .artifacts import ArtifactStore, spec_for
from .config import SETTINGS
from .ollama_client import OllamaClient


def parse_requirement_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return None
    m = re.match(r"^([A-Za-z0-9_.-]+)\s*==\s*([^;\s]+)", line)
    if not m:
        return None
    return m.group(1).lower().replace("_", "-"), m.group(2)


def requirement_conflicts() -> dict[str, dict[str, list[str]]]:
    versions: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for key in ("pd", "ews", "lgd", "pd_cluster"):
        req = spec_for(key).requirements
        if req and req.exists():
            for line in req.read_text(encoding="utf-8").splitlines():
                parsed = parse_requirement_line(line)
                if parsed:
                    pkg, ver = parsed
                    versions[pkg][ver].append(key)
    return {pkg: dict(v) for pkg, v in versions.items() if len(v) > 1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    store = ArtifactStore()
    print("=== Artifact preflight ===")
    ok = True
    for key in ("pd", "ews", "lgd", "pd_cluster"):
        spec = spec_for(key)
        required_paths = [spec.champion, spec.schema, spec.metadata, spec.manifest, spec.metrics, spec.reference, spec.requirements]
        if key != "pd_cluster":
            required_paths.append(spec.policy)
        missing = [str(p) for p in required_paths if p is not None and not Path(p).exists()]
        if missing:
            ok = False
            print(f"[{key}] MISSING")
            for p in missing:
                print("  -", p)
        else:
            try:
                n = len(store.feature_names(key))
                print(f"[{key}] OK | features={n} | champion={spec.champion.name}")
                if args.load_models:
                    store.bundle(key)
                    print("  model load: OK")
            except Exception as exc:
                ok = False
                print(f"[{key}] ERROR: {exc}")

    conflicts = requirement_conflicts()
    print("\n=== Requirement conflicts ===")
    if not conflicts:
        print("No exact-version conflicts found across *_requirements.txt")
    else:
        ok = False
        for pkg, by_version in conflicts.items():
            print(pkg, by_version)

    print("\n=== Installed exact-version check ===")
    for key in ("pd", "ews", "lgd", "pd_cluster"):
        req = spec_for(key).requirements
        if not req or not req.exists():
            continue
        mismatches = []
        for line in req.read_text(encoding="utf-8").splitlines():
            parsed = parse_requirement_line(line)
            if not parsed:
                continue
            pkg, expected = parsed
            try:
                actual = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                actual = "NOT_INSTALLED"
            if actual != expected:
                mismatches.append((pkg, expected, actual))
        if mismatches:
            ok = False
            print(f"[{key}] version mismatches:")
            for pkg, exp, act in mismatches[:20]:
                print(f"  {pkg}: expected {exp}, installed {act}")
        else:
            print(f"[{key}] exact versions match")

    if not args.skip_ollama:
        print("\n=== Ollama ===")
        client = OllamaClient()
        if not client.health():
            ok = False
            print(f"Ollama not reachable at {SETTINGS.ollama_host}")
        else:
            installed = client.models()
            print("Installed models:", installed)
            if SETTINGS.qwen_agent_model not in installed:
                print("WARNING: Qwen agent model not listed exactly:", SETTINGS.qwen_agent_model)
            if not SETTINGS.sahabat_model:
                ok = False
                print("SAHABAT_MODEL is not configured")
            elif SETTINGS.sahabat_model not in installed:
                print("WARNING: Sahabat model not listed exactly:", SETTINGS.sahabat_model)

    print("\nPRECHECK:", "PASS" if ok else "CHECK WARNINGS/ERRORS ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
