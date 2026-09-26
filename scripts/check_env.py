"""Environment check for the v3 pipeline.

Usage:
    python scripts/check_env.py --profile live|survey [--allow-cpu] [--json]

Exit code 0 when every requirement of the chosen profile is met, 1 otherwise.
The pipeline calls `check_environment()` at startup so a run never starts on a
machine that would silently degrade (CPU-only torch, missing COLMAP, ...).
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def _import_version(name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # ImportError, DLL policy errors, ...
        return False, f"{type(exc).__name__}: {exc}".splitlines()[0][:160]
    return True, str(getattr(mod, "__version__", "ok"))


def _colmap_cli() -> Dict[str, Any]:
    exe = shutil.which("colmap")
    info: Dict[str, Any] = {"path": exe, "cuda": False, "global_mapper": False}
    if not exe:
        return info
    try:
        out = subprocess.run([exe, "help"], capture_output=True, text=True, timeout=30)
        text = (out.stdout or "") + (out.stderr or "")
        info["runs"] = out.returncode == 0 and "COLMAP" in text
        if not info["runs"]:
            info["error"] = text.strip().splitlines()[0][:200] if text.strip() else f"exit {out.returncode}"
        info["cuda"] = "with CUDA" in text and "without CUDA" not in text
        info["global_mapper"] = "global_mapper" in text
        first = [line for line in text.splitlines() if "COLMAP" in line]
        info["version"] = first[0].strip() if first else "unknown"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def collect() -> Dict[str, Any]:
    report: Dict[str, Any] = {"python": sys.version.split()[0], "platform": sys.platform, "modules": {}}
    for name in ["numpy", "scipy", "cv2", "open3d", "pycolmap", "pymap3d", "pyproj",
                 "rasterio", "laspy", "shapely", "trimesh", "yaml", "torch",
                 "transformers", "ultralytics"]:
        ok, ver = _import_version(name)
        report["modules"][name] = {"ok": ok, "version": ver}

    torch_info = {"cuda": False, "device": None}
    if report["modules"]["torch"]["ok"]:
        import torch  # noqa: WPS433

        torch_info["cuda"] = bool(torch.cuda.is_available())
        if torch_info["cuda"]:
            torch_info["device"] = torch.cuda.get_device_name(0)
    report["torch"] = torch_info

    pyc = {"cuda": False}
    if report["modules"]["pycolmap"]["ok"]:
        import pycolmap  # noqa: WPS433

        pyc["cuda"] = bool(getattr(pycolmap, "has_cuda", False))
    report["pycolmap"] = pyc
    report["colmap_cli"] = _colmap_cli()
    report["glomap_cli"] = shutil.which("glomap")
    report["ffmpeg"] = shutil.which("ffmpeg")
    return report


REQUIRED_ALWAYS = ["numpy", "scipy", "cv2", "open3d", "pycolmap", "pymap3d", "rasterio", "shapely", "yaml"]


def evaluate(report: Dict[str, Any], profile: str, allow_cpu: bool) -> List[str]:
    """Return a list of human-readable problems. Empty list means ready."""
    problems: List[str] = []
    for name in REQUIRED_ALWAYS:
        if not report["modules"][name]["ok"]:
            problems.append(f"python module '{name}' unavailable: {report['modules'][name]['version']}")
    if profile == "live":
        for name in ["torch", "transformers"]:
            if not report["modules"][name]["ok"]:
                problems.append(f"LIVE lane needs '{name}'")
        if not report["torch"]["cuda"] and not allow_cpu:
            problems.append("torch has no CUDA device (pass --allow-cpu to accept a slow run)")
    cli = report["colmap_cli"]
    if cli.get("path") and not cli.get("runs"):
        problems.append(f"colmap CLI is installed but does not run: {cli.get('error')}")
    if profile == "survey":
        if not cli.get("path"):
            problems.append("SURVEY lane needs the colmap CLI (conda-forge colmap)")
        elif not cli.get("cuda"):
            problems.append("SURVEY lane needs a CUDA build of colmap for patch_match_stereo")
    return problems


def check_environment(profile: str = "live", allow_cpu: bool = False) -> Dict[str, Any]:
    report = collect()
    report["profile"] = profile
    report["problems"] = evaluate(report, profile, allow_cpu)
    report["ok"] = not report["problems"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["live", "survey", "core"], default="live")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check_environment(args.profile, args.allow_cpu)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"python {report['python']} on {report['platform']}")
        for name, info in report["modules"].items():
            mark = "ok " if info["ok"] else "MISSING"
            print(f"  [{mark}] {name:13s} {info['version']}")
        print(f"  torch cuda: {report['torch']}")
        print(f"  pycolmap cuda: {report['pycolmap']['cuda']}")
        print(f"  colmap cli: {report['colmap_cli']}")
        print(f"  glomap cli: {report['glomap_cli']}")
        print(f"profile '{args.profile}': {'READY' if report['ok'] else 'NOT READY'}")
        for p in report["problems"]:
            print(f"  - {p}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
