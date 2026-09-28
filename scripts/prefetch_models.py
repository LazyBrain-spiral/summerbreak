"""Download every model weight the pipeline uses into the local caches.

Run once on a machine with internet, then the pipeline works offline:

    python scripts/prefetch_models.py            # all models (~6.5 GB, Pi3X is ~5 GB)
    python scripts/prefetch_models.py --no-pi3x  # skip the Pi3X weights

In Docker the caches live on the /cache volume (HF_HOME, TORCH_HOME), so this
only has to run once per volume:

    docker compose run --rm app python scripts/prefetch_models.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pi3x", action="store_true", help="skip the ~5 GB Pi3X weights")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    from src.core.config import DEFAULTS

    repos = [DEFAULTS["dense"]["model"], DEFAULTS["completion"]["facade_model"]]
    if not args.no_pi3x:
        repos.append("yyfz233/Pi3X")
    failed = []
    for repo in repos:
        print(f"[prefetch] {repo}", flush=True)
        try:
            snapshot_download(repo)
        except Exception as exc:  # keep going, report at the end
            failed.append(f"{repo}: {exc}")

    print("[prefetch] YOLO segmentation weights", flush=True)
    try:
        from ultralytics import YOLO

        YOLO(os.path.join(ROOT, DEFAULTS["masks"]["model"]))
    except Exception as exc:
        failed.append(f"yolo: {exc}")

    print("[prefetch] LaMa inpainting weights", flush=True)
    try:
        from simple_lama_inpainting import SimpleLama

        SimpleLama(device="cpu")
    except Exception as exc:
        failed.append(f"lama: {exc}")

    if failed:
        print("[prefetch] FAILED:\n  " + "\n  ".join(failed))
        return 1
    print("[prefetch] all models cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
