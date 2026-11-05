"""Run the book's listings and fail if any of them crashes.

Default mode runs the NumPy chapters (Parts I-II, ch01-ch08) exactly as printed.

Fast mode (--fast) runs every chapter, ch01-ch36, with the environment variable
WTL_FAST=1 set. Scripts that train for more than a few seconds honour that variable
by cutting the run to one epoch (or one batch) so the whole book can be checked on a
CPU runner in CI; the numbers such a run prints are meaningless and are never pasted
into the book. Scripts that serve a web app, open a browser or need an API key are
skipped (SERVING below).

Usage:
    python tests/smoke_test.py                 # ch01-ch08, full runs
    python tests/smoke_test.py --fast          # ch01-ch36 with WTL_FAST=1
    python tests/smoke_test.py --fast ch16 ch30
    python tests/smoke_test.py --timeout 900   # seconds per script (default 600)

Trained files that later chapters load (see code/scripts/fetch_artifacts.py) must be in
place before a fast run of ch10, ch25 and ch26-ch36; CI fetches them first.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
NUMPY_CHAPTERS = [f"ch{n:02d}" for n in range(1, 9)]
ALL_CHAPTERS = [f"ch{n:02d}" for n in range(1, 37)]

# Scripts that never finish on their own (web servers, dashboards) or need resources a
# CI runner does not have. They are smoke-tested by their chapter's test_*.py instead.
SERVING = {
    "serve.py", "app.py", "qa_app.py", "dashboard.py", "triage_service.py",
    "rec_service.py", "make_demo_photos.py", "screenshot.py",
}


def run(script: Path, env: dict, timeout: int) -> tuple[str, float, str]:
    start = time.time()
    try:
        r = subprocess.run([sys.executable, script.name], cwd=script.parent, env=env,
                           capture_output=True, text=True, timeout=timeout)
        status = "ok" if r.returncode == 0 else "FAILED"
        err = r.stderr[-2000:]
    except subprocess.TimeoutExpired:
        status, err = "TIMEOUT", f"no result after {timeout}s"
    return status, time.time() - start, err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chapters", nargs="*",
                    help="chapter folders to run (default: all for the mode)")
    ap.add_argument("--fast", action="store_true",
                    help="run every chapter with WTL_FAST=1 (one epoch / one batch)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per script")
    args = ap.parse_args(argv)

    chapters = args.chapters or (ALL_CHAPTERS if args.fast else NUMPY_CHAPTERS)
    env = dict(os.environ)
    if args.fast:
        env["WTL_FAST"] = "1"
        env.setdefault("MPLBACKEND", "Agg")

    failed = []
    for ch in chapters:
        folder = CODE / ch
        if not folder.is_dir():
            print(f"skip   {ch}: no such folder")
            continue
        for script in sorted(folder.glob("*.py")):
            if script.name in SERVING:
                continue
            status, seconds, err = run(script, env, args.timeout)
            print(f"{status:7} {ch}/{script.name:28} {seconds:6.1f}s", flush=True)
            if status != "ok":
                failed.append(script)
                print(err)
    print(f"\n{len(failed)} failed" + (" (fast mode)" if args.fast else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
