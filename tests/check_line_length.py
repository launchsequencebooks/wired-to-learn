"""Fail if any source line in code/ is longer than the book's column (P3-BOOK-002).

The book prints listings in a column that holds 88 characters of DejaVu Sans Mono at
the listing size. A longer line would be soft-wrapped by the typesetter, and a wrapped
line is the single most common cause of "the code in the book does not run" reviews.
So the rule is simple: no line in any .py file under code/ may exceed MAX characters.

Usage:
    python tests/check_line_length.py            # check everything under code/
    python tests/check_line_length.py path ...   # check the given files or folders
    python tests/check_line_length.py --max 79   # a different limit

Exit status is 1 when any line is too long; each offender is printed as
    path:line: length
so an editor can jump to it. Run it as a pre-commit hook and in CI.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX = 88


def too_long(path: Path, limit: int):
    """Yield (line_number, length) for every line of `path` longer than `limit`."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    for n, line in enumerate(text.splitlines(), 1):
        if len(line.rstrip("\n")) > limit:
            yield n, len(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=[str(ROOT / "code")])
    ap.add_argument("--max", type=int, default=MAX,
                    help=f"maximum line length (default {MAX})")
    args = ap.parse_args(argv)

    files = []
    for p in map(Path, args.paths):
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]

    bad = 0
    for f in files:
        for n, length in too_long(f, args.max):
            bad += 1
            try:
                shown = f.relative_to(ROOT)
            except ValueError:
                shown = f
            print(f"{shown}:{n}: {length} characters (limit {args.max})")
    if bad:
        print(f"\n{bad} line(s) over {args.max} characters in {len(files)} files")
        return 1
    print(f"ok: {len(files)} files, no line over {args.max} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
