"""Check that every printed output box matches a committed log (P3-BOOK-006).

For each chapters/chNN.tex, every \\begin{console} ... \\end{console} block is compared
with the .log files of code/chNN. A box passes when its lines appear, in order and
byte-for-byte (trailing whitespace ignored), as a contiguous run inside one of the
chapter's logs; boxes made of several non-contiguous excerpts pass when every line is
present in the same log in order. Chapters with no .log files are skipped: their
outputs are checked by run_verify.py against verify_results/*.out.

Usage:
    python tests/check_outputs.py      # chapters/ beside code/ or ../latex/chapters
    python tests/check_outputs.py --chapters path/to/chapters
    python tests/check_outputs.py --chapters ... ch15 ch25

Exit status 1 when any box in a chapter that has logs does not match. The report names
the chapter, the line of the \\begin{console} and the first unmatched line, which
is what the author needs to re-run the script and regenerate the box with
scripts/outputs.py.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
BLOCK = re.compile(r"\\begin\{console\}\n(.*?)\\end\{console\}", re.S)
# "[[ ... paste ... ]]" and "??" mark unfinished boxes
PLACEHOLDER = re.compile(r"\[\[.*?\]\]|\?\?")


def default_chapters() -> Path:
    for cand in (ROOT / "chapters", ROOT.parent / "latex" / "chapters",
                 ROOT.parent / "chapters"):
        if cand.is_dir():
            return cand
    sys.exit("cannot find the chapters/ folder; pass --chapters")


def norm(lines):
    return [ln.rstrip() for ln in lines]


def contiguous(box, log):
    n = len(box)
    return any(log[i:i + n] == box for i in range(len(log) - n + 1))


def in_order(box, log):
    """Every box line occurs in the log, in order (allowing gaps between them)."""
    pos = 0
    for ln in box:
        try:
            pos = log.index(ln, pos) + 1
        except ValueError:
            return ln
    return None


def check_chapter(tex: Path, logs: list[Path]):
    text = tex.read_text(encoding="utf-8")
    log_lines = {}
    for lg in logs:
        text_log = lg.read_text(encoding="utf-8", errors="replace")
        log_lines[lg.name] = norm(text_log.splitlines())
    problems, boxes = [], 0
    for m in BLOCK.finditer(text):
        boxes += 1
        line_no = text[:m.start()].count("\n") + 1
        box = norm(m.group(1).rstrip("\n").splitlines())
        if not box:
            continue
        if PLACEHOLDER.search(m.group(1)):
            problems.append(f"{tex.name}:{line_no}: placeholder text in output box")
            continue
        if any(contiguous(box, lg) for lg in log_lines.values()):
            continue
        misses = {name: in_order(box, lg) for name, lg in log_lines.items()}
        if any(v is None for v in misses.values()):
            continue  # all lines present in order in one log: an excerpted box
        first = min(misses.values(), key=len)
        problems.append(f"{tex.name}:{line_no}: no log contains this box; "
                        f"first unmatched line: {first!r}")
    return boxes, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chapters_only", nargs="*", help="chapter ids, e.g. ch15 ch25")
    ap.add_argument("--chapters", type=Path, help="folder holding chNN.tex")
    args = ap.parse_args(argv)
    chapters = args.chapters or default_chapters()

    total_boxes, problems, checked = 0, [], 0
    for tex in sorted(chapters.glob("ch*.tex")):
        ch = tex.stem
        if args.chapters_only and ch not in args.chapters_only:
            continue
        logs = sorted((CODE / ch).glob("*.log"))
        if not logs:
            continue
        checked += 1
        boxes, probs = check_chapter(tex, logs)
        total_boxes += boxes
        problems += probs
    for p in problems:
        print(p)
    print(f"\n{checked} chapters with logs, {total_boxes} output boxes, "
          f"{len(problems)} mismatch(es)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
