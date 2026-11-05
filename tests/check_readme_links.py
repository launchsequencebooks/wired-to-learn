"""Fail if a README links to a file that does not exist (P3-BOOK-009).

The chapter READMEs are generated, and the generator used to assume that every listing
lives in its own chapter folder. Listings that live in a kit (code/tinynet,
code/wtlkit,
code/wtlapps) therefore produced dead links such as code/ch08/layers.py. This check
walks
every README.md under the repository, resolves each relative link against the README's
folder and fails on any target that does not exist. External links (http, https, mailto)
and GitHub's virtual issue links (../../issues/...) are skipped.

Usage:
    python tests/check_readme_links.py            # every README.md under the repo
    python tests/check_readme_links.py code/ch08  # one folder
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
BAD_TEXT = ("28times28",)  # LaTeX macros that the generator used to strip into garbage


def check(readme: Path):
    problems = []
    text = readme.read_text(encoding="utf-8")
    for bad in BAD_TEXT:
        if bad in text:
            problems.append(f"{readme.relative_to(ROOT)}: contains '{bad}' "
                            "(unconverted LaTeX)")
    for m in LINK.finditer(text):
        target = m.group(1)
        if target.startswith(SKIP_PREFIXES):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        # GitHub resolves ../../issues/new/choose and similar against the repository
        # URL, not the file tree; a link above the repository root is one of these.
        resolved = (readme.parent / path).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            continue
        if not resolved.exists():
            problems.append(f"{readme.relative_to(ROOT)}: dead link {target}")
    return problems


def main(argv=None):
    roots = [Path(a) for a in (argv or sys.argv[1:])] or [ROOT]
    readmes = []
    for r in roots:
        readmes += sorted(r.rglob("README.md")) if r.is_dir() else [r]
    problems = [p for rm in readmes for p in check(rm)]
    for p in problems:
        print(p)
    if problems:
        print(f"\n{len(problems)} problem(s) in {len(readmes)} README files")
        return 1
    print(f"ok: {len(readmes)} README files, every link resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
