"""This book's own text (Chapters 1-28) as a corpus of short passages with citations."""
import json
import re
from pathlib import Path

CHAPTERS = Path(__file__).resolve().parents[2] / "chapters"
PASSAGES_JSON = Path(__file__).resolve().parent / "data" / "passages.json"   # shipped result


def tex_to_text(tex):
    """A rough LaTeX-to-text conversion: keep the prose, drop code, figures and markup."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)                                   # comments
    for env in ["figure", "table", "tikzpicture", "lstlisting", "console", "mathcorner", "chaptergoals"]:
        tex = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", tex, flags=re.S)
    tex = re.sub(r"\\codefile\{[^}]*\}\{[^}]*\}", " ", tex)
    for cmd, txt in {"ln": "ln", "log": "log", "approx": "≈", "times": "×", "sigma": "σ", "eta": "η",
                     "alpha": "α", "lambda": "λ", "theta": "θ", "partial": "∂", "le": "≤", "ge": "≥"}.items():
        tex = re.sub(rf"\\{cmd}(?![a-zA-Z])", txt, tex)                   # keep simple math readable
    tex = re.sub(r"\$([^$]*)\$", r"\1", tex)                                 # inline math: keep its content
    tex = re.sub(r"\\(textbf|emph|texttt|textit|mbox)\{([^{}]*)\}", r"\2", tex)
    tex = re.sub(r"\s*\((Figure|Table|Listing)s?~?\\ref\{[^}]*\}[a-z]?\)", "", tex)   # "(Figure 3)" -> ""
    tex = re.sub(r"(Figure|Table|Listing)~?\\ref\{[^}]*\}", lambda m: "the " + m.group(1).lower(), tex)
    tex = re.sub(r"\\(ref|label|cite|index)\{[^}]*\}", "", tex)
    tex = re.sub(r"\\begin\{\w+\}(\[[^\]]*\])?|\\end\{\w+\}|\\item", " ", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex).replace("~", " ")
    tex = re.sub(r"[{}\\]", "", tex).replace("``", '"').replace("''", '"').replace("---", "—").replace("--", "–")
    return re.sub(r"[ \t]+", " ", tex)


def passages(max_words=100):
    """Split every chapter into passages of about max_words words, never across sections."""
    if not CHAPTERS.exists():                     # the LaTeX sources are not public: use the shipped split
        return json.loads(PASSAGES_JSON.read_text(encoding="utf-8"))["passages"]
    out = []
    for n in range(1, 29):
        tex = (CHAPTERS / f"ch{n:02d}.tex").read_text()
        title = re.search(r"\\chapter\{([^}]*)\}", tex).group(1)
        parts = re.split(r"\\(?:sub)?section\*?\{([^}]*)\}", tex)          # [intro, title1, body1, title2, ...]
        sections = [("Introduction", parts[0])] + list(zip(parts[1::2], parts[2::2]))
        for sec, body in sections:
            if sec in ("Summary", "Exercises"):
                continue
            sentences = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9\"(])", " ".join(tex_to_text(body).split()))
            chunk = []
            for s in sentences + [None]:                  # whole sentences, about max_words per passage
                if s is None or (chunk and len(" ".join(chunk + [s]).split()) > max_words):
                    if len(" ".join(chunk).split()) >= 25:
                        out.append({"chapter": n, "chapter_title": title, "section": sec, "text": " ".join(chunk)})
                    chunk = []
                if s is not None:
                    chunk.append(s)
    return out


if __name__ == "__main__":
    P = passages()
    if CHAPTERS.exists():                         # built from the sources: refresh the shipped copy
        PASSAGES_JSON.parent.mkdir(exist_ok=True)
        note = ("Derived from the book's text for use with the companion code; "
                "not for redistribution outside this repository.")
        PASSAGES_JSON.write_text(json.dumps({"note": note, "passages": P}, indent=0,
                                            ensure_ascii=False), encoding="utf-8")
    print(f"{len(P):,} passages, {sum(len(p['text'].split()) for p in P):,} words, from 28 chapters")
    p = P[200]
    print(f"passage 200 (Chapter {p['chapter']}, '{p['section']}'):\n  {p['text'][:230]}...")
