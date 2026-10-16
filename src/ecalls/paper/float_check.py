"""Where each exhibit of the compiled paper lands against the page that first mentions it.

    python -m ecalls.paper.float_check [paper_dir]

Reads the float pages from `main.aux` and the text of each page of `main.pdf` (pdftotext), then
prints, per figure and table, its page, the page of its first mention in the text, and a flag when
the exhibit comes before its mention, more than one page after it, or after the page where the
references start. Captions do not count as mentions: a figure caption reads "Fig. 3. ..." and a
table caption is set in capitals ("TABLE IV").
"""
import re
import subprocess
import sys
from pathlib import Path

LABEL = re.compile(r"\\newlabel\{(fig|tab):[^}]+\}\{\{([0-9IVX]+)\}\{(\d+)\}")
REFERENCES = re.compile(r"R ?EFERENCES")


def floats(aux: str) -> list[tuple[str, str, int]]:
    """(kind, number, page) of every figure and table label, in source order."""
    return [(k, n, int(p)) for k, n, p in LABEL.findall(aux)]


def mention(kind: str, num: str) -> re.Pattern:
    """A mention in running text, excluding the exhibit's own caption and longer numbers."""
    if kind == "fig":
        return re.compile(rf"Fig\. {num}(?![0-9]|\. [A-Z])")
    return re.compile(rf"Table {num}(?![IVX0-9])")


def first_page(pages: list[str], pat: re.Pattern) -> int | None:
    return next((i + 1 for i, t in enumerate(pages) if pat.search(t)), None)


def verdict(fp: int, first: int | None, refp: int | None) -> str:
    if first is None:
        return "not mentioned"
    if refp is not None and fp > refp:
        return "after the references"
    if fp < first:
        return "before its mention"
    if fp - first > 1:
        return "more than a page after its mention"
    return ""


def page_texts(pdf: Path) -> list[str]:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    out = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf), "-"],
                           capture_output=True, text=True, check=True).stdout
        out.append(re.sub(r"\s+", " ", t.replace("-\n", "")))
    return out


def main(paper_dir: str = "paper") -> None:
    d = Path(paper_dir)
    pages = page_texts(d / "main.pdf")
    refp = first_page(pages, REFERENCES)
    print(f"references start on page {refp}")
    for kind, num, fp in floats((d / "main.aux").read_text(encoding="utf-8")):
        first = first_page(pages, mention(kind, num))
        v = verdict(fp, first, refp)
        print(f"{kind} {num:>4}: page {fp}, first mention page {first}" + (f"  <-- {v}" if v else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
