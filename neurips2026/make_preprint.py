#!/usr/bin/env python3
"""Generate preprint.tex from main.tex.

main.tex is the single source of truth and stays in double-blind submission
form. This script produces the de-anonymized arXiv/preprint variant by:

  1. swapping the style option `dblblindworkshop` -> `preprint`, which drops
     the margin line numbers, the anonymous author block and the
     "Submitted to ... Do not distribute" footer (see neurips_2026.sty);
  2. substituting the real author block;
  3. dropping the review-only acknowledgments stub.

preprint.tex is a build artifact. Edit main.tex, never preprint.tex.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

AUTHOR_BLOCK = r"""\author{%
  Aadi Chauhan\thanks{Equal contribution.} \\
  Stanford University \\
  \texttt{aadic@stanford.edu} \\
  \And
  Arthur Ilyasov\footnotemark[1] \\
  Stanford University \\
  \texttt{ailyasov@stanford.edu} \\
}"""

SUBSTITUTIONS = [
    ("\\usepackage[dblblindworkshop]{neurips_2026}", "\\usepackage[preprint]{neurips_2026}"),
    ("\\author{Anonymous Author(s)}", AUTHOR_BLOCK),
    ("\\begin{ack}\nAcknowledgments are omitted for review.\n\\end{ack}\n\n", ""),
    # main.tex keeps the anonymous promise; the preprint can name the repo
    ("Code, split indices, prompts, and per-example logs will be released with the camera-ready.",
     "Code, split indices, prompts, and the per-example logs behind every table are available at "
     "\\url{https://github.com/aadcha/action-conditioned-gui-agent}."),
]


def main() -> int:
    src = (HERE / "main.tex").read_text()
    for old, new in SUBSTITUTIONS:
        if src.count(old) != 1:
            print(f"error: expected exactly one occurrence of {old[:60]!r}, found {src.count(old)}",
                  file=sys.stderr)
            print("main.tex has drifted; update SUBSTITUTIONS in make_preprint.py.", file=sys.stderr)
            return 1
        src = src.replace(old, new)
    header = ("% GENERATED FILE -- do not edit. Produced by make_preprint.py from main.tex.\n"
              "% Run `make preprint` after editing main.tex.\n")
    (HERE / "preprint.tex").write_text(header + src)
    print("wrote preprint.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
