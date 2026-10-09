"""
Merge improved §5.4, §7.3, §8 from manuscript/patches/ into manuscript_master.md.

Usage (after saving the client's ~15k-word master as manuscript/manuscript_master.md):

    python merge_into_master.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "manuscript" / "manuscript_master.md"
PATCHES = ROOT / "manuscript" / "patches"


def _replace_section(text: str, start_pat: str, end_pat: str, body: str) -> str:
    body = body.strip() + "\n\n"
    pat = re.compile(
        rf"({start_pat}.*?)(?={end_pat})",
        re.DOTALL | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        raise SystemExit(f"section not found: {start_pat!r}")
    return text[: m.start()] + body + text[m.end() :]


def main():
    if not MASTER.exists():
        raise SystemExit(f"missing {MASTER} — save the client master here first")
    text = MASTER.read_text(encoding="utf-8")
    n_words = len(text.split())
    if n_words < 8000:
        print(f"warning: master is only {n_words} words — expected client ~15k file")
    p54 = (PATCHES / "section_5_4_agreement.md").read_text(encoding="utf-8")
    p73 = (PATCHES / "section_7_3_regimes.md").read_text(encoding="utf-8")
    p8 = (PATCHES / "section_8_discussion.md").read_text(encoding="utf-8")
    text = _replace_section(text, r"### 5\.4 Agreement", r"### 5\.5|## 6\.|^---\s*$", p54)
    text = _replace_section(text, r"### 7\.3 ", r"### 7\.4 ", p73)
    text = _replace_section(text, r"## 8\. Discussion", r"## 9\. ", p8)
    backup = MASTER.with_suffix(".md.bak")
    backup.write_text(MASTER.read_text(encoding="utf-8"), encoding="utf-8")
    MASTER.write_text(text, encoding="utf-8")
    print(f"merged patches -> {MASTER} ({len(text.split())} words); backup {backup}")


if __name__ == "__main__":
    main()
