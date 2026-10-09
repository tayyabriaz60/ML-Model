#!/usr/bin/env python
"""Fill [[KEY]] in letter/letter_text.txt from outputs/manuscript_numbers.json."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTER = ROOT / "letter" / "letter_text.txt"
OUT_TXT = ROOT / "letter" / "response_letter_filled.txt"
NUMBERS = Path(__file__).resolve().parent / "outputs" / "manuscript_numbers.json"


def main():
    if not NUMBERS.exists():
        raise SystemExit(f"missing {NUMBERS} — run: python -m src.export_numbers")
    vals = json.load(open(NUMBERS, encoding="utf-8"))
    txt = LETTER.read_text(encoding="utf-8")
    keys = set(re.findall(r"\[\[([A-Z0-9_@]+)\]\]", txt))
    missing = sorted(k for k in keys if k not in vals)
    filled = txt
    for k, v in vals.items():
        filled = filled.replace(f"[[{k}]]", str(v))
    OUT_TXT.write_text(filled, encoding="utf-8")
    left = sorted(set(re.findall(r"\[\[([A-Z0-9_@]+)\]\]", filled)))
    print(f"wrote {OUT_TXT}")
    print(f"placeholders {len(keys)}; still open {len(left)}")
    if left:
        print("open:", ", ".join(left[:40]))
    if missing:
        print("not in json:", ", ".join(missing[:40]))


if __name__ == "__main__":
    main()
