#!/usr/bin/env python
"""Fill [[KEY]] in manuscript/manuscript_master.md from manuscript_numbers.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "manuscript" / "manuscript_master.md"
OUT_MD = ROOT / "manuscript" / "manuscript_filled.md"
NUMBERS = Path(__file__).resolve().parent / "outputs" / "manuscript_numbers.json"


def fill(text: str, vals: dict) -> str:
    out = text
    for k, v in vals.items():
        out = out.replace(f"[[{k}]]", str(v))
    return out


def main():
    if not MASTER.exists():
        raise SystemExit(f"missing {MASTER}")
    vals = json.load(open(NUMBERS, encoding="utf-8")) if NUMBERS.exists() else {}
    src = MASTER.read_text(encoding="utf-8")
    keys = set(re.findall(r"\[\[([A-Z0-9_@]+)\]\]", src))
    filled = fill(src, vals)
    OUT_MD.write_text(filled, encoding="utf-8")
    left = sorted(set(re.findall(r"\[\[([A-Z0-9_@]+)\]\]", filled)))
    print(f"wrote {OUT_MD}")
    print(f"placeholders {len(keys)}; still open {len(left)}")
    if left:
        print("open:", ", ".join(left[:50]))
    missing = sorted(k for k in keys if k not in vals)
    if missing:
        print("not in json:", ", ".join(missing[:50]))


if __name__ == "__main__":
    main()
