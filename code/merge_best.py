#!/usr/bin/env python
"""Merge best_hyperparameters.json files from split --arch/--target pods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSON files to merge")
    ap.add_argument("-o", "--out", default="outputs/tables/best_hyperparameters.json")
    args = ap.parse_args()
    merged = {}
    for p in args.paths:
        blob = json.loads(Path(p).read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            raise SystemExit(f"{p} is not an object")
        merged.update(blob)
        print(f"  {p}: {len(blob)} keys")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(merged)} keys)")


if __name__ == "__main__":
    main()
