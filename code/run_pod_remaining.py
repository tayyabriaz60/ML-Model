"""
Client milestone compute: observed-mask CIs, per-station, Table 9 + Fig. 8.

    python run_pod_remaining.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent


def main():
    steps = [
        ("regenerate_headline_eval.py", []),
        ("ci_mask_report.py", []),
        ("build_regime_shap.py", []),
    ]
    for script, args in steps:
        print(f"\n=== {script} ===")
        subprocess.check_call([sys.executable, str(CODE / script), *args])
    print("\nAll pod-remaining steps finished.")


if __name__ == "__main__":
    main()
