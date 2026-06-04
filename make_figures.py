from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "output" / "results.json"
MANIFEST_PATH = ROOT / "output" / "figure_manifest.json"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Run simulate.py and experiment scripts before make_figures.py.")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    manifest = {
        "status": "generated",
        "notes": "This simulator ships metric manifests; publication figures can be rendered from these values.",
        "highlights": {
            "adaptive_cells": results.get("v2", {}).get("adaptive_ecc", {}).get("cells"),
            "adaptive_saving_pct": results.get("v2", {}).get("adaptive_ecc", {}).get(
                "saving_vs_512_pct"
            ),
            "isolation_recall_pct": (
                results.get("v2", {}).get("isolation_forest", {}).get("recall", 0.0) * 100.0
            ),
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
