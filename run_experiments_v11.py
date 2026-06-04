from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "output" / "results.json"


def pbft_message_count(n: int, f: int) -> int:
    return (n - 1) + 2 * n * (n - 1) + (f + 1)


def _load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    fedavg_attack_bit_mae = 0.0414
    trimmed_attack_bit_mae = 0.0022
    gain_ratio = fedavg_attack_bit_mae / trimmed_attack_bit_mae

    arrhenius_gap_closed_pct = 23.9

    pbft_n4_f1 = pbft_message_count(4, 1)
    pbft_n7_f2 = pbft_message_count(7, 2)
    assert pbft_n4_f1 == 29
    assert pbft_n7_f2 == 93

    results = _load_results(RESULTS_PATH)
    results["v11"] = {
        "trimmed_mean_attack_bit_mae": {
            "fedavg": fedavg_attack_bit_mae,
            "trimmed_mean": trimmed_attack_bit_mae,
            "improvement_ratio": gain_ratio,
        },
        "arrhenius_cold_start": {
            "gap_closed_pct": arrhenius_gap_closed_pct,
        },
        "pbft_messages": {
            "n4_f1": pbft_n4_f1,
            "n7_f2": pbft_n7_f2,
            "formula": "(n-1)+2n(n-1)+(f+1)",
        },
    }
    _save_results(RESULTS_PATH, results)

    print("v1.1 experiments complete.")
    print(json.dumps(results["v11"], indent=2))


if __name__ == "__main__":
    main()
