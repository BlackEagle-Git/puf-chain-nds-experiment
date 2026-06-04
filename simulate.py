from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from devices_and_nds import NDS, Sensor
from puf_models import ArbiterPuf, HybridPuf, SRAMPuf, characterize_puf

SEED = 42
N_BITS = 64
PACKETS_PER_FAMILY = 6000
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"
RESULTS_PATH = OUTPUT_DIR / "results.json"

SRAM_REGIMES = [
    ("cold", 2676, 0),
    ("nominal", 2552, 0),
    ("warm", 302, 0),
    ("attack", 470, 1),
]


def _sample_env(rng: np.random.Generator, regime: str, t: int, is_attack: int) -> dict:
    if regime == "cold":
        temp = float(rng.uniform(10.0, 24.9))
        voltage = float(rng.normal(1.80, 0.03))
    elif regime == "nominal":
        temp = float(rng.uniform(25.0, 44.9))
        voltage = float(rng.normal(1.80, 0.02))
    elif regime == "warm":
        temp = float(rng.uniform(45.0, 64.9))
        voltage = float(rng.normal(1.80, 0.03))
    else:
        temp = float(rng.uniform(65.0, 90.0))
        voltage = float(rng.normal(1.80, 0.10) + rng.choice([-0.20, 0.20]))
    voltage = float(np.clip(voltage, 1.45, 2.15))
    age_days = float(rng.uniform(0.0, 365.0))
    return {
        "t": int(t),
        "temp": temp,
        "voltage": voltage,
        "age_days": age_days,
        "is_attack": int(is_attack),
    }


def _fieldnames() -> list[str]:
    return [
        "timestamp",
        "device_id",
        "puf_type",
        "temp",
        "voltage",
        "age_days",
        "is_attack",
        "counter",
        "n_flipped",
    ] + [f"flipped_{i}" for i in range(N_BITS)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _generate_family_dataset(
    *,
    family: str,
    puf_ctor,
    seed_offset: int,
    regimes: list[tuple[str, int, int]],
) -> list[dict]:
    rng = np.random.default_rng(SEED + seed_offset)
    sensors = [
        Sensor(
            device_id=f"{family}_dev_{idx}",
            puf=puf_ctor(n_bits=N_BITS, seed=SEED + seed_offset + idx),
            puf_type=family,
            n_bits=N_BITS,
        )
        for idx in range(2)
    ]
    nds = NDS(n_bits=N_BITS)

    packet_idx = 0
    for regime, count, is_attack in regimes:
        for _ in range(count):
            env = _sample_env(rng, regime, t=packet_idx, is_attack=is_attack)
            challenge = rng.integers(0, 2, size=N_BITS, dtype=np.int8)
            payload = {
                "sensor_value": float(rng.normal(100.0, 10.0)),
                "packet": packet_idx,
                "family": family,
            }
            sensor = sensors[packet_idx % len(sensors)]
            block = sensor.send(payload, env, challenge, rng=rng)
            nds.process(block)
            packet_idx += 1

    return nds.log


def _load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sram_rows = _generate_family_dataset(
        family="sram",
        puf_ctor=SRAMPuf,
        seed_offset=100,
        regimes=SRAM_REGIMES,
    )
    arbiter_rows = _generate_family_dataset(
        family="arbiter",
        puf_ctor=ArbiterPuf,
        seed_offset=200,
        regimes=[("nominal", PACKETS_PER_FAMILY - 470, 0), ("attack", 470, 1)],
    )
    hybrid_rows = _generate_family_dataset(
        family="hybrid",
        puf_ctor=HybridPuf,
        seed_offset=300,
        regimes=[("nominal", PACKETS_PER_FAMILY - 470, 0), ("attack", 470, 1)],
    )

    _write_csv(DATASET_DIR / "traffic_sram.csv", sram_rows)
    _write_csv(DATASET_DIR / "traffic_arbiter.csv", arbiter_rows)
    _write_csv(DATASET_DIR / "traffic_hybrid.csv", hybrid_rows)

    characterization = characterize_puf(None, None, n=500)
    summary = {
        "seed": SEED,
        "rows": {
            "sram": len(sram_rows),
            "arbiter": len(arbiter_rows),
            "hybrid": len(hybrid_rows),
        },
    }

    results = _load_results(RESULTS_PATH)
    results["seed"] = SEED
    results["offline_characterization"] = characterization
    results["dataset_summary"] = summary
    _save_results(RESULTS_PATH, results)

    print("Generated dataset CSVs and offline characterization.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
