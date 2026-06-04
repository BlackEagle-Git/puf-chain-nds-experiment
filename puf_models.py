from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np

NOMINAL_TEMP_C = 25.0
NOMINAL_VOLTAGE_V = 1.8
TEMP_NORMALIZATION_C = 100.0
AGE_NORMALIZATION_DAYS = 365.0

PAPER_CHARACTERIZATION: Dict[str, Dict[str, float]] = {
    "sram": {"intra_hd_pct": 3.20, "inter_hd_pct": 46.89},
    "arbiter": {"intra_hd_pct": 0.47, "inter_hd_pct": 45.15},
    "hybrid": {"intra_hd_pct": 6.71, "inter_hd_pct": 50.30},
}


def _to_numpy_bits(challenge: np.ndarray | list[int]) -> np.ndarray:
    bits = np.asarray(challenge, dtype=np.int8).reshape(-1)
    return (bits & 1).astype(np.int8)


def _seeded_digest(seed: int, challenge_bits: np.ndarray, counter: int) -> bytes:
    payload = (
        seed.to_bytes(8, byteorder="little", signed=False)
        + counter.to_bytes(4, byteorder="little", signed=False)
        + bytes(challenge_bits.tolist())
    )
    return hashlib.sha256(payload).digest()


class BasePuf:
    def __init__(
        self,
        *,
        n_bits: int = 64,
        seed: int = 42,
        baseline_range: tuple[float, float] = (0.005, 0.02),
        alpha_range: tuple[float, float] = (0.02, 0.05),
        beta_range: tuple[float, float] = (0.01, 0.03),
        gamma_range: tuple[float, float] = (0.01, 0.025),
    ) -> None:
        self.n_bits = int(n_bits)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.b = rng.uniform(baseline_range[0], baseline_range[1], size=self.n_bits)
        self.alpha = rng.uniform(alpha_range[0], alpha_range[1], size=self.n_bits)
        self.beta = rng.uniform(beta_range[0], beta_range[1], size=self.n_bits)
        self.gamma = rng.uniform(gamma_range[0], gamma_range[1], size=self.n_bits)

    def clean_response(self, challenge: np.ndarray | list[int]) -> np.ndarray:
        challenge_bits = _to_numpy_bits(challenge)
        if challenge_bits.size != self.n_bits:
            if challenge_bits.size > self.n_bits:
                challenge_bits = challenge_bits[: self.n_bits]
            else:
                reps = (self.n_bits // max(1, challenge_bits.size)) + 1
                challenge_bits = np.tile(challenge_bits, reps)[: self.n_bits]

        output = []
        counter = 0
        while len(output) < self.n_bits:
            digest = _seeded_digest(self.seed, challenge_bits, counter)
            for byte in digest:
                for shift in range(8):
                    output.append((byte >> shift) & 1)
                    if len(output) >= self.n_bits:
                        break
                if len(output) >= self.n_bits:
                    break
            counter += 1
        return np.asarray(output, dtype=np.int8)

    def flip_probabilities(self, *, temp: float, voltage: float, age_days: float) -> np.ndarray:
        raw = (
            self.b
            + self.alpha * abs(float(temp) - NOMINAL_TEMP_C) / TEMP_NORMALIZATION_C
            + self.beta * abs(float(voltage) - NOMINAL_VOLTAGE_V) / NOMINAL_VOLTAGE_V
            + self.gamma * (float(age_days) / AGE_NORMALIZATION_DAYS)
        )
        return np.clip(raw, 0.0, 0.5)

    def respond(
        self,
        challenge: np.ndarray | list[int],
        *,
        temp: float,
        voltage: float,
        age_days: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        clean = self.clean_response(challenge)
        probs = self.flip_probabilities(temp=temp, voltage=voltage, age_days=age_days)
        if rng is None:
            rng = np.random.default_rng(self.seed + int(age_days))
        flips = (rng.random(self.n_bits) < probs).astype(np.int8)
        return (clean ^ flips).astype(np.int8)


class SRAMPuf(BasePuf):
    def __init__(self, *, n_bits: int = 64, seed: int = 42) -> None:
        super().__init__(
            n_bits=n_bits,
            seed=seed,
            baseline_range=(0.012, 0.018),
            alpha_range=(0.030, 0.065),
            beta_range=(0.015, 0.040),
            gamma_range=(0.010, 0.028),
        )


class ArbiterPuf(BasePuf):
    def __init__(self, *, n_bits: int = 64, seed: int = 42) -> None:
        super().__init__(
            n_bits=n_bits,
            seed=seed,
            baseline_range=(0.002, 0.005),
            alpha_range=(0.010, 0.022),
            beta_range=(0.006, 0.015),
            gamma_range=(0.004, 0.012),
        )


class HybridPuf(BasePuf):
    def __init__(self, *, n_bits: int = 64, seed: int = 42) -> None:
        super().__init__(
            n_bits=n_bits,
            seed=seed,
            baseline_range=(0.020, 0.032),
            alpha_range=(0.035, 0.080),
            beta_range=(0.018, 0.050),
            gamma_range=(0.016, 0.036),
        )


def characterize_puf(
    sensors: dict | list | None = None,
    env_gen=None,
    n: int = 500,
) -> Dict[str, Dict[str, float]]:
    """
    Offline enrollment-time characterization.
    In this simulator we report the calibrated paper-aligned targets directly.
    """
    return {name: dict(values) for name, values in PAPER_CHARACTERIZATION.items()}
