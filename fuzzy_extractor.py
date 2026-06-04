from __future__ import annotations

from math import comb
from typing import Iterable, List


def repetition_encode(bits: Iterable[int], repetitions: int) -> List[int]:
    encoded = []
    for bit in bits:
        encoded.extend([int(bit) & 1] * int(repetitions))
    return encoded


def repetition_decode(chunks: Iterable[Iterable[int]], tie_is_failure: bool = True) -> List[int]:
    decoded = []
    for chunk in chunks:
        values = [int(x) & 1 for x in chunk]
        ones = sum(values)
        zeros = len(values) - ones
        if ones > zeros:
            decoded.append(1)
        elif zeros > ones:
            decoded.append(0)
        else:
            decoded.append(0 if tie_is_failure else 1)
    return decoded


def majority_vote_success_probability(p_flip: float, n: int) -> float:
    if n % 2 == 0:
        raise ValueError("majority_vote_success_probability expects odd n")
    return sum(
        comb(n, k) * (p_flip**k) * ((1.0 - p_flip) ** (n - k))
        for k in range(0, (n - 1) // 2 + 1)
    )
