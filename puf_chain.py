from __future__ import annotations

import base64
import hashlib
import json

import numpy as np


class PUFChain:
    def __init__(self) -> None:
        self.prev_hash = "0" * 64

    @staticmethod
    def _keystream_bytes(noisy_response: np.ndarray, length: int) -> bytes:
        bits = np.asarray(noisy_response, dtype=np.int8).reshape(-1)
        if bits.size == 0:
            bits = np.zeros(1, dtype=np.int8)
        key = bytes(int(x) & 1 for x in bits.tolist())
        repeats = (length // len(key)) + 1
        return (key * repeats)[:length]

    @staticmethod
    def _xor_bytes(a: bytes, b: bytes) -> bytes:
        return bytes(x ^ y for x, y in zip(a, b))

    def new_block(self, payload: dict, noisy_response: np.ndarray, counter: int) -> dict:
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        keystream = self._keystream_bytes(noisy_response, len(payload_bytes))
        ciphertext = self._xor_bytes(payload_bytes, keystream)

        hash_material = (
            self.prev_hash.encode("utf-8")
            + f"|{int(counter)}|".encode("utf-8")
            + ciphertext
        )
        block_hash = hashlib.sha256(hash_material).hexdigest()

        block = {
            "counter": int(counter),
            "prev_hash": self.prev_hash,
            "hash": block_hash,
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        }
        self.prev_hash = block_hash
        return block
