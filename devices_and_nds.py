from __future__ import annotations

import numpy as np

from puf_chain import PUFChain


class Sensor:
    def __init__(self, *, device_id: str, puf, puf_type: str, n_bits: int = 64) -> None:
        self.device_id = str(device_id)
        self.puf = puf
        self.puf_type = str(puf_type)
        self.n_bits = int(n_bits)
        self.counter = 0
        self.chain = PUFChain()

    def send(self, payload, env, challenge, rng=None):
        clean = self.puf.clean_response(challenge)
        noisy = self.puf.respond(
            challenge,
            temp=env["temp"],
            voltage=env["voltage"],
            age_days=env["age_days"],
            rng=rng,
        )
        mask = (clean ^ noisy).astype(np.int8)

        block = self.chain.new_block(payload, noisy, self.counter)
        self.counter += 1
        block.update(
            sender_id=self.device_id,
            puf_type=self.puf_type,
            env=env,
            challenge=np.asarray(challenge, dtype=np.int8).reshape(-1).tolist(),
            flip_mask=mask.tolist(),
        )
        return block


class NDS:
    def __init__(self, n_bits: int = 64):
        self.n_bits = int(n_bits)
        self.log = []

    def process(self, block):
        env = block["env"]
        flip = np.array(block["flip_mask"], dtype=np.int8)
        row = dict(
            timestamp=env["t"],
            device_id=block["sender_id"],
            puf_type=block["puf_type"],
            temp=float(env["temp"]),
            voltage=float(env["voltage"]),
            age_days=float(env["age_days"]),
            is_attack=int(env["is_attack"]),
            counter=int(block["counter"]),
            n_flipped=int(flip.sum()),
        )
        for i in range(self.n_bits):
            row[f"flipped_{i}"] = int(flip[i])
        self.log.append(row)
        return block
