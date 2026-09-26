"""Minimal: does a straight-through hard gate learn the hardness?

The trained sigmoid gate learns the ADDRESSING exactly (its 0.5-threshold is the
hand-set gate) but not the HARDNESS: raw it scores 0.420, and reaching 1.000 needs
a temperature chosen at evaluation. This replaces the soft forward pass with a
hard one, passing the sigmoid gradient straight through, so no temperature is
needed at inference.

The answer is no, and it is worse than the gate it was meant to repair: a hard
forward pass does not recover the hardness, it destroys the addressing the soft
gate had already got right. The run writes ``straight-through.json``, which the
README's gate section is rendered from, so the 0.160 is a committed measurement
rather than a sentence in prose.

    python experiments/learned_gate_straight_through.py --out straight-through.json

The soft-gate control is printed and carried into the payload first. The harness
has to reproduce 0.420 and 1.000 before the 0.160 means anything: a
straight-through number from a harness that never reproduced them is a number
about the harness. It runs in about ten seconds, which is why it is cheap enough
to guard in CI rather than trust.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from beyond_attention.agent import selective_suite, SELECTIVE_KEYS
from beyond_attention.learned_gate import (
    LearnedGate, HandSetGate, at_temperature, gate_batch, stack_batch, state_loss,
)
from learned_gate import (
    solve_rate, summarise, TRAIN_TASKS, EVAL_TASKS, SEEDS, STEPS, LR,
    SHARP_TEMPERATURE,
)

KEYS = WIDTH = SELECTIVE_KEYS

# The published soft-gate control, quoted by the experiment this one is checking
# against. The run prints its own control next to these, so a drift is visible in
# the output rather than only in the payload.
SOFT_GATE_CONTROL = {"raw": 0.420, "sharpened": 1.000}


class StraightThroughGate(LearnedGate):
    """Hard 0/1 forward; the sigmoid's gradient backward.

    `hard + (p - p.detach())` evaluates to exactly `hard` in the forward pass --
    bit-exact 0.0 or 1.0, which is what `w=1` replaces / `w=0` holds requires --
    while the backward pass sees d/dp of the soft gate.
    """
    def forward(self, features, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        p = torch.sigmoid(self.logits(features) / temperature)
        hard = (p > 0.5).to(p.dtype)
        return hard + (p - p.detach())


def as_gate(gate):
    """Adapt the module to the one-argument ``Gate`` the reader takes. No
    temperature: the forward pass is already hard."""
    return lambda instr: gate.gate(instr, 1.0)


def train(gate, tasks, *, steps=STEPS, lr=LR):
    feats, values, target, steps_target = stack_batch(gate_batch(tasks, KEYS, WIDTH))
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = state_loss(gate, feats, values, target, steps_target, dense=True)
        loss.backward()
        opt.step()
    # `.detach()` because this value is reported, not differentiated. Without it
    # torch warns that a requires_grad tensor is being read as a scalar. The
    # reported number is identical either way -- what is dropped is the graph
    # edge, not the value.
    return float(loss.detach())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="straight-through.json",
                        help="where the payload the README's gate section is "
                             "rendered from is written")
    args = parser.parse_args()

    started = time.time()
    train_tasks = selective_suite(TRAIN_TASKS, seed=0, n_keys=KEYS)
    eval_tasks = selective_suite(EVAL_TASKS, seed=1, n_keys=KEYS)

    hand_rate, tasks = solve_rate(eval_tasks, HandSetGate(KEYS, WIDTH), KEYS, WIDTH)
    print(f"hand-set gate                     {hand_rate:.3f}")

    # --- HARNESS CONTROL ---------------------------------------------------
    # The same training loop must reproduce the known soft-gate numbers. If it
    # does not, the straight-through row below is measuring the harness, not the
    # idea. Both rows go into the payload for exactly that reason.
    _plain = LearnedGate(KEYS, WIDTH, seed=0)
    train(_plain, train_tasks)
    raw_rate, _ = solve_rate(eval_tasks, at_temperature(_plain, 1.0), KEYS, WIDTH)
    print(f"soft gate, same loop, raw         {raw_rate:.3f}   "
          f"(experiment says {SOFT_GATE_CONTROL['raw']:.3f})")
    sharp_rate, _ = solve_rate(eval_tasks, at_temperature(_plain, SHARP_TEMPERATURE),
                               KEYS, WIDTH)
    print(f"soft gate, same loop, sharpened   {sharp_rate:.3f}   "
          f"(experiment says {SOFT_GATE_CONTROL['sharpened']:.3f})")

    trained, untrained, per_seed = [], [], []
    for seed in SEEDS:
        g = StraightThroughGate(KEYS, WIDTH, seed=seed)
        u = StraightThroughGate(KEYS, WIDTH, seed=seed)
        final = train(g, train_tasks)
        t = solve_rate(eval_tasks, as_gate(g), KEYS, WIDTH)[0]
        n = solve_rate(eval_tasks, as_gate(u), KEYS, WIDTH)[0]
        trained.append(t); untrained.append(n)
        per_seed.append({"seed": seed, "trained": round(t, 6),
                         "untrained": round(n, 6), "final_loss": round(final, 6)})
        print(f"  seed {seed}: straight-through trained {t:.3f}   "
              f"untrained {n:.3f}   final loss {final:.2f}")

    trained_summary = summarise(trained, [tasks] * len(SEEDS))
    untrained_summary = summarise(untrained, [tasks] * len(SEEDS))
    print(f"\nstraight-through trained:   mean {trained_summary['mean']:.3f}  "
          f"min {trained_summary['min']:.3f}  max {trained_summary['max']:.3f}  "
          f"sd {trained_summary['stdev']:.3f}")
    print(f"straight-through untrained: mean {untrained_summary['mean']:.3f}")

    payload = {
        "config": {
            "keys": KEYS, "state_width": WIDTH,
            "train_tasks": TRAIN_TASKS, "eval_tasks": EVAL_TASKS,
            "steps": STEPS, "lr": LR, "seeds": list(SEEDS),
            "sharp_temperature": SHARP_TEMPERATURE,
        },
        "conditions": {
            "hand_set_gate": {
                "source": "the gate the selective family ships, from agent.py",
                "rate": round(hand_rate, 6), "tasks": tasks,
            },
            "soft_gate_raw": {
                "source": "the same training loop, evaluated at temperature 1.0",
                "rate": round(raw_rate, 6), "tasks": tasks,
            },
            "soft_gate_sharpened": {
                "source": f"the same training loop, evaluated at temperature "
                          f"{SHARP_TEMPERATURE:g}",
                "rate": round(sharp_rate, 6), "tasks": tasks,
            },
            "straight_through_trained": {
                "source": "hard 0/1 forward, sigmoid gradient passed through, "
                          "trained",
                "spread": trained_summary,
            },
            "straight_through_untrained": {
                "source": "the same hard forward pass, at initialisation",
                "spread": untrained_summary,
            },
        },
        "seeds": per_seed,
        "wall_seconds": round(time.time() - started, 1),
    }

    # The control is why this run is publishable, so a payload without it is not
    # written at all. A straight-through number whose harness never reproduced
    # 0.420 / 1.000 would be a number about the harness.
    missing = [key for key in ("hand_set_gate", "soft_gate_raw", "soft_gate_sharpened")
               if key not in payload["conditions"]]
    if missing:
        print(f"refusing to write {args.out}: {', '.join(missing)} missing from "
              f"the payload", file=sys.stderr)
        return 1

    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
