"""Minimal: does a straight-through hard gate learn the hardness?

The trained sigmoid gate learns the ADDRESSING exactly (its 0.5-threshold is the
hand-set gate) but not the HARDNESS: raw it scores 0.420, and reaching 1.000 needs
a temperature chosen at evaluation. This replaces the soft forward pass with a
hard one, passing the sigmoid gradient straight through, so no temperature is
needed at inference.
"""
from __future__ import annotations
import sys, time
import torch

from beyond_attention.agent import selective_suite, SELECTIVE_KEYS
from beyond_attention.learned_gate import (
    LearnedGate, HandSetGate, run_learned, gate_batch, stack_batch, state_loss,
)
from learned_gate import solve_rate, TRAIN_TASKS, EVAL_TASKS, SEEDS, STEPS, LR

KEYS = WIDTH = SELECTIVE_KEYS


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
    return float(loss)


from beyond_attention.learned_gate import at_temperature

train_tasks = selective_suite(TRAIN_TASKS, seed=0, n_keys=KEYS)
eval_tasks = selective_suite(EVAL_TASKS, seed=1, n_keys=KEYS)

hand = HandSetGate(KEYS, WIDTH)
print(f"hand-set gate                     {solve_rate(eval_tasks, hand, KEYS, WIDTH)[0]:.3f}")

# --- HARNESS CONTROL -------------------------------------------------------
# The same training loop must reproduce the known soft-gate numbers. If it does
# not, the straight-through row below is measuring the harness, not the idea.
_plain = LearnedGate(KEYS, WIDTH, seed=0)
train(_plain, train_tasks)
print(f"soft gate, same loop, raw         "
      f"{solve_rate(eval_tasks, at_temperature(_plain, 1.0), KEYS, WIDTH)[0]:.3f}   (experiment says 0.420)")
print(f"soft gate, same loop, sharpened   "
      f"{solve_rate(eval_tasks, at_temperature(_plain, 0.05), KEYS, WIDTH)[0]:.3f}   (experiment says 1.000)")

trained, untrained = [], []
for seed in SEEDS:
    g = StraightThroughGate(KEYS, WIDTH, seed=seed)
    u = StraightThroughGate(KEYS, WIDTH, seed=seed)
    final = train(g, train_tasks)
    t = solve_rate(eval_tasks, as_gate(g), KEYS, WIDTH)[0]
    n = solve_rate(eval_tasks, as_gate(u), KEYS, WIDTH)[0]
    trained.append(t); untrained.append(n)
    print(f"  seed {seed}: straight-through trained {t:.3f}   untrained {n:.3f}   final loss {final:.2f}")

import statistics
print(f"\nstraight-through trained:   mean {statistics.mean(trained):.3f}  "
      f"min {min(trained):.3f}  max {max(trained):.3f}  sd {statistics.pstdev(trained):.3f}")
print(f"straight-through untrained: mean {statistics.mean(untrained):.3f}")
