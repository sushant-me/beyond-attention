"""Training and evaluation, identical for both architectures.

One function, one set of hyperparameters, one seed, two models. If the two
architectures got different learning rates or different step counts, whatever
the experiment reported would be a fact about the tuning rather than about the
architecture.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import nn

from .tasks import mqar_batch, loss_and_accuracy


@dataclass
class Result:
    name: str
    parameters: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    steps: int
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def as_row(self) -> str:
        return (
            f"{self.name:<12} {self.parameters:>9,} "
            f"{self.train_accuracy:>10.3f} {self.test_accuracy:>11.3f} "
            f"{self.final_loss:>11.3f}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train(
    model: nn.Module,
    name: str,
    n_pairs: int,
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    n_train_queries: int = 1,
    seed: int = 0,
    device: str = "cpu",
    eval_pairs: tuple[int, ...] = (),
    eval_batch_size: int = 256,
    log_every: int = 0,
    progress=None,
) -> Result:
    """Train one model on MQAR and measure it on the same task at several sizes.

    `eval_pairs` is evaluated with the weights frozen and a fresh sample every
    time, so it measures the task rather than the batch.
    """
    set_seed(seed)
    model.to(device)
    model.train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, total_steps=steps, pct_start=0.1
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    history: list[tuple[int, float, float]] = []
    final_loss = float("nan")
    for step in range(steps):
        batch = mqar_batch(
            batch_size, n_pairs, n_train_queries, generator, device
        )
        loss, acc = loss_and_accuracy(model, batch)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        final_loss = loss.item()
        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append((step, final_loss, acc))
            if progress is not None:
                progress(step, final_loss, acc)

    train_accuracy = _evaluate(
        model, n_pairs, n_train_queries, eval_batch_size, generator, device
    )
    test_accuracy = (
        max(
            _evaluate(
                model, pairs, n_train_queries, eval_batch_size, generator, device
            )
            for pairs in eval_pairs
        )
        if eval_pairs
        else train_accuracy
    )
    return Result(
        name=name,
        parameters=_count(model),
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        final_loss=final_loss,
        steps=steps,
        history=history,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    n_pairs: int,
    n_queries: int,
    batch_size: int,
    generator: torch.Generator,
    device: str,
) -> float:
    """Exact-match accuracy on freshly sampled batches of the given size."""
    from .tasks import accuracy

    model.eval()
    batches = max(1, 2048 // batch_size)
    total = 0.0
    for _ in range(batches):
        batch = mqar_batch(batch_size, n_pairs, n_queries, generator, device)
        total += accuracy(model(batch.tokens), batch)
    model.train()
    return total / batches


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
