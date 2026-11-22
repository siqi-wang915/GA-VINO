from __future__ import annotations

import math


class RelativeEarlyStopping:
    """Track significant relative improvements for early stopping."""

    def __init__(
        self,
        *,
        enabled: bool,
        min_epochs: int,
        patience: int,
        min_delta: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        if self.min_epochs < 0:
            raise ValueError("min_epochs must be non-negative")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")

        self.reference_best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Update the counter and return whether training should stop."""
        value = float(value)
        epoch = int(epoch)

        if not math.isfinite(self.reference_best):
            significant_improvement = math.isfinite(value)
        else:
            threshold = self.min_delta * max(abs(self.reference_best), 1.0e-12)
            significant_improvement = value < self.reference_best - threshold

        if significant_improvement:
            self.reference_best = value
            self.bad_epochs = 0
        elif epoch >= self.min_epochs:
            self.bad_epochs += 1

        return (
            self.enabled
            and epoch >= self.min_epochs
            and self.bad_epochs >= self.patience
        )
