"""Plot the MiniGPT train/validation loss history."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Support both package imports and direct script execution.
    from .config import TrainConfig
except ImportError:  # pragma: no cover - direct `python plot.py` fallback
    from config import TrainConfig


def main() -> None:
    cfg = TrainConfig()
    log_path = cfg.results_dir / cfg.train_log_name
    output_path = cfg.results_dir / cfg.loss_curve_name

    if not log_path.is_file():
        raise FileNotFoundError(
            f"training log not found: {log_path}; run train.py first"
        )

    records = json.loads(log_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"training log is empty: {log_path}")

    steps = [record["step"] for record in records]
    train_losses = [record["train_loss"] for record in records]
    validation_losses = [record["validation_loss"] for record in records]

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, train_losses, label="Train loss", linewidth=1.8)
    axis.plot(steps, validation_losses, label="Validation loss", linewidth=1.8)
    axis.set(title="MiniGPT training loss", xlabel="Training step", ylabel="Cross-entropy loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"saved loss curve to {output_path}")


if __name__ == "__main__":
    main()
