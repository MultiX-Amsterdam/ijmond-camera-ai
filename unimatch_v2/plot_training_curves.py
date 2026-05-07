"""Plot training and validation loss curves from a saved loss_history.json.

Reads the JSON written during training and saves a ``loss_curve.png`` in the
same directory.  Can be run after training finishes or at any point during
training to visualise progress so far.

Example usage:
    python plot_training_curves.py --save-path exp/dinov3_small_testrun/m-expert
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(save_path):
    """Read loss_history.json and save a training/validation loss line chart.

    Training loss is plotted in blue and validation loss in orange.  The chart
    is saved as ``loss_curve.png`` inside *save_path*.  Does nothing if the
    history file does not exist or is empty.

    Parameters
    ----------
    save_path : str
        Directory containing ``loss_history.json``.
    """
    history_path = os.path.join(save_path, "loss_history.json")
    if not os.path.exists(history_path):
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history:
        return

    epochs = [e["epoch"] for e in history]
    train_losses = [e["train_loss"] for e in history]
    val_losses = [e["val_loss"] for e in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, color="blue", label="Training Loss")
    ax.plot(epochs, val_losses, color="orange", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves.")
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory containing loss_history.json (e.g. exp/my_exp/m-expert)",
    )
    args = parser.parse_args()
    plot_loss_curve(args.save_path)
    print("Saved loss_curve.png to %s" % args.save_path)
