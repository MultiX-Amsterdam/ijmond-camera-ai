"""Parse a unimatch_v2 training log file and generate training-curve plots.

Three PNG files are saved in the same directory as the log file:
  - loss_curve.png        : train loss, val loss, robust val loss vs epoch
  - angle_score_curve.png : angle score vs epoch
  - gf1_curve.png         : gF1 (normal) and gF1 (robust) vs epoch

Example usage:
    python parse_log_plots.py \
        --log-file exp/dinov3_base_sbatch_run1/m-citizen/out_unimatch_v2_20260529_035230.log
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

RE_MODEL_NAME = re.compile(r"Training config \(splits/(.+?)\.yaml\)")
RE_TRAIN_LOSS = re.compile(r"\*\*\*\*\* Epoch (\d+) \*\*\*\*\* >>>> Train Loss: ([\d.]+)")
RE_VAL_LOSS = re.compile(r"\*\*\*\*\* Evaluation \*\*\*\*\* >>>> Val Loss: ([\d.]+)")
RE_ROBUST_VAL_LOSS = re.compile(r"\*\*\*\*\* Robust Evaluation \*\*\*\*\* >>>> Val Loss: ([\d.]+)")
RE_ANGLE_SCORE = re.compile(r"\*\*\*\*\* Convergence Check \*\*\*\*\* >>>> Angle Score: ([\d.]+)")
RE_VAL_GF1 = re.compile(r"\*\*\*\*\* Evaluation \*\*\*\*\* >>>> gIoU: [\d.]+, gF1: ([\d.]+)")
RE_ROBUST_GF1 = re.compile(r"\*\*\*\*\* Robust Evaluation \*\*\*\*\* >>>> gIoU: [\d.]+, gF1: ([\d.]+)")


def parse_log(log_path):
    """Parse the training log and return structured data.

    Parameters
    ----------
    log_path : str
        Absolute or relative path to the log file.

    Returns
    -------
    model_name : str
        Model name extracted from the ``Training config`` line.
    epochs : list of int
    train_losses : list of float
    val_losses : list of float
    robust_val_losses : list of float
    angle_scores : list of float
    val_gf1s : list of float
    robust_gf1s : list of float
    """
    model_name = "unknown"
    epochs = []
    train_losses = []
    val_losses = []
    robust_val_losses = []
    angle_scores = []
    val_gf1s = []
    robust_gf1s = []

    # Pending values collected per epoch block before we know they belong
    # to the same epoch.  The log emits them in this order per epoch:
    #   1. Train Loss
    #   2. Val Loss  (immediately after)
    #   3. Val gF1   (immediately after)
    #   4. Robust Val Loss
    #   5. Robust gF1
    #   6. Angle Score
    pending_epoch = None
    pending_train_loss = None
    pending_val_loss = None
    pending_robust_val_loss = None
    pending_val_gf1 = None
    pending_robust_gf1 = None
    pending_angle_score = None

    def flush_epoch():
        """Commit the pending epoch data to the output lists."""
        if pending_epoch is None:
            return
        if pending_train_loss is None:
            return
        epochs.append(pending_epoch)
        train_losses.append(pending_train_loss)
        val_losses.append(pending_val_loss)
        robust_val_losses.append(pending_robust_val_loss)
        val_gf1s.append(pending_val_gf1)
        robust_gf1s.append(pending_robust_gf1)
        angle_scores.append(pending_angle_score)

    with open(log_path, "r") as fh:
        for line in fh:
            # Model name
            m = RE_MODEL_NAME.search(line)
            if m:
                model_name = m.group(1)
                continue

            # New epoch – flush any previous pending epoch first
            m = RE_TRAIN_LOSS.search(line)
            if m:
                flush_epoch()
                pending_epoch = int(m.group(1))
                pending_train_loss = float(m.group(2))
                pending_val_loss = None
                pending_robust_val_loss = None
                pending_val_gf1 = None
                pending_robust_gf1 = None
                pending_angle_score = None
                continue

            m = RE_VAL_LOSS.search(line)
            if m:
                pending_val_loss = float(m.group(1))
                continue

            m = RE_ROBUST_VAL_LOSS.search(line)
            if m:
                pending_robust_val_loss = float(m.group(1))
                continue

            m = RE_VAL_GF1.search(line)
            if m:
                pending_val_gf1 = float(m.group(1))
                continue

            m = RE_ROBUST_GF1.search(line)
            if m:
                pending_robust_gf1 = float(m.group(1))
                continue

            m = RE_ANGLE_SCORE.search(line)
            if m:
                pending_angle_score = float(m.group(1))
                continue

    # Flush the last epoch
    flush_epoch()

    return (
        model_name,
        epochs,
        train_losses,
        val_losses,
        robust_val_losses,
        angle_scores,
        val_gf1s,
        robust_gf1s,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_Y_TICKS = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0


def _apply_common_axes(ax, title):
    """Apply shared axis formatting to *ax*."""
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_yticks(_Y_TICKS)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_loss_curve(log_dir, model_name, epochs, train_losses, val_losses, robust_val_losses):
    """Save a training / validation loss line chart.

    Parameters
    ----------
    log_dir : str
        Directory where the PNG is saved.
    model_name : str
        Used as the plot title.
    epochs : list of int
    train_losses : list of float
    val_losses : list of float
    robust_val_losses : list of float
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, color="blue", label="Train Loss")
    ax.plot(epochs, val_losses, color="orange", label="Val Loss")
    ax.plot(epochs, robust_val_losses, color="red", label="Robust Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    _apply_common_axes(ax, model_name)
    plt.tight_layout()
    out_path = os.path.join(log_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved %s" % out_path)


def plot_angle_score_curve(log_dir, model_name, epochs, angle_scores):
    """Save an angle score line chart.

    Parameters
    ----------
    log_dir : str
        Directory where the PNG is saved.
    model_name : str
        Used as the plot title.
    epochs : list of int
    angle_scores : list of float or None
        ``None`` entries (missing angle scores) are skipped.
    """
    valid_epochs = [e for e, s in zip(epochs, angle_scores) if s is not None]
    valid_scores = [s for s in angle_scores if s is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(valid_epochs, valid_scores, color="green", label="Angle Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Angle Score")
    _apply_common_axes(ax, model_name)
    plt.tight_layout()
    out_path = os.path.join(log_dir, "angle_score_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved %s" % out_path)


def plot_gf1_curve(log_dir, model_name, epochs, val_gf1s, robust_gf1s):
    """Save a gF1 (normal and robust) line chart.

    Parameters
    ----------
    log_dir : str
        Directory where the PNG is saved.
    model_name : str
        Used as the plot title.
    epochs : list of int
    val_gf1s : list of float or None
    robust_gf1s : list of float or None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_gf1s, color="orange", label="Val gF1")
    ax.plot(epochs, robust_gf1s, color="red", label="Robust Val gF1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("gF1")
    _apply_common_axes(ax, model_name)
    plt.tight_layout()
    out_path = os.path.join(log_dir, "gf1_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved %s" % out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse arguments and generate all three plots."""
    parser = argparse.ArgumentParser(
        description="Parse a unimatch_v2 training log and generate training-curve plots."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to the training log file (e.g. exp/dinov3_base_sbatch_run1/m-citizen/out_*.log)",
    )
    args = parser.parse_args()

    log_path = args.log_file
    if not os.path.isfile(log_path):
        raise FileNotFoundError("Log file not found: %s" % log_path)

    log_dir = os.path.dirname(os.path.abspath(log_path))

    (
        model_name,
        epochs,
        train_losses,
        val_losses,
        robust_val_losses,
        angle_scores,
        val_gf1s,
        robust_gf1s,
    ) = parse_log(log_path)

    if not epochs:
        print("No epoch data found in %s" % log_path)
        return

    print("Model: %s | Epochs parsed: %d" % (model_name, len(epochs)))

    plot_loss_curve(log_dir, model_name, epochs, train_losses, val_losses, robust_val_losses)
    plot_angle_score_curve(log_dir, model_name, epochs, angle_scores)
    plot_gf1_curve(log_dir, model_name, epochs, val_gf1s, robust_gf1s)


if __name__ == "__main__":
    main()
