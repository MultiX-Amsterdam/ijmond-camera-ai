"""Parse test_model evaluation results from an experiment folder and produce a
LaTeX table saved as a .txt file next to the experiment folder.

Each experiment folder contains model subfolders. If a subfolder name ends with
"-run-N" (e.g. "m-expert-run-1", "m-expert-run-2"), it is treated as one of
multiple training runs for the same model. Metrics are averaged over all runs
and the standard deviation (sample, ddof=1) is reported.

Run this script from inside the unimatch_v2/ folder:

    cd unimatch_v2/
    python parse_test_results.py exp/dinov3_base_sbatch

Or with any other experiment folder path:

    python parse_test_results.py <exp_folder_path>

Output: a pandas table printed to stdout and a LaTeX table saved to
<exp_folder_path>_latex_table.txt.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


# Matches the tab-indented metric lines in log files, e.g.:
#   [2026-05-30 05:29:03,801][    INFO] \tgIoU: 0.4636
METRIC_RE = re.compile(r"\t(\w+):\s*([\d.]+)\s*$")

EVAL_HEADER_RE = re.compile(r"\*\*\*\*\* Evaluation \*\*\*\*\*")
BAND_FILTERED_RE = re.compile(r"\*\*\*\*\* Evaluation \(band-filtered\) \*\*\*\*\*")
ROBUST_HEADER_RE = re.compile(r"\*\*\*\*\* Robust Evaluation \*\*\*\*\*")

# Suffix pattern to strip from folder names to get the base model name.
RUN_SUFFIX_RE = re.compile(r"-run-\d+$")

LATEX_METRICS = ["gIoU", "gF1", "FAR"]


def scan_exp_folder(exp_folder):
    """Scan an experiment folder and group model run folders by base model name.

    Parameters
    ----------
    exp_folder : str
        Path to the experiment folder containing model subfolders.

    Returns
    -------
    dict
        Mapping from base model name (str) to a list of subfolder paths (str).
        Single-run folders (no "-run-N" suffix) appear as a one-element list.
    """
    groups = {}
    for entry in sorted(os.listdir(exp_folder)):
        full_path = os.path.join(exp_folder, entry)
        if not os.path.isdir(full_path):
            continue
        base_name = RUN_SUFFIX_RE.sub("", entry)
        groups.setdefault(base_name, []).append(full_path)
    return groups


def _collect_metrics_after(lines, start_idx, wanted):
    """Collect metric values from log lines starting just after a section header.

    Stops at the first blank line or a line that does not match METRIC_RE.

    Parameters
    ----------
    lines : list of str
        All lines from the log file.
    start_idx : int
        Index of the first line to examine (i.e. the line after the header).
    wanted : set of str
        Set of metric names to collect (e.g. {"gIoU", "gF1", "FAR"}).

    Returns
    -------
    dict
        Mapping from metric name (str) to float value.
    """
    metrics = {}
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        m = METRIC_RE.search(line)
        if m:
            name, value = m.group(1), float(m.group(2))
            if name in wanted:
                metrics[name] = value
        else:
            break
    return metrics


def parse_log_file(log_path):
    """Parse a single out_test_model log file for Evaluation and Robust Evaluation metrics.

    Parameters
    ----------
    log_path : str
        Path to the log file.

    Returns
    -------
    dict
        Dictionary with keys "eval" and "robust", each mapping metric names to
        float values (gIoU, gF1, FAR). Returns empty dicts for missing sections.
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    wanted = set(LATEX_METRICS)
    result = {"eval": {}, "robust": {}}

    for i, line in enumerate(lines):
        if ROBUST_HEADER_RE.search(line):
            result["robust"] = _collect_metrics_after(lines, i + 1, wanted)
        elif EVAL_HEADER_RE.search(line) and not BAND_FILTERED_RE.search(line):
            result["eval"] = _collect_metrics_after(lines, i + 1, wanted)

    return result


def aggregate_runs(folder_paths):
    """Parse all run folders for a model and aggregate metrics across runs.

    Parameters
    ----------
    folder_paths : list of str
        Paths to the model run subfolders.

    Returns
    -------
    dict
        Keys "eval" and "robust"; each maps metric name to a dict with keys
        "mean" (float) and "std" (float or None if only one run).
    """
    per_run_eval = []
    per_run_robust = []

    for folder in folder_paths:
        pattern = os.path.join(folder, "out_test_model*.log")
        matches = sorted(glob.glob(pattern))
        if not matches:
            continue
        parsed = parse_log_file(matches[0])
        if parsed["eval"]:
            per_run_eval.append(parsed["eval"])
        if parsed["robust"]:
            per_run_robust.append(parsed["robust"])

    def _agg(run_list):
        if not run_list:
            return {}
        all_keys = set(run_list[0].keys())
        agg = {}
        for key in all_keys:
            values = [r[key] for r in run_list if key in r]
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else None
            agg[key] = {"mean": mean, "std": std}
        return agg

    return {"eval": _agg(per_run_eval), "robust": _agg(per_run_robust)}


def build_dataframe(groups):
    """Build a summary DataFrame from grouped and aggregated experiment results.

    Parameters
    ----------
    groups : dict
        Output of scan_exp_folder(): base model name → list of folder paths.

    Returns
    -------
    pd.DataFrame
        One row per model. Columns: model, plus for each section (eval, robust)
        and each metric (gIoU, gF1, FAR): "<section>_<metric>_mean" and
        "<section>_<metric>_std".
    """
    records = []
    for base_name, folders in groups.items():
        is_multi = len(folders) > 1 or RUN_SUFFIX_RE.search(
            os.path.basename(folders[0])
        )
        agg = aggregate_runs(folders)
        row = {"model": base_name}
        for section in ("eval", "robust"):
            for metric in LATEX_METRICS:
                entry = agg[section].get(metric, {})
                row[f"{section}_{metric}_mean"] = entry.get("mean")
                row[f"{section}_{metric}_std"] = entry.get("std") if is_multi else None
        records.append(row)

    return pd.DataFrame(records)


def fmt_mean_std(mean, std):
    """Format a mean and optional standard deviation for display in a LaTeX table.

    If mean is None (model not found), returns "n/a".
    If std is None (single run), returns just the mean formatted without a
    leading zero (e.g. ".46").
    Otherwise returns mean±std (e.g. ".46$\\pm$.03").

    Parameters
    ----------
    mean : float or None
        The mean value.
    std : float or None
        The standard deviation, or None for single-run models.

    Returns
    -------
    str
        Formatted LaTeX string.
    """
    if mean is None or pd.isna(mean):
        return "n/a"

    def _fmt(v):
        return f"{v:.2f}".lstrip("0") or ".00"

    if std is None or pd.isna(std):
        return _fmt(mean)
    return f"{_fmt(mean)}$\\pm${_fmt(std)}"


# Row definitions: (model_key, display_name, citizen_col, expert_col)
# None entries trigger a \midrule separator.
TABLE_ROWS = [
    ("m-zeroshot",    "Pretrained",  "0\\%",    "0\\%"),
    ("m-citizen",     "Citizen",     "100\\%",  "0\\%"),
    None,
    ("m-expert-25",   "Expert-25",   "0\\%",    "25\\%"),
    ("m-mix-25-box",  "Filter-25",   "100\\%",  "25\\%"),
    ("m-mix-25",      "Mix-25",      "100\\%",  "25\\%"),
    ("m-mix-25-awl",  "AWL-25",      "100\\%",  "25\\%"),
    None,
    ("m-expert-50",   "Expert-50",   "0\\%",    "50\\%"),
    ("m-mix-50-box",  "Filter-50",   "100\\%",  "50\\%"),
    ("m-mix-50",      "Mix-50",      "100\\%",  "50\\%"),
    ("m-mix-50-awl",  "AWL-50",      "100\\%",  "50\\%"),
    None,
    ("m-expert",      "Expert-100",  "0\\%",    "100\\%"),
    ("m-mix-100-box", "Filter-100",  "100\\%",  "100\\%"),
    ("m-mix-100",     "Mix-100",     "100\\%",  "100\\%"),
    ("m-mix-100-awl", "AWL-100",     "100\\%",  "100\\%"),
]


def generate_latex_table(df):
    """Generate a LaTeX table string from the summary DataFrame.

    The table has 8 columns: Model, Citizen, Expert, then gIoU/gF1/FAR for
    the standard evaluation, then gIoU/gF1/FAR for the robust evaluation.
    Multi-run models show mean±std; single-run models show the mean only.
    Missing models show "n/a".

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_dataframe().

    Returns
    -------
    str
        Full LaTeX table as a string.
    """
    metrics_by_model = df.set_index("model").to_dict(orient="index")

    lines = [
        r"\begin{table}[!t]",
        r"    \centering",
        r"    \caption{Caption goes here.}",
        r"    \label{tab:experiment_results_2}",
        r"    \begin{tabular}{lcc|ccc|ccc}",
        r"        \toprule",
        (
            r"        \textbf{Model} & \textbf{Citizen} & \textbf{Expert}"
            r" & \textbf{gIoU} & \textbf{gF1} & \textbf{FAR}"
            r" & \textbf{gIoU} & \textbf{gF1} & \textbf{FAR} \\"
        ),
        r"        \midrule",
    ]

    for entry in TABLE_ROWS:
        if entry is None:
            lines.append(r"        \midrule")
            continue

        model_key, display_name, citizen_col, expert_col = entry
        row_data = metrics_by_model.get(model_key, {})

        cells = []
        for section in ("eval", "robust"):
            for metric in LATEX_METRICS:
                mean = row_data.get(f"{section}_{metric}_mean")
                std = row_data.get(f"{section}_{metric}_std")
                cells.append(fmt_mean_std(mean, std))

        lines.append(
            f"        {display_name} & {citizen_col} & {expert_col}"
            f" & {' & '.join(cells)} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main():
    """Main entry point: parse an experiment folder and write a LaTeX table.

    Example usage
    -------------
    cd unimatch_v2/
    python parse_test_results.py exp/dinov3_base_sbatch
    """
    if len(sys.argv) < 2:
        print("Usage: python parse_test_results.py <exp_folder_path>")
        sys.exit(1)

    exp_folder = sys.argv[1]
    groups = scan_exp_folder(exp_folder)
    df = build_dataframe(groups)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string(index=False))

    latex = generate_latex_table(df)
    out_path = exp_folder.rstrip("/") + "_latex_table.txt"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {out_path}")


if __name__ == "__main__":
    main()
