"""Parse test_model evaluation results from a training log file and produce a
LaTeX table saved as a .txt file next to the log file.

Run this script from inside the unimatch_v2/ folder:

    cd unimatch_v2/
    python parse_test_results.py exp/sbatch_164011.out

Or with any other log file path:

    python parse_test_results.py <log_file_path>

Output: a pandas table printed to stdout and a LaTeX table saved to
<log_file_path>_latex_table.txt.
"""
import os
import re
import sys

import pandas as pd


BANNER_RE = re.compile(
    r"^Running: model=(\S+)\s+method=test_model\s",
)
EVAL_HEADER_RE = re.compile(
    r"\*\*\*\*\* Evaluation \*\*\*\*\*$"
)
METRIC_RE = re.compile(
    r"\[.*?\]\s+(\w+): ([\d.]+)$"
)
BAND_FILTERED_RE = re.compile(
    r"\*\*\*\*\* Evaluation \(band-filtered\) \*\*\*\*\*"
)


def parse_log(log_path):
    """Parse a training log file and extract test_model evaluation metrics.

    Parameters
    ----------
    log_path : str
        Path to the log file.

    Returns
    -------
    pd.DataFrame
        Table with one row per model and columns for each metric.
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    records = []
    i = 0
    while i < len(lines):
        banner_match = BANNER_RE.search(lines[i])
        if banner_match:
            model_name = banner_match.group(1)
            # Scan forward to find "***** Evaluation *****" (not band-filtered)
            j = i + 1
            while j < len(lines):
                # Check for a new banner (next model section)
                if BANNER_RE.search(lines[j]):
                    break
                if EVAL_HEADER_RE.search(lines[j]) and not BAND_FILTERED_RE.search(lines[j]):
                    # Collect metric lines immediately following
                    metrics = {}
                    k = j + 1
                    while k < len(lines):
                        stripped = lines[k].strip()
                        if not stripped:
                            break
                        m = METRIC_RE.search(stripped)
                        if m:
                            metrics[m.group(1)] = float(m.group(2))
                        else:
                            break
                        k += 1
                    if metrics:
                        records.append({"model": model_name, **metrics})
                    break
                j += 1
            i = j
        else:
            i += 1

    df = pd.DataFrame(records)
    return df


# Row definitions: (model_key, display_name, citizen_col, expert_col)
# Groups are separated by None (triggers a \midrule).
TABLE_ROWS = [
    ("m-zeroshot",     "Zeroshot",           "0\\% (n/a)",   "0\\%"),
    ("m-citizen",      "Citizen",            "100\\% (hard with mask)", "0\\%"),
    None,
    ("m-expert-25",    "Expert-25",          "0\\% (n/a)",   "25\\%"),
    ("m-mix-25-box",   "Mix-25-soft-box",    "100\\% (soft with box)", "25\\%"),
    ("m-mix-25",       "Mix-25-hard-mask",   "100\\% (hard with mask)", "25\\%"),
    ("m-mix-25-awl",   "Mix-25-hard-box",    "100\\% (hard with box)", "25\\%"),
    None,
    ("m-expert-50",    "Expert-50",          "0\\% (n/a)",   "50\\%"),
    ("m-mix-50-box",   "Mix-50-soft-box",    "100\\% (soft with box)", "50\\%"),
    ("m-mix-50",       "Mix-50-hard-mask",   "100\\% (hard with mask)", "50\\%"),
    ("m-mix-50-awl",   "Mix-50-hard-box",    "100\\% (hard with box)", "50\\%"),
    None,
    ("m-expert",       "Expert-100",         "0\\% (n/a)",   "100\\%"),
    ("m-mix-100-box",  "Mix-100-soft-box",   "100\\% (soft with box)", "100\\%"),
    ("m-mix-100",      "Mix-100-hard-mask",  "100\\% (hard with mask)", "100\\%"),
    ("m-mix-100-awl",  "Mix-100-hard-box",   "100\\% (hard with box)", "100\\%"),
]

METRIC_COLS = ["gIoU", "gF1", "gPre", "gRec", "mIoU", "mF1", "FAR"]


def fmt(value):
    """Format a float as a 2-decimal string without a leading zero (e.g. .45).

    Parameters
    ----------
    value : float or None
        The value to format.

    Returns
    -------
    str
        Formatted string, or '—' if the value is missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    return f"{value:.2f}".lstrip("0") or ".00"


def generate_latex_table(df):
    """Generate a LaTeX table string from the parsed metrics DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of parse_log(), indexed by model name.

    Returns
    -------
    str
        Full LaTeX table as a string.
    """
    metrics_by_model = df.set_index("model").to_dict(orient="index")

    lines = [
        r"\begin{table}[!t]",
        r"    \centering",
        r"    \caption{Caption here}",
        r"    \label{tab:experiment_results_2}",
        r"    \begin{tabular}{lccc|ccccccc}",
        r"        \toprule",
        (
            r"        \textbf{Model} & \textbf{Citizen (Role)} & \textbf{Expert}"
            r" & \textbf{gIoU} & \textbf{gF1} & \textbf{gPre} & \textbf{gRec}"
            r" & \textbf{mpIoU} & \textbf{mpF1} & \textbf{FAR} \\"
        ),
        r"        \midrule",
    ]

    for entry in TABLE_ROWS:
        if entry is None:
            lines.append(r"        \midrule")
            continue

        model_key, display_name, citizen_col, expert_col = entry
        row_metrics = metrics_by_model.get(model_key, {})

        metric_values = " & ".join(
            fmt(row_metrics.get(col)) for col in METRIC_COLS
        )
        lines.append(
            f"        {display_name} & {citizen_col} & {expert_col}"
            f" & {metric_values} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_test_results.py <log_file_path>")
        sys.exit(1)

    log_path = sys.argv[1]
    df = parse_log(log_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    latex = generate_latex_table(df)
    out_path = os.path.splitext(log_path)[0] + "_latex_table.txt"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {out_path}")


if __name__ == "__main__":
    main()
