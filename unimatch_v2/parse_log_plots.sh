#!/bin/bash
# Loop through all subdirectories under a given experiment folder and run
# parse_log_plots.py on every log file whose name starts with out_unimatch_v2.
#
# Usage:
#   bash parse_log_plots.sh exp/dinov3_base_sbatch_run1

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment_folder>"
    echo "Example: $0 exp/dinov3_base_sbatch_run1"
    exit 1
fi

EXP_FOLDER="$1"

if [[ ! -d "$EXP_FOLDER" ]]; then
    echo "Error: folder not found: $EXP_FOLDER"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

find "$EXP_FOLDER" -mindepth 1 -maxdepth 1 -type d | sort | while read -r subdir; do
    for log_file in "$subdir"/out_unimatch_v2*.log; do
        [[ -f "$log_file" ]] || continue
        echo "Processing: $log_file"
        python "$SCRIPT_DIR/parse_log_plots.py" --log-file "$log_file"
    done
done
