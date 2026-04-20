#!/bin/bash

# Ensure this script always runs under bash, not sh/dash (which lacks array support)
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# Training script: runs all experiments in sequence.
# Each entry is: "<model> <method>"
# Usage: bash scripts/train.sh [NUM_GPUS [PORT [MASTER_ADDR]]]
#   NUM_GPUS     number of GPUs per node (default: 1)
#   PORT         master port for distributed training (default: 9271)
#   MASTER_ADDR  master address for distributed training (default: localhost)
# Example usage:
#   bash scripts/train.sh

# modify these arguments if you want to try other splits or methods
# method: ['unimatch_v2', 'supervised', 'test_model']
# exp: just for specifying the 'save_path'
# model: ['m-zeroshot', 'm-citizen', ...]. Please check directory './splits' for available model splits

NUM_GPUS=${1:-1}
PORT=${2:-9271}
MASTER_ADDR=${3:-"localhost"}

exp='dinov2_base_dcp'
unlabeled_sample_size=1500
unlabeled_sample_seed=23838742

MODELS=(
    "m-zeroshot"
    "m-zeroshot"
    "m-citizen"
    "m-citizen"
    "m-expert"
    "m-expert"
    "m-mix-20"
    "m-mix-20"
)

METHODS=(
    "supervised"
    "test_model"
    "unimatch_v2"
    "test_model"
    "unimatch_v2"
    "test_model"
    "unimatch_v2"
    "test_model"
)

# Move to the unimatch_v2 root so relative paths (splits/, exp/) resolve correctly
cd "$(dirname "$0")/.." || exit 1

RUN_TS=$(date +%Y%m%d_%H%M%S)

for i in $(seq 0 $((${#MODELS[@]} - 1))); do
    model="${MODELS[$i]}"
    method="${METHODS[$i]}"

    training_config="splits/${model}.yaml"
    save_path="exp/${exp}/${model}"

    mkdir -p "$save_path"

    echo "========================================================"
    echo "Running: model=${model}  method=${method}  gpus=${NUM_GPUS}"
    echo "  training_config=${training_config}"
    echo "  save_path=${save_path}"
    echo "========================================================"

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$PORT" \
            "${method}.py" \
            --training-config "$training_config" \
            --save-path "$save_path" \
            --port "$PORT" \
            --unlabeled-sample-size "$unlabeled_sample_size" \
            --unlabeled-sample-seed "$unlabeled_sample_seed" \
            2>&1 | tee "${save_path}/out_${RUN_TS}.log"
    else
        python "${method}.py" \
            --training-config "$training_config" \
            --save-path "$save_path" \
            --port "$PORT" \
            --unlabeled-sample-size "$unlabeled_sample_size" \
            --unlabeled-sample-seed "$unlabeled_sample_seed" \
            2>&1 | tee "${save_path}/out_${RUN_TS}.log"
    fi

    exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        echo "ERROR: ${model}/${method} failed with exit code ${exit_code}. Stopping pipeline."
        exit "$exit_code"
    fi

    echo "Done: ${model}/${method}"
    echo ""
done

echo "All experiments completed successfully."
