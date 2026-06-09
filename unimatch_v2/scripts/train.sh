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

# You can also train the model using interactive srun sessions.
# Before running, allocate a node with srun:
#   srun -u --pty --nodelist=ivi-cn002,ivi-cn004,ivi-cn010,ivi-cn014 --gres=gpu:2 --mem=160G --cpus-per-task=32 --time=2:00:00 -D $(pwd) bash -i
# Then run:
#   bash scripts/train.sh 2

# modify these arguments if you want to try other splits or methods
# method: ['unimatch_v2', 'supervised', 'test_model']
# exp: just for specifying the 'save_path'
# model: ['m-zeroshot', 'm-citizen', ...]. Please check directory './splits' for available model splits

NUM_GPUS=${1:-1}
PORT=${2:-9271}
MASTER_ADDR=${3:-"localhost"}

exp='dinov3_small_testrun'

MODELS=(
    # "m-zeroshot"
    # "m-zeroshot"
    "m-citizen-box-run-1"
    "m-citizen-box-run-1"
    # "m-mix-100-awl-run-1"
    # "m-mix-100-awl-run-1"
    # "m-citizen-run-1"
    # "m-citizen-run-1"
    # "m-expert-run-1"
    # "m-expert-run-1"
    # "m-mix-100-run-1"
    # "m-mix-100-run-1"
    # "m-mix-100-box-run-1"
    # "m-mix-100-box-run-1"
)

METHODS=(
    # "unimatch_v2"
    # "test_model"
    "unimatch_v2"
    "test_model"
    # "unimatch_v2"
    # "test_model"
    # "unimatch_v2"
    # "test_model"
    # "unimatch_v2"
    # "test_model"
    # "unimatch_v2"
    # "test_model"
    # "unimatch_v2"
    # "test_model"
)

# Move to the unimatch_v2 root so relative paths (splits/, exp/) resolve correctly
cd "$(dirname "$0")/.." || exit 1

RUN_TS=$(date +%Y%m%d_%H%M%S)

if [ "${#MODELS[@]}" -ne "${#METHODS[@]}" ]; then
    echo "ERROR: MODELS (${#MODELS[@]}) and METHODS (${#METHODS[@]}) arrays must have the same length."
    exit 1
fi

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

    ITER_TS=$(date +%Y%m%d_%H%M%S)

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$PORT" \
            "${method}.py" \
            --training-config "$training_config" \
            --save-path "$save_path" \
            --port "$PORT" \
            2>&1 | tee "${save_path}/out_${method}_${ITER_TS}.log"
    else
        python "${method}.py" \
            --training-config "$training_config" \
            --save-path "$save_path" \
            --port "$PORT" \
            2>&1 | tee "${save_path}/out_${method}_${ITER_TS}.log"
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
