#!/bin/bash

# Training script for interactive srun sessions: runs all experiments in sequence.
# Usage: bash scripts/srun_train.sh [NUM_GPUS [PORT [MASTER_ADDR]]]
#
# Before running, allocate a node with srun:
#   srun -u --pty --nodelist=ivi-cn002 --gres=gpu:2 --mem=120G --cpus-per-task=24 --time=2:00:00 -D $(pwd) bash -i
# Then run:
#   bash scripts/srun_train.sh 2

NUM_GPUS=${1:-1}
PORT=${2:-9271}
MASTER_ADDR=${3:-"localhost"}

exp='dinov2_base_srun_testrun'
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
    "m-mix-40"
    "m-mix-40"
    "m-mix-60"
    "m-mix-60"
    "m-mix-80"
    "m-mix-80"
    "m-mix-100"
    "m-mix-100"
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
    "unimatch_v2"
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

for i in "${!MODELS[@]}"; do
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
        conda run --no-capture-output -n ijmond-camera-ai torchrun \
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
        conda run --no-capture-output -n ijmond-camera-ai python "${method}.py" \
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