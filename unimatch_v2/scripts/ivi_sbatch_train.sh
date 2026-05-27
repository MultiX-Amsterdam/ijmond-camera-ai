#!/bin/bash
# =============================================================================
# Train a semantic segmentation model on a Slurm cluster.
#
# USAGE:
#   Submit from the unimatch_v2/ directory:
#     sbatch scripts/sbatch_train.sh
#   List submitted jobs:
#     squeue -u $USER
#   Detailed info about a specific job:
#     scontrol show job <job_id>
#   Cancel a job:
#     scancel <job_id>
#   Check node status:
#     sinfo -N -o "%20N %10c %10m %25G %10T %P"
#     sinfo -n ivi-cn032 -o "%N %C %e %G"
#     scontrol show node ivi-cn032
#
# BEFORE SUBMITTING:
#   1. Set NUM_GPUS below and update --gres=gpu:N in the header accordingly.
#      NUM_GPUS needs to match the number of GPUs allocated by Slurm for this job.
#   2. Adjust --time to the maximum wall-clock time for the full pipeline.
#      (resources are released automatically as soon as the script finishes).
#   3. Update the environment setup section to match your cluster's modules
#      and conda environment name.
#   4. Optionally change exp and other training parameters.
#
# AVAILABLE CONFIGS:
#   Dataset configs : configs/  (e.g., ijmond.yaml, smoke5k.yaml)
#   Model splits    : splits/   (e.g., m-zeroshot.yaml, m-expert.yaml)
# =============================================================================

#SBATCH --job-name=sbatch_ijmond_ai_train
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --nodelist=ivi-cn002,ivi-cn004,ivi-cn010,ivi-cn014
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=160G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=all6000users
#SBATCH --partition=all

# ---- Training parameters (modify as needed) ---------------------------------
exp="dinov3_base_sbatch_testrun"

# Number of GPUs per node must match --gres=gpu:N in the SBATCH header above
NUM_GPUS=2
PORT=9271

MODELS=(
    "m-zeroshot"
    "m-zeroshot"
    "m-citizen"
    "m-citizen"
    "m-expert"
    "m-expert"
    "m-mix-100-box"
    "m-mix-100-box"
    "m-mix-100-awl"
    "m-mix-100-awl"
    "m-mix-100"
    "m-mix-100"
    "m-expert-25"
    "m-expert-25"
    "m-mix-25-box"
    "m-mix-25-box"
    "m-mix-25-awl"
    "m-mix-25-awl"
    "m-mix-25"
    "m-mix-25"
    "m-expert-50"
    "m-expert-50"
    "m-mix-50-box"
    "m-mix-50-box"
    "m-mix-50-awl"
    "m-mix-50-awl"
    "m-mix-50"
    "m-mix-50"
)

METHODS=(
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

# ---- Environment setup ------------------------------------------------------
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ijmond-camera-ai

# ---- Redirect top-level log -------------------------------------------------
mkdir -p "exp/${exp}"
exec > >(tee "exp/${exp}/sbatch_${SLURM_JOB_ID}.out") 2>&1

# ---- Launch all experiments -------------------------------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)

if [ "${#MODELS[@]}" -ne "${#METHODS[@]}" ]; then
    echo "ERROR: MODELS (${#MODELS[@]}) and METHODS (${#METHODS[@]}) arrays must have the same length."
    exit 1
fi

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