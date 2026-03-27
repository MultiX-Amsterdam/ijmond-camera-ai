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
#
# BEFORE SUBMITTING:
#   1. Set NUM_GPUS below and update --gres=gpu:N in the header accordingly.
#      NUM_GPUS needs to match the number of GPUs allocated by Slurm for this job.
#   2. Adjust --time to the maximum wall-clock time for your run.
#      (resources are released automatically as soon as the script finishes).
#   3. Update the environment setup section to match your cluster's modules
#      and conda environment name.
#   4. Optionally change model, method, exp, and other training parameters.
#
# AVAILABLE CONFIGS:
#   Dataset configs : configs/  (e.g., ijmond.yaml, smoke5k.yaml)
#   Model splits    : splits/   (e.g., m-zeroshot.yaml, m-expert.yaml)
# =============================================================================

#SBATCH --job-name=sbatch_ijmond_ai_train
#SBATCH --ntasks=1
#SBATCH --nodelist=ivi-cn015
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=all

# ---- Training parameters (modify as needed) ---------------------------------
# model : see splits/ for available options
# method: supervised | unimatch_v2 | test_model
model="m-zeroshot"
method="supervised"

exp="dinov2_base_sbatch"
unlabeled_sample_size=1500
unlabeled_sample_seed=23838742

# Number of GPUs per node must match --gres=gpu:N in the SBATCH header above
NUM_GPUS=4
PORT=9271

# ---- Derived paths ----------------------------------------------------------
training_config="splits/$model.yaml"
save_path="exp/$exp/$model"

mkdir -p "$save_path"

# Redirect all stdout/stderr to the save path
exec > >(tee "$save_path/sbatch_${SLURM_JOB_ID}.out") 2>&1

# ---- Environment setup ------------------------------------------------------
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ijmond-camera-ai

# ---- Launch training --------------------------------------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$PORT" \
        "${method}.py" \
        --training-config "$training_config" \
        --save-path "$save_path" \
        --port "$PORT" \
        --unlabeled-sample-size "$unlabeled_sample_size" \
        --unlabeled-sample-seed "$unlabeled_sample_seed" 2>&1 | tee "$save_path/out.log"
else
    python "${method}.py" \
        --training-config "$training_config" \
        --save-path "$save_path" \
        --port "$PORT" \
        --unlabeled-sample-size "$unlabeled_sample_size" \
        --unlabeled-sample-seed "$unlabeled_sample_seed" 2>&1 | tee "$save_path/out.log"
fi