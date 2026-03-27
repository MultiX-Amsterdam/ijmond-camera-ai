#!/bin/bash

# modify these arguments if you want to try other splits or methods
# method: ['unimatch_v2', 'supervised', 'test_model']
# exp: just for specifying the 'save_path'
# model: ['m-zeroshot', 'm-citizen', ...]. Please check directory './splits' for available model splits

# The m-zeroshot experiment
model='m-zeroshot'
method='supervised'

# The m-citizen experiment
# model='m-citizen'
# method='unimatch_v2'

exp='dinov2_base_ddp'
unlabeled_sample_size=1500
unlabeled_sample_seed=23838742

training_config=splits/$model.yaml
save_path=exp/$exp/$model

mkdir -p $save_path

# Parse arguments
# For example, for single GPU: `sh scripts/srun_train.sh 1`
# For multi-GPU: `sh scripts/srun_train.sh 2`
NUM_GPUS=${1:-1}
PORT=${2:-9271}
LAUNCHER=${3:-"torch.distributed.launch"}  # torchrun or torch_distributed_launch
NNODES=${4:-1}
RANK=${5:-0}
MASTER_ADDR=${6:-"localhost"}

# To use this script, you need to have the conda environment 'ijmond-camera-ai' installed.
# Also, you need to first use srun to allocate a GPU and then run this script with the allocated GPU.
# srun -u --pty --nodelist=ivi-cn015 --gres=gpu:2 --mem=120G --cpus-per-task=24 --time=1:00:00 -D `pwd` bash -i

## Multi-GPU launch (torchrun)
if [ "$NUM_GPUS" -gt 1 ]; then
    conda run --no-capture-output -n ijmond-camera-ai torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$PORT \
        $method.py \
        --training-config $training_config \
        --save-path $save_path \
        --port $PORT \
        --unlabeled-sample-size $unlabeled_sample_size \
        --unlabeled-sample-seed $unlabeled_sample_seed 2>&1 | tee $save_path/out.log
else
    ## Single GPU launch
    conda run --no-capture-output -n ijmond-camera-ai python $method.py \
        --training-config $training_config \
        --save-path $save_path \
        --port $PORT \
        --unlabeled-sample-size $unlabeled_sample_size \
        --unlabeled-sample-seed $unlabeled_sample_seed 2>&1 | tee $save_path/out.log
fi