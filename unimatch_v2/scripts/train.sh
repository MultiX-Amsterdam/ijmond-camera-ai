#!/bin/bash

# modify these arguments if you want to try other splits or methods
# method: ['unimatch_v2', 'supervised', 'test_model']
# exp: just for specifying the 'save_path'
# model: ['m-zeroshot', 'm-citizen', ...]. Please check directory './splits' for available model splits

model='m-zeroshot'
method='supervised'
exp='dinov2_small'
unlabeled_sample_size=1500
unlabeled_sample_seed=23838742

training_config=splits/$model.yaml
save_path=exp/$exp/$model

mkdir -p $save_path

# Parse arguments
NUM_GPUS=${1:-1}
PORT=${2:-9271}
LAUNCHER=${3:-"torch.distributed.launch"}  # torchrun or torch_distributed_launch
NNODES=${4:-1}
RANK=${5:-0}
MASTER_ADDR=${6:-"localhost"}

## For distributed launch (multiple GPUs) uncomment the following lines and comment out single GPU launch

#python -m $LAUNCHER \
#    --nproc_per_node=$NUM_GPUS \
#    --master_addr=$MASTER_ADDR \
#    --master_port=$PORT \
#    $method.py \
#    --training-config $training_config \
#    --save-path $save_path
#    --port $PORT
#    --unlabeled-sample-size $unlabeled_sample_size \
#    --unlabeled-sample-seed $unlabeled_sample_seed 2>&1 | tee $save_path/out.log

## Single GPU launch

python $method.py \
    --training-config $training_config \
    --save-path $save_path \
    --port $PORT \
    --unlabeled-sample-size $unlabeled_sample_size \
    --unlabeled-sample-seed $unlabeled_sample_seed 2>&1 | tee $save_path/out.log