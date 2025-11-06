#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['unimatch_v2', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
method='unimatch_v2'
exp='dinov2_small'
split='smoke5k'

config=configs/${split}.yaml
labeled_id_path=splits/$split/train/labeled.txt
unlabeled_id_path=splits/$split/train/unlabeled.txt
save_path=exp/$method/$exp/$split

mkdir -p $save_path

# Parse arguments
NUM_GPUS=$1
PORT=${2:-9271}
LAUNCHER=${3:-"torch.distributed.launch"}  # torchrun or torch_distributed_launch
NNODES=${4:-1}
RANK=${5:-0}
MASTER_ADDR=${6:-"localhost"}

echo "=== Distributed Training Configuration ==="
echo "Launcher: $LAUNCHER"
echo "Number of nodes: $NNODES"
echo "Number of GPUs per node: $NUM_GPUS"
echo "Total GPUs: $((NNODES * NUM_GPUS))"
echo "Master address: $MASTER_ADDR"
echo "Master port: $PORT"
echo "Node rank: $RANK"
echo "=========================================="



#python -m $LAUNCHER \
#    --nproc_per_node=$1 \
#    --master_addr=$MASTER_ADDR \
#    --master_port=$2 \
#    $method.py \
#    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
#    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log

python \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log