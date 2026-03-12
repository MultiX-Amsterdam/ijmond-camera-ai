#!/bin/bash

job='m_mix_20_unimatch_v2_dinov2_small_1500_23838742'

# modify these arguments if you want to try other splits or methods
# method: ['unimatch_v2', 'supervised', 'test_model']
# exp: just for specifying the 'save_path'
# model: ['m-zeroshot', 'm-citizien', ...]. Please check directory './splits' for available model splits

model='m-mix-20'
method='unimatch_v2'
exp='dinov2_small'
unlabeled_sample_size=1500
unlabeled_sample_seed=23838742

training_config=splits/$model.yaml
save_path=exp/$exp/$model

mkdir -p $save_path

srun --mpi=pmi2 -p $3 -n $1 \
     --gres=gpu:$1 \
     --ntasks-per-node=$1 \
     --job-name=$job \
     --open-mode=append -o $save_path/out.log \
     --quotatype=reserved \
     python3 -u $method.py \
     --training-config $training_config \
     --save-path $save_path \
     --port $PORT \
     --unlabeled-sample-size $unlabeled_sample_size \
     --unlabeled-sample-seed $unlabeled_sample_seed