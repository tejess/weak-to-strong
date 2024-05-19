#!/bin/sh

for PER in 0.00001
do
python train_simple.py \
    --model_size=gpt2-large \
    --ds_name=simple_hierarchical_equivalence \
    --batch_size=32 \
    --lr=1e-05 \
    --epochs=3 \
    --sweep_subfolder=random_init_average \
    --loss=xent \
    --model_ckpt=WTS-$PER \
    --use_validation=False \
    --random_init=$PER \
    --num_trials=10 \
    --weak_labels_path=/data/tejess/weak-to-strong/results/random_init_average/ac=0.0-bs=32-dn=simp_hier_equi-e=3-l=1e-05-mc=gpt2-$PER/weak_labels \

done
    #--strong_ckpt_path=/data/tejess/weak-to-strong/results/hierarchical_equivalencek5/ac=0.0-bs=32-dn=simp_hier_equi-e=4-l=1e-05-mc=gpt2-large/model.safetensors \
    #--just_evaluate=True \
    #--weak_labels_path=/data/tejess/weak-to-strong/results/hierarchical_equivalence/ac=0.0-bs=32-dn=simp_hier_equi-e=2-l=1e-05-mc=gpt2/weak_labels \
