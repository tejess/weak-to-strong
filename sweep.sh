#!/bin/sh

# WEAK_MODEL=gpt2
# STRONG_MODEL=gpt2-large
# python sweep.py \
#     --model_sizes=$WEAK_MODEL,$STRONG_MODEL \
#     --ds_name=winograd \
#     --epochs=4 \
#     --sweep_subfolder=winograd \
#     --loss logconf \
#     --aux_coeff 0.1 \
#     --n_docs=200 

python train_simple.py \
    --model_size=gpt2-large \
    --ds_name=winograd \
    --batch_size=32 \
    --lr=1e-05 \
    --epochs=3 \
    --sweep_subfolder=winograd \
    --loss=logconf \
    --aux_coeff=0.1 \
    --model_ckpt=gpt2-large-winograd \
    --strong_ckpt_path=./results/winograd/bs=32-dn=winograd-e=4-l=1e-05-mc=gpt2-large/model.safetensors \
    --weak_labels_path=./results/winograd/bs=32-dn=winograd-e=4-l=1e-05-mc=gpt2-large/weak_labels:q