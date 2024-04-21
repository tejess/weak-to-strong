#!/bin/sh

WEAK_MODEL=gpt2
STRONG_MODEL=gpt2-large

python sweep.py \
    --model_sizes=$WEAK_MODEL,$STRONG_MODEL \
    --ds_name=hierarchical_equivalence \
    --epochs=4

# python train_simple.py \
#     --model_sizes=$STRONG_MODEL \
#     --ds_name=hierarchical_equivalence \
#     --lr=2e-05 \
#     --epochs=8 \
#     --batch_size=128 \
#     --weak_labels_path=./results/default/dn=equi_rela-e=3-l=1e-05-ms=gpt2/weak_labels