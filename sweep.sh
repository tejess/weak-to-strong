#!/bin/sh

WEAK_MODEL=gpt2
STRONG_MODEL=gpt2-large
python sweep.py \
    --model_sizes=$WEAK_MODEL,$STRONG_MODEL \
    --ds_name=winograd \
    --epochs=4 \
    --sweep_subfolder=hierarchical_equivalence \
    --n_docs=200 

# python train_simple.py \
#     --model_size=gpt2-large \
#     --ds_name=hierarchical_equivalence \
#     --batch_size=32 \
#     --lr=1e-05 \
#     --epochs=3 \
#     --sweep_subfolder=hierarchical_equivalence/subtask_checkpoint \
#     --loss=logconf \
#     --model_ckpt=gpt2-large-equiv-rel \
#     --strong_ckpt_path=./results/equivalence_relation/bs=32-dn=equi_rela-e=4-l=1e-05-ms=gpt2-large/model.safetensors \
#     --weak_labels_path=./results/hierarchical_equivalence/bs=32-dn=hier_equi-e=3-l=1e-05-ms=gpt2/weak_labels