#!/bin/sh

MODEL_PATH=results/default/dn=equi_rela-e=4-l=1e-05-ms=gpt2-wms=gpt2-coeff=0.3/model.safetensors
NAME=gpt2
TASK=hierarchical


python incontext.py \
    --model_name $NAME \
    --model_path $MODEL_PATH \
    --task $TASK