#!/bin/sh

WEAK_MODEL=gpt2
STRONG_MODEL=gpt2-medium

python sweep.py \
    --model_sizes=$WEAK_MODEL,$STRONG_MODEL \
    --ds_name=equivalence_relation