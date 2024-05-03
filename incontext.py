import pandas as pd
import random
import re
import argparse

from weak_to_strong.model import TransformerWithHead
from safetensors.torch import load_model
from datasets import load_dataset
from weak_to_strong.train import ModelConfig
from train_simple import MODEL_CONFIGS, MODELS_DICT
from weak_to_strong.common import get_tokenizer

def load_transformer_model(path, model):
    load_model(model, path)

def load_custom_dataset(task):
    if task == 'equivalence':
        path = 'dangnguyen0420/equivalence_relation'
    else:
        path = 'dangnguyen0420/hierarchical_equivalence'
    
    dataset = load_dataset(path)
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    return train_df, test_df

def zero_shot(df, model, tokenizer, max_tokens=5):
    for ind in df.index:
        input = df['input'][ind]
        model_inputs = tokenizer([input], return_tensors="pt").to('cuda')
        output = model.lm.generate(**model_inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        output = output[len(input):]
        print("Prompt:", input)
        print("Response:", output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, choices=MODELS_DICT.keys(), help='Model name')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--task', type=str, choices=['equivalence', 'hierarchical'], help='Task name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_config = MODELS_DICT[args.model_name]
    custom_kwargs = model_config.custom_kwargs or {}
    tokenizer = get_tokenizer(args.model_name)
    model = TransformerWithHead.from_pretrained(
            model_config.name, num_labels=2, linear_probe=False, **custom_kwargs
        ).to("cuda")
    load_transformer_model(args.model_path, model)
    _, test_df = load_custom_dataset(args.task)
    zero_shot(test_df, model, tokenizer)

if __name__ == "__main__":
    main()