from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer

## load corpus
math_corpus = load_dataset("suhaaspk/math_corpus_dataset_10M", split="train")
math_corpus = math_corpus.train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
math_corpus = math_corpus.flatten()


## pre-process data
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["equations"]])

tokenized_mc = math_corpus.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=math_corpus["train"].column_names,
)

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_mc.map(group_texts, batched=True, num_proc=4)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# randomize model

model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
def randomize_weights(model):
    for param in model.parameters():
        param.data = torch.randn_like(param)

randomize_weights(model)

## training

training_args = TrainingArguments(
    output_dir="math-corpus-model-10M",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()