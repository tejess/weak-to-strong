from weak_to_strong.model import TransformerWithHead
from transformers import AutoConfig, GPT2LMHeadModel

model = TransformerWithHead.from_pretrained(
    'gpt2-large', 
    num_labels=2, 
    linear_probe=False
)
config = AutoConfig.from_pretrained('gpt2-large')
rand_model = GPT2LMHeadModel(config)
model.lm = rand_model
model.transformer = rand_model.transformer

save_path = './results/random_checkpoints/'
(model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
    save_path
)