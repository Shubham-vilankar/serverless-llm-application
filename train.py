# train.py

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_dir = os.environ.get("SM_MODEL_DIR", "./model_output")
train_path = os.path.join(os.environ.get("SM_CHANNEL_TRAIN", "./data"), "train.jsonl")
val_path = os.path.join(os.environ.get("SM_CHANNEL_VAL", "./data"), "val.jsonl")

base_model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()   # <-- fixes the gradient error
model.print_trainable_parameters()

train_dataset = load_dataset("json", data_files=train_path, split="train")
val_dataset = load_dataset("json", data_files=val_path, split="train")

def format_fn(example):
    return example["prompt"] + example["completion"]

# ---- SFTConfig replaces TrainingArguments + carries max_seq_length ----
training_args = SFTConfig(
    output_dir=model_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    # max_steps=5,              # dry run
    learning_rate=2e-4,
    logging_steps=1,
    eval_strategy="no",
    save_strategy="no",
    bf16=True,
    report_to="none",
    max_seq_length=512,       # now lives here, not in SFTTrainer
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=format_fn,
)

trainer.train()

trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

print("Dry run complete. Model saved to:", model_dir)