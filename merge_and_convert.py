import subprocess
subprocess.run(["pip", "install", "-q", "peft"], check=True)

import torch
import tarfile
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_channel = os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter")
adapter_path = "/tmp/adapter_extracted"
merged_output = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# ---- Extract the tar.gz ----
os.makedirs(adapter_path, exist_ok=True)
tar_files = glob.glob(os.path.join(adapter_channel, "*.tar.gz"))
print("Found tar files:", tar_files)

with tarfile.open(tar_files[0]) as tar:
    tar.extractall(adapter_path)

print("Extracted adapter contents:", os.listdir(adapter_path))

# ---- Load base model + apply adapter ----
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

os.makedirs(merged_output, exist_ok=True)
merged_model.save_pretrained(merged_output)
tokenizer.save_pretrained(merged_output)

print("Merged model saved successfully to:", merged_output)