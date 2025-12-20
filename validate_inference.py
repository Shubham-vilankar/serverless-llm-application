import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import os, tarfile, glob

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape

model_channel = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
extracted = "/tmp/model_extracted"
os.makedirs(extracted, exist_ok=True)
tar_files = glob.glob(os.path.join(model_channel, "*.tar.gz"))
with tarfile.open(tar_files[0]) as tar:
    tar.extractall(extracted)

tokenizer = AutoTokenizer.from_pretrained(extracted, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    extracted, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)

prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("SUCCESS. MODEL RESPONSE:", result)
