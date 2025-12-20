import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    prompt = data.get("inputs", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": result}
