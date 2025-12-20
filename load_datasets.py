from datasets import load_dataset

# Load the dataset directly from Hugging Face Hub
dataset = load_dataset("databricks/databricks-dolly-15k")

# Inspect it
print(dataset)
print(dataset["train"][0])