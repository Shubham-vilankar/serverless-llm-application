# repackage.py
import os
import shutil
import tarfile
import glob

model_channel = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# Extract the existing merged model tar.gz
extracted = "/tmp/model_extracted"
os.makedirs(extracted, exist_ok=True)
tar_files = glob.glob(os.path.join(model_channel, "*.tar.gz"))
print("Found:", tar_files)

with tarfile.open(tar_files[0]) as tar:
    tar.extractall(extracted)

print("Extracted contents:", os.listdir(extracted))

# Copy all model files to output_dir
for item in os.listdir(extracted):
    src = os.path.join(extracted, item)
    dst = os.path.join(output_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

# Add inference.py into a 'code' subfolder (SageMaker convention)
code_dir = os.path.join(output_dir, "code")
os.makedirs(code_dir, exist_ok=True)
shutil.copy2("/opt/ml/code/inference.py", os.path.join(code_dir, "inference.py"))

print("Repackaging complete. Final contents:", os.listdir(output_dir))