import csv
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from utils.image_utils import prepare_image_for_onnx

# Download model and tags
model_path = hf_hub_download(
    repo_id="SmilingWolf/wd-vit-large-tagger-v3",
    filename="model.onnx",
    cache_dir="./models",
)
tags_path = hf_hub_download(
    repo_id="SmilingWolf/wd-vit-large-tagger-v3",
    filename="selected_tags.csv",
    cache_dir="./models",
)

# Load tags
tag_names = []
with open(tags_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tag_names.append(row.get("name", row.get("tag_id", "")))

# Load session
opts = ort.SessionOptions()
opts.log_severity_level = 3
session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load image
img_path = "test_hentai.png"
image = Image.open(img_path)

print("Original image mode:", image.mode, "size:", image.size)

# Preprocess 1: Original (squashed)
img1 = image.convert("RGB").resize((448, 448), Image.BICUBIC)
arr1 = np.array(img1, dtype=np.float32) / 255.0
arr1 = arr1[:, :, ::-1].copy()
arr1 = np.expand_dims(arr1, axis=0)

# Preprocess 2: prepare_image_for_onnx (padded)
arr2 = prepare_image_for_onnx(image, target_size=448, to_bgr=True)

# Run Preprocess 1
out1 = session.run([output_name], {input_name: arr1})[0][0]
# Run Preprocess 2
out2 = session.run([output_name], {input_name: arr2})[0][0]

# Print top 15 tags for original preprocessing
print("\n--- Top 15 Tags (Original Squashed Preprocessing) ---")
idx1 = np.argsort(out1)[::-1]
for i in range(15):
    idx = idx1[i]
    print(f"  {tag_names[idx]}: {out1[idx]:.4f}")

# Print top 15 tags for padded preprocessing
print("\n--- Top 15 Tags (Padded Preprocessing) ---")
idx2 = np.argsort(out2)[::-1]
for i in range(15):
    idx = idx2[i]
    print(f"  {tag_names[idx]}: {out2[idx]:.4f}")
