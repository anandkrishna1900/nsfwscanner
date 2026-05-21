from PIL import Image
import numpy as np

for name in ["test_hentai.png", "test_real_breasts.png"]:
    try:
        img = Image.open(name)
        arr = np.array(img)
        print(f"File: {name}")
        print(f"  Format: {img.format}")
        print(f"  Mode: {img.mode}")
        print(f"  Size: {img.size}")
        print(f"  Mean color: {arr.mean(axis=(0, 1)) if len(arr.shape) >= 3 else arr.mean()}")
        print(f"  Min/Max pixel values: {arr.min()}, {arr.max()}")
    except Exception as e:
        print(f"Error reading {name}: {e}")
