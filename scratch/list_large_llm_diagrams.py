import os
from PIL import Image

images_dir = "plots/llm_from_scratch"
files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))], key=lambda x: int(x.split('_')[1]))

large_diagrams = []
for file in files:
    path = os.path.join(images_dir, file)
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w > 500:
                large_diagrams.append((file, w, h, img.format))
    except Exception as e:
        pass

print(f"Total large diagrams: {len(large_diagrams)}")
for diag in large_diagrams:
    print(f"File: {diag[0]:<25} | Size: {diag[1]}x{diag[2]:<5} | Format: {diag[3]}")
