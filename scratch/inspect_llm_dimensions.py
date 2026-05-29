import os
from PIL import Image

images_dir = "plots/llm_from_scratch"
files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))], key=lambda x: int(x.split('_')[1]))

print(f"Total images: {len(files)}")
for file in files:
    path = os.path.join(images_dir, file)
    try:
        with Image.open(path) as img:
            w, h = img.size
            # Only print images that have a reasonable size (e.g. w > 100 and h > 100)
            if w > 100 and h > 100:
                print(f"{file}: size={w}x{h}, format={img.format}")
    except Exception as e:
        print(f"Error reading {file}: {e}")
