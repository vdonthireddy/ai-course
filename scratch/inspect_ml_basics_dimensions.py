import os
import struct

image_dir = "plots/ml_basics"
files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
images_info = []

for f in files:
    try:
        size_bytes = os.path.getsize(f)
        if f.lower().endswith((".jpg", ".jpeg")):
            # Just read the file size for now for JPGs, or get dimensions if we can.
            # A simple way to get JPG dimensions in pure python:
            with open(f, "rb") as fp:
                data = fp.read(4)
                # We can just record size in bytes, but let's try to parse JPG dimensions if needed.
                # For this check, size in bytes is usually enough to identify major images.
                images_info.append((f, 0, 0, size_bytes))
            continue
            
        with open(f, "rb") as fp:
            sig = fp.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                continue
            chunk_length_bytes = fp.read(4)
            chunk_type = fp.read(4)
            if chunk_type == b"IHDR":
                width_bytes = fp.read(4)
                height_bytes = fp.read(4)
                width = struct.unpack(">I", width_bytes)[0]
                height = struct.unpack(">I", height_bytes)[0]
                images_info.append((f, width, height, size_bytes))
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Sort by size in bytes descending
images_info.sort(key=lambda x: x[3], reverse=True)

print("ML Basics Image Dimensions & Sizes (Top 30):")
for f, w, h, size in images_info[:30]:
    if w == 0 and h == 0:
        print(f"{f}: (JPG) ({size} bytes)")
    else:
        print(f"{f}: {w}x{h} ({size} bytes)")
