import os
import struct

image_dir = "plots/basic_maths"
files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
images_info = []

for f in files:
    try:
        with open(f, "rb") as fp:
            # Verify PNG signature
            sig = fp.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                continue
            # Read IHDR chunk
            chunk_length_bytes = fp.read(4)
            chunk_type = fp.read(4)
            if chunk_type == b"IHDR":
                width_bytes = fp.read(4)
                height_bytes = fp.read(4)
                width = struct.unpack(">I", width_bytes)[0]
                height = struct.unpack(">I", height_bytes)[0]
                size_bytes = os.path.getsize(f)
                images_info.append((f, width, height, size_bytes))
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Sort by size in bytes descending
images_info.sort(key=lambda x: x[3], reverse=True)

print("Image Dimensions & Sizes:")
for f, w, h, size in images_info:
    print(f"{f}: {w}x{h} ({size} bytes)")
