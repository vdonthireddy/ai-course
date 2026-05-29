import os

image_dir = "plots/basic_maths"
files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
files_with_sizes = [(f, os.path.getsize(f)) for f in files]
# Sort by file size descending
files_with_sizes.sort(key=lambda x: x[1], reverse=True)

print("Total images extracted:", len(files))
print("\nTop 15 largest images by file size:")
for f, size in files_with_sizes[:15]:
    print(f"{f}: {size} bytes")
