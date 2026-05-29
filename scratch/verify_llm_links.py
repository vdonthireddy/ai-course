import os
import re

guide_path = "/Users/donthireddy/code/ai-course/llm_scratch_guide.md"
with open(guide_path, "r") as f:
    content = f.read()

# Find all occurrences of image syntax ![alt](path)
img_matches = re.findall(r'!\[.*?\]\((.*?)\)', content)

print(f"Verifying {len(img_matches)} image paths in {guide_path}...")
all_ok = True
for img_path in img_matches:
    # Resolve relative path to absolute path
    full_path = os.path.join("/Users/donthireddy/code/ai-course", img_path)
    exists = os.path.exists(full_path)
    status = "[OK]" if exists else "[NOT FOUND]"
    print(f"Path: {img_path:<60} | Status: {status}")
    if not exists:
        all_ok = False

if all_ok:
    print("All image references are verified and valid!")
else:
    print("Error: Some image references are missing!")
