import os
from pypdf import PdfReader

# Create directory for images if it doesn't exist
os.makedirs("plots/basic_maths", exist_ok=True)

reader = PdfReader("/Users/donthireddy/code/ai-course/001-ML - Basic Maths.pdf")
page = reader.pages[0]

print("Images on page 0:", len(page.images))
for i, image_file_object in enumerate(page.images):
    name = f"plots/basic_maths/image_{i}_{image_file_object.name}"
    # Clean name if needed
    name = "".join(c for c in name if c.isalnum() or c in "._-/")
    print(f"Saving image to {name}")
    with open(name, "wb") as fp:
        fp.write(image_file_object.data)
