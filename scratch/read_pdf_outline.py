from pypdf import PdfReader

reader = PdfReader("/Users/donthireddy/code/ai-course/001-ML - Basic Maths.pdf")
print("Total pages:", len(reader.pages))

# Try to print outline/metadata
try:
    print("Metadata:", reader.metadata)
except Exception as e:
    print("No metadata:", e)

try:
    outline = reader.outline
    print("Outline exists. Top levels:")
    for item in outline[:15]:
        print(item)
except Exception as e:
    print("Error reading outline:", e)

# Print first 2 pages text
print("\n--- Page 1 Text Preview ---")
print(reader.pages[0].extract_text()[:1500])

print("\n--- Page 2 Text Preview ---")
print(reader.pages[1].extract_text()[:1500])
