from pypdf import PdfReader

reader = PdfReader("/Users/donthireddy/code/ai-course/002-ML Basics.pdf")
print("Total pages:", len(reader.pages))

# Inspect outline/metadata
try:
    print("Metadata:", reader.metadata)
except Exception as e:
    print("No metadata:", e)

# Print first page text
print("\n--- Page 1 Text Preview ---")
text = reader.pages[0].extract_text()
print("Length of text:", len(text))
print(text[:1500])

if len(reader.pages) > 1:
    print("\n--- Page 2 Text Preview ---")
    text2 = reader.pages[1].extract_text()
    print("Length of text:", len(text2))
    print(text2[:1500])
