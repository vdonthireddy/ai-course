from pypdf import PdfReader

reader = PdfReader("/Users/donthireddy/code/ai-course/001-ML - Basic Maths.pdf")
print("Total pages:", len(reader.pages))

text = reader.pages[0].extract_text()
print("Length of text:", len(text))
print("--- Full Text ---")
print(text)
