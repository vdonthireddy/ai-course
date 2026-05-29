from pypdf import PdfReader

reader = PdfReader("/Users/donthireddy/code/ai-course/002-ML Basics.pdf")
text = reader.pages[0].extract_text()
print("Full Text:")
print(text)
