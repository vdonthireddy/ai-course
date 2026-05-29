import pypdf

reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
page = reader.pages[0]
print("MediaBox:", page.mediabox)
print("CropBox:", page.cropbox)
