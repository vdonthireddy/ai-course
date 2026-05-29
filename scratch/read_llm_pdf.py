import pypdf

reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
page = reader.pages[0]
text = page.extract_text()
print("--- Text Content ---")
print(text)
print("--------------------")
