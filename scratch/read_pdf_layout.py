import pypdf

reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
page = reader.pages[0]

def visitor_body(text, cm, tm, font_dict, font_size):
    if text.strip():
        print(f"Text: {text.strip():<40} | Pos: ({tm[4]:.1f}, {tm[5]:.1f})")

page.extract_text(visitor_text=visitor_body)
