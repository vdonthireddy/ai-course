import sys

libs = ['pypdf', 'fitz', 'pdfplumber', 'pypdf2', 'pdfminer']
available = []
for lib in libs:
    try:
        __import__(lib)
        available.append(lib)
    except ImportError:
        pass

print("Python version:", sys.version)
print("Available PDF libraries:", available)
