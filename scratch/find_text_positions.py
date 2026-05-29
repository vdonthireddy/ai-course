import pypdf
import sys

reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
page = reader.pages[0]

contents = page.get_contents()
if contents is not None:
    current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    matrix_stack = []
    
    text_placements = []
    
    for operands, operator in contents.operations:
        if operator == b'q':
            matrix_stack.append((list(current_matrix), list(text_matrix)))
        elif operator == b'Q':
            if matrix_stack:
                current_matrix, text_matrix = matrix_stack.pop()
        elif operator == b'cm':
            a, b, c, d, e, f = [float(x) for x in operands]
            m = current_matrix
            current_matrix = [
                m[0]*a + m[2]*b,
                m[1]*a + m[3]*b,
                m[0]*c + m[2]*d,
                m[1]*c + m[3]*d,
                m[0]*e + m[2]*f + m[4],
                m[1]*e + m[3]*f + m[5]
            ]
        elif operator == b'BT': # Begin Text
            text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif operator == b'Tm': # Text Matrix
            text_matrix = [float(x) for x in operands]
        elif operator in (b'Td', b'TD'): # Move Text Position
            tx, ty = [float(x) for x in operands]
            text_matrix[4] += tx
            text_matrix[5] += ty
        elif operator in (b'Tj', b'TJ'): # Draw Text
            # Get text string
            # Tj operand is a string, TJ is a list of strings and numbers
            text_str = ""
            if operator == b'Tj':
                op = operands[0]
                if isinstance(op, bytes):
                    text_str = op.decode('utf-8', errors='ignore')
                else:
                    text_str = str(op)
            else: # TJ
                for x in operands[0]:
                    if isinstance(x, bytes):
                        text_str += x.decode('utf-8', errors='ignore')
                    elif isinstance(x, str):
                        text_str += x
            
            clean_text = text_str.strip()
            if clean_text:
                # Bounding box calculation (approximate)
                # Compute absolute position using current_matrix and text_matrix
                # P_abs = CM * TM * P_local
                # For (0,0) in local text space:
                # x_abs = CM[0]*TM[4] + CM[2]*TM[5] + CM[4]
                # y_abs = CM[1]*TM[4] + CM[3]*TM[5] + CM[5]
                # Let's compute this:
                x_abs = current_matrix[0]*text_matrix[4] + current_matrix[2]*text_matrix[5] + current_matrix[4]
                y_abs = current_matrix[1]*text_matrix[4] + current_matrix[3]*text_matrix[5] + current_matrix[5]
                
                text_placements.append({
                    'text': clean_text,
                    'x': x_abs,
                    'y': y_abs
                })

    print(f"Extracted {len(text_placements)} text strings:")
    # Print strings containing numbers 1 to 7 followed by dots
    for p in text_placements:
        t = p['text']
        if any(t.startswith(f"{ch}.") for ch in "1234567") or "understanding" in t.lower() or "attention" in t.lower() or "gpt" in t.lower() or "fine-tuning" in t.lower():
            print(f"Text: {t:<60} | Pos: ({p['x']:.1f}, {p['y']:.1f})")
