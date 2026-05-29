import pypdf
import sys

try:
    reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
    page = reader.pages[0]

    contents = page.get_contents()
    if contents is not None:
        print("Iterating over operations...")
        current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        matrix_stack = []
        
        placements = []
        
        for i, (operands, operator) in enumerate(contents.operations):
            try:
                if operator == b'q': # save state
                    matrix_stack.append(list(current_matrix))
                elif operator == b'Q': # restore state
                    if matrix_stack:
                        current_matrix = matrix_stack.pop()
                elif operator == b'cm': # concatenate matrix
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
                elif operator == b'Do': # Draw XObject
                    name = operands[0]
                    placements.append({
                        'name': name,
                        'x': current_matrix[4],
                        'y': current_matrix[5],
                        'w': current_matrix[0],
                        'h': current_matrix[3]
                    })
            except Exception as e:
                print(f"Error at op {i} ({operator}): {e}", file=sys.stderr)
                raise e

        print(f"Found {len(placements)} image placements on the page:")
        placements_sorted = sorted(placements, key=lambda p: (-p['y'], p['x']))
        for i, p in enumerate(placements_sorted):
            name_str = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
            print(f"{i:3d}: Name={name_str:<10} | Pos=({p['x']:.1f}, {p['y']:.1f}) | Size=({p['w']:.1f}, {p['h']:.1f})")
except Exception as e:
    print(f"Global error: {e}", file=sys.stderr)
