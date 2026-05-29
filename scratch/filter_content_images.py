import pypdf
import os
import re

reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
page = reader.pages[0]

contents = page.get_contents()
if contents is not None:
    current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    matrix_stack = []
    
    placements = []
    
    for operands, operator in contents.operations:
        if operator == b'q':
            matrix_stack.append(list(current_matrix))
        elif operator == b'Q':
            if matrix_stack:
                current_matrix = matrix_stack.pop()
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
        elif operator == b'Do':
            name = operands[0]
            placements.append({
                'name': name,
                'x': current_matrix[4],
                'y': current_matrix[5],
                'w': current_matrix[0],
                'h': current_matrix[3]
            })

    image_files = os.listdir("plots/llm_from_scratch")
    mapping = {}
    for filename in image_files:
        if filename.endswith(('.png', '.jpg')):
            parts = filename.split('_')
            if len(parts) >= 3:
                im_name = parts[2].split('.')[0]
                mapping[im_name] = filename

    # Filter: Keep only XObjects where the number after 'Im' is even
    filtered_placements = []
    for p in placements:
        name_str = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        clean_name = name_str.lstrip('/')
        # Extract number
        match = re.search(r'Im(\d+)', clean_name)
        if match:
            num = int(match.group(1))
            if num % 2 == 0:
                filtered_placements.append(p)

    # Sort by rounded y (nearest 100 points) descending, then by x ascending
    placements_sorted = sorted(filtered_placements, key=lambda p: (-round(p['y'], -2), p['x']))
    
    output_lines = []
    output_lines.append(f"Filtered list of all {len(placements_sorted)} content image placements:\n")
    for i, p in enumerate(placements_sorted):
        name_str = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        clean_name = name_str.lstrip('/')
        mapped_file = mapping.get(clean_name, "NOT FOUND")
        output_lines.append(f"Index: {i:3d} | XObject: {name_str:<10} | Pos: ({p['x']:7.1f}, {p['y']:7.1f}) | Size: ({p['w']:6.1f}, {p['h']:6.1f}) | File: {mapped_file}")

    with open("/Users/donthireddy/code/ai-course/scratch/filtered_content_images.txt", "w") as f:
        f.write("\n".join(output_lines))
    print(f"Saved {len(placements_sorted)} sorted content images to scratch/filtered_content_images.txt")
