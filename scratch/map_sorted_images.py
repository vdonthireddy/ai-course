import pypdf
import os

reader = pypdf.PdfReader('/Users/donthireddy/code/ai-course/003-LLM-Building LLM from scratch.pdf')
page = reader.pages[0]

contents = page.get_contents()
if contents is not None:
    current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    matrix_stack = []
    
    placements = []
    
    for operands, operator in contents.operations:
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

    # Sort by y descending (top to bottom), then by x ascending (left to right)
    placements_sorted = sorted(placements, key=lambda p: (-p['y'], p['x']))
    
    # We want to match the XObject name (e.g. /ImX) to the extracted image file name
    # The extracted image file name is plots/llm_from_scratch/image_Y_ImX.png/jpg
    # Let's read files in plots/llm_from_scratch
    image_files = os.listdir("plots/llm_from_scratch")
    
    # Map XObject name (e.g., 'Im123') to the file path
    mapping = {}
    for filename in image_files:
        if filename.endswith(('.png', '.jpg')):
            parts = filename.split('_')
            if len(parts) >= 3:
                # format: image_index_ImName.ext
                im_name = parts[2].split('.')[0] # e.g. Im123
                mapping[im_name] = filename

    output_lines = []
    output_lines.append(f"Sorted list of all {len(placements_sorted)} image placements on the poster canvas:\n")
    for i, p in enumerate(placements_sorted):
        name_str = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        # Strip slash if present
        clean_name = name_str.lstrip('/')
        mapped_file = mapping.get(clean_name, "NOT FOUND")
        output_lines.append(f"Index: {i:3d} | XObject: {name_str:<10} | Pos: ({p['x']:7.1f}, {p['y']:7.1f}) | Size: ({p['w']:6.1f}, {p['h']:6.1f}) | File: {mapped_file}")

    with open("/Users/donthireddy/code/ai-course/scratch/sorted_images_list.txt", "w") as f:
        f.write("\n".join(output_lines))
    print("Saved sorted list to scratch/sorted_images_list.txt")
