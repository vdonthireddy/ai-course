with open("/Users/donthireddy/code/ai-course/scratch/filtered_content_images.txt", "r") as f:
    lines = f.readlines()

# Parsed data structure
placements = []
for line in lines:
    if line.startswith("Index:"):
        # Format: Index:   0 | XObject: /Im4       | Pos: ( 1062.3,  7733.4) | Size: ( 258.0,  123.0) | File: image_3_Im4.jpg
        parts = line.split(" | ")
        index_str = parts[0].split(":")[1].strip()
        xobject_str = parts[1].split(":")[1].strip()
        
        pos_part = parts[2].split(":")[1].strip() # ( 1062.3,  7733.4)
        pos_coords = [float(x) for x in pos_part.lstrip('(').rstrip(')').split(',')]
        x, y = pos_coords[0], pos_coords[1]
        
        file_str = parts[4].split(":")[1].strip()
        placements.append({
            'index': int(index_str),
            'xobject': xobject_str,
            'x': x,
            'y': y,
            'file': file_str
        })

# Bounding coordinates for chapters
# Chapter 1: y >= 7379.1
# Chapter 2: 7379.1 > y >= 6240.1
# Chapter 3: 6240.1 > y >= 4838.6
# Chapter 4: 4838.6 > y >= 3581.3
# Chapter 5: 3581.3 > y >= 2396.1
# Chapter 6: 2396.1 > y >= 1186.9
# Chapter 7: 1186.9 > y

chapters = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: []
}

for p in placements:
    y = p['y']
    if y >= 7379.1:
        chapters[1].append(p)
    elif y >= 6240.1:
        chapters[2].append(p)
    elif y >= 4838.6:
        chapters[3].append(p)
    elif y >= 3581.3:
        chapters[4].append(p)
    elif y >= 2396.1:
        chapters[5].append(p)
    elif y >= 1186.9:
        chapters[6].append(p)
    else:
        chapters[7].append(p)

chapter_names = {
    1: "Section 1: Understanding Large Language Models",
    2: "Section 2: Working with Text Data",
    3: "Section 3: Coding Attention Mechanisms",
    4: "Section 4: Implementing a GPT Model from Scratch",
    5: "Section 5: Pretraining on Unlabeled Data",
    6: "Section 6: Fine-Tuning for Classification",
    7: "Section 7: Fine-Tuning to Follow Instructions"
}

markdown_output = []
for ch_num in range(1, 8):
    markdown_output.append(f"### {chapter_names[ch_num]}\n")
    markdown_output.append(f"Contains {len(chapters[ch_num])} diagrams:\n")
    for i, p in enumerate(chapters[ch_num]):
        markdown_output.append(f"*   `image_{p['index']}`: `plots/llm_from_scratch/{p['file']}` (Pos: {p['x']:.1f}, {p['y']:.1f})")
    markdown_output.append("")

with open("/Users/donthireddy/code/ai-course/scratch/chapters_markdown.txt", "w") as f:
    f.write("\n".join(markdown_output))
print("Saved chapter groupings to scratch/chapters_markdown.txt")
