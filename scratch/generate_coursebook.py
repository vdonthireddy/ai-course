import os
import re

def demote_headings(text, levels=1):
    # Demotes headings by prefixing with '#' * levels
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('#'):
            # count hashes
            num_hashes = 0
            for char in line:
                if char == '#':
                    num_hashes += 1
                else:
                    break
            # get title text
            title_text = line[num_hashes:].strip()
            # replace with demoted level
            new_lines.append('#' * (num_hashes + levels) + ' ' + title_text)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def clean_intro(text, title_to_remove):
    # Remove the first heading if it matches title_to_remove
    lines = text.split('\n')
    new_lines = []
    removed_title = False
    for line in lines:
        if not removed_title and line.startswith('#') and title_to_remove.lower() in line.lower():
            removed_title = True
            continue
        # Also clean up standard "All illustrations are referenced from..." lines if we want a clean single coursebook
        if "all illustrations are referenced from" in line.lower() or "all illustration assets are referenced from" in line.lower():
            continue
        new_lines.append(line)
    return '\n'.join(new_lines)

def get_content(filename, title_to_remove):
    filepath = os.path.join('/Users/donthireddy/code/ai-course/docs', filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the intro first
    cleaned = clean_intro(content, title_to_remove)
    # Demote headings by 1 level (so ## becomes ###, etc.)
    demoted = demote_headings(cleaned, levels=1)
    return demoted

def main():
    coursebook_header = """# Complete Illustrated Machine Learning & AI Coursebook

Welcome to the **Complete Illustrated Machine Learning & AI Coursebook**. This textbook consolidates foundational mathematics, optimization mechanics, classical machine learning models, deep learning, building large language models from scratch, and agentic AI architectures into a single, cohesive, and pedagogically ordered learning pathway.

---

## Module 1: Mathematics & Calculus Foundations

This module covers the core mathematical building blocks of artificial intelligence: descriptive statistics, vector spaces, matrix operations, derivatives, differentiation rules, and the multi-variable Chain Rule.

### Part 1.1: Foundations of Machine Learning Mathematics
"""

    maths_content = get_content('basic_maths_guide.md', 'Machine Learning Mathematics')
    
    part_1_2_header = "\n\n---\n\n### Part 1.2: Visualizing Derivatives & The Chain Rule\n"
    derivatives_content = get_content('derivatives_explain.md', 'Visualizing Derivatives')

    module_2_header = """

---

## Module 2: Classical Machine Learning & Optimization

This module covers supervised and unsupervised classical machine learning paradigms, regression model fitting, optimization via gradient descent, performance metrics, classification boundaries, and clustering.

"""
    ml_basics_content = get_content('ml_basics_guide.md', 'Machine Learning Basics')

    module_3_header = """

---

## Module 3: Deep Learning, CNNs & Backpropagation Mechanics

This module covers deep learning foundations, starting from single artificial neurons and activation functions, through multi-layer architectures, backpropagation gradient calculations, Convolutional Neural Networks (CNNs), and Transformer blocks.

### Part 3.1: Visualizing Backpropagation & Gradient Descent
"""
    backprop_content = get_content('backpropagation_explain.md', 'Visualizing Backpropagation')

    part_3_2_header = "\n\n---\n\n### Part 3.2: Deep Learning Foundations (ANN, CNN, and Transformers)\n"
    deep_learning_content = get_content('deep_learning_guide.md', 'Deep Learning Foundations')

    module_4_header = """

---

## Module 4: Building Large Language Models from Scratch

This module provides a complete developer and mathematical guide to building generative GPT-like Large Language Models from scratch, covering tokenization, embedding layouts, self-attention, GPT block stack composition, next-token pretraining, and task-specific classification/instruction fine-tuning.

"""
    llm_scratch_content = get_content('llm_scratch_guide.md', 'Build a Large Language Model')

    module_5_header = """

---

## Module 5: Agentic AI & Modern LLM Applications

This module explains the design patterns and execution sequence of Agentic AI systems, dynamic tool introspection, composite skill packages, and standard interoperability layers like the Model Context Protocol (MCP).

"""
    agentic_content = get_content('agentic_ai_developer_guide.md', 'Developer Guide: Agentic AI')

    # Construct complete coursebook
    full_coursebook = (
        coursebook_header +
        maths_content +
        part_1_2_header +
        derivatives_content +
        module_2_header +
        ml_basics_content +
        module_3_header +
        backprop_content +
        part_3_2_header +
        deep_learning_content +
        module_4_header +
        llm_scratch_content +
        module_5_header +
        agentic_content
    )

    out_path = '/Users/donthireddy/code/ai-course/docs/complete_illustrated_coursebook.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full_coursebook)
        
    print(f"Successfully generated coursebook at {out_path} ({len(full_coursebook)} bytes).")

if __name__ == '__main__':
    main()
