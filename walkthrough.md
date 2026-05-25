# Walkthrough: Developer Guides & Presentation Slides Expansion

This walkthrough documents the creation of two comprehensive developer guides and the expansion of the PowerPoint presentation (`machine_learning_fundamentals.pptx`) with new content covering Deep Learning and Agentic AI.

---

## 1. Developer Guides & Diagram Integrations

Two comprehensive Markdown guides were created in the workspace, integrating 8 high-resolution, dark-themed technical diagrams:

### Deep Learning Foundations Guide
Created **[deep_learning_guide.md](file:///Users/donthireddy/code/ai-course/deep_learning_guide.md)** detailing the math, parameters, and layout steps of neural network architectures, with five custom-generated dark-themed diagrams:
- **Artificial Neuron**: [plots/perceptron_diagram.png](file:///Users/donthireddy/code/ai-course/plots/perceptron_diagram.png)
- **Deep Neural Network**: [plots/dnn_architecture.png](file:///Users/donthireddy/code/ai-course/plots/dnn_architecture.png)
- **Backpropagation Flow**: [plots/backpropagation_diagram.png](file:///Users/donthireddy/code/ai-course/plots/backpropagation_diagram.png)
- **CNN Architecture**: [plots/cnn_architecture.png](file:///Users/donthireddy/code/ai-course/plots/cnn_architecture.png)
- **Transformer Network**: [plots/transformer_architecture.png](file:///Users/donthireddy/code/ai-course/plots/transformer_architecture.png)

### Agentic AI Developer Guide
Created **[agentic_ai_developer_guide.md](file:///Users/donthireddy/code/ai-course/agentic_ai_developer_guide.md)** based on the Python sandbox codebase in `/Users/donthireddy/code/agentic/`. It explains agents, skills, tools, function calling handshakes, schemas, and the Model Context Protocol (MCP), with three custom-generated dark-themed diagrams:
- **Core Architecture Concepts**: [plots/agentic_concepts_diagram.png](file:///Users/donthireddy/code/ai-course/plots/agentic_concepts_diagram.png)
- **Function Calling Reasoning Loop**: [plots/agentic_loop_sequence.png](file:///Users/donthireddy/code/ai-course/plots/agentic_loop_sequence.png)
- **Model Context Protocol (MCP)**: [plots/mcp_architecture_diagram.png](file:///Users/donthireddy/code/ai-course/plots/mcp_architecture_diagram.png)

---

## 2. PPTX Slides Expansion (Slides 12 to 26)

We modified `generate_pptx.py` to insert 10 new slides, expanding the presentation from 21 slides to 31 slides. The layout utilizes the premium Modern Developer Dark Mode theme (Charcoal `#121214` background, Panel `#1E1E24` background, white title text, gray body text, and themed borders/accents).

### Newly Added Slides & Visuals:
- **Slide 12: The Artificial Neuron (Perceptron)**: Explains summation $z = \mathbf{w}^T \mathbf{x} + b$, step function activation, and AND gate weights, embedding `plots/perceptron_diagram.png`.
- **Slide 13: Activation Functions & Non-Linearity**: Multi-panel comparative grid showing equations, ranges, derivatives, pros, and cons of Sigmoid, Tanh, ReLU, Leaky ReLU, and Softmax.
- **Slide 14: Deep Neural Networks (ANN / DNN)**: Details forward propagation layer matrix calculations, embedding `plots/dnn_architecture.png`.
- **Slide 15: Backpropagation (Chain Rule)**: Details backward error propagation formulas and parameter optimization, embedding `plots/backpropagation_diagram.png`.
- **Slide 16: Convolutional Neural Networks (CNN)**: Covers spatial locality, shared weights, pooling, and the stride/padding size formula, embedding `plots/cnn_architecture.png`.
- **Slide 19: The Transformer Network Architecture**: Introduces the complete encoder-decoder attention-based architecture, embedding `plots/transformer_architecture.png`.
- **Slide 23: Agentic AI Core Architecture & Concepts**: Explains autonomous reasoning loops (Thought ➔ Action ➔ Observation) and function calling, embedding `plots/agentic_concepts_diagram.png`.
- **Slide 24: Runtime Introspection & Tool Schemas**: Dual-panel showing how Python function signatures and docstrings map to structured JSON declarations.
- **Slide 25: Composite Skills & Dynamic Code Execution**: Explains local packaging (`SKILL.md`/`script.py`) and execution via `exec()`, embedding `plots/agentic_loop_sequence.png`.
- **Slide 26: Model Context Protocol (MCP)**: Details clients, hosts, servers, and transport protocols, embedding `plots/mcp_architecture_diagram.png`.

---

## 3. Compilation Verification

We ran `python3 generate_pptx.py` in the workspace directory. The execution completed successfully with zero warnings/errors, outputting the updated compiled presentation file:
- **Presentation Deck**: [machine_learning_fundamentals.pptx](file:///Users/donthireddy/code/ai-course/machine_learning_fundamentals.pptx)
