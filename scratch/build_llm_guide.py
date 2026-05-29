import re

# Read chapter groupings
with open("scratch/chapters_markdown.txt", "r") as f:
    text = f.read()

# Let's extract all image lists grouped by section
sections = re.split(r'### Section ', text)[1:]

image_groups = {}
for s in sections:
    lines = s.strip().split("\n")
    header = lines[0].strip()
    sec_num = int(header.split(":")[0])
    
    images = []
    for line in lines:
        if line.strip().startswith("*"):
            # Format: *   image_0: plots/llm_from_scratch/image_3_Im4.jpg (Pos: 1062.3, 7733.4)
            match = re.search(r'plots/llm_from_scratch/(image_\d+_Im\d+\.(?:jpg|png))', line)
            if match:
                images.append(match.group(1))
    image_groups[sec_num] = images

# Now we write the markdown document
md = []
md.append("""# Build a Large Language Model (from Scratch)
### The Complete Illustrated Developer & Mathematical Guide

This comprehensive, step-by-step developer guide details the theoretical, mechanical, and mathematical foundations of building a generative GPT-like Large Language Model from scratch. This document incorporates **all 122 content diagrams** extracted from the course materials, organized sequentially by poster layout coordinates.

---

## Section 1: Understanding Large Language Models

This section covers the high-level roadmap of building Large Language Models, their nested relationship with other AI fields, BERT vs. GPT architecture modules, and zero/few-shot paradigms.

### 1.1 The Roadmap of Building an LLM
The pipeline of constructing an LLM contains three primary phases: data preparation/sampling, next-token pretraining, and task-specific or instruction-based fine-tuning.

""")

# Chapter 1 Images: image_groups[1]
ch1 = image_groups[1]
md.append(f"![Building Stages Pipeline](plots/llm_from_scratch/{ch1[0]})\n*Figure 1.1: The building blocks of LLM development: Data Prep, Pretraining, and Fine-Tuning.*")
md.append(f"\n![AI ML DL LLM Hierarchy](plots/llm_from_scratch/{ch1[1]})\n*Figure 1.2: Bounding relationship between Artificial Intelligence, Machine Learning, Deep Learning, and GenAI/LLMs.*")
md.append(f"\n![Pretraining vs Fine-Tuning](plots/llm_from_scratch/{ch1[2]})\n*Figure 1.3: Contrast between Pretraining on unlabeled text and Fine-Tuning on task-specific labeled text.*")

md.append("""
### 1.2 Transformer Architectures: Encoder vs. Decoder
Modern Transformers are split into submodules: BERT-style Encoders process bidirectional text for mask-prediction, while GPT-style Decoders generate text autoregressively (left-to-right).
""")
md.append(f"\n![BERT vs GPT submodules](plots/llm_from_scratch/{ch1[3]})\n*Figure 1.4: Submodule comparison showing Bidirectional Encoder representations (BERT) and Left-to-Right Decoder representations (GPT).*")
md.append(f"\n![Original Transformer Architecture](plots/llm_from_scratch/{ch1[4]})\n*Figure 1.5: The original Encoder-Decoder translation structure.*")

md.append("""
### 1.3 Few-Shot Learning and Datasets
Emergent abilities are demonstrated by Zero-shot, One-shot, and Few-shot prompting, allowing models to perform tasks without parameter updates by learning in-context.
""")
md.append(f"\n![In-Context Prompting](plots/llm_from_scratch/{ch1[5]})\n*Figure 1.6: Visual demonstration of zero-shot, zero-shot with instructions, and few-shot in-context learning.*")
md.append(f"\n![GPT-3 Pretraining Dataset](plots/llm_from_scratch/{ch1[6]})\n*Figure 1.7: Overview table of the GPT-3 pretraining corpus tokens and proportions.*")
md.append(f"\n![Iterative Text Generation Loop](plots/llm_from_scratch/{ch1[7]})\n*Figure 1.8: Loop showing how the model predicts the next word, appends it, and repeats.*")

# Chapter 2: Working with Text Data
ch2 = image_groups[2]
md.append("""
---

## Section 2: Working with Text Data

To feed text into deep learning architectures, we must convert raw characters into subword tokens, vocabulary indices, and finally into continuous dense vectors containing semantic and positional coordinates.

### 2.1 Text Embedding Workflows
Deep learning models are natively numerical and cannot process raw strings. We map text to token arrays and embed them in low-dimensional continuous vector space.
""")
md.append(f"\n![Multimodal Embeddings](plots/llm_from_scratch/{ch2[0]})\n*Figure 2.1: Converting video, audio, and text samples into dense numerical vectors.*")
md.append(f"\n![Word Embedding Scatterplot](plots/llm_from_scratch/{ch2[1]})\n*Figure 2.2: 2D scatterplot demonstrating concept clustering: similar words reside close to each other.*")
md.append(f"\n![Data Sampling Pipeline Highlight](plots/llm_from_scratch/{ch2[2]})\n*Figure 2.3: Highlighting step 1 of Stage 1: The data preparation and sampling pipeline.*")

md.append("""
### 2.2 Tokenization Algorithms and Vocabulary Mapping
We convert raw strings to tokens using tokenizers. Vocabulary maps every token to a unique integer index (Token ID).
""")
md.append(f"\n![Word Level Tokenizer](plots/llm_from_scratch/{ch2[3]})\n*Figure 2.4: Tokenizing input text into individual words and mapping them to vocabulary indices.*")
md.append(f"\n![Token ID array mapping](plots/llm_from_scratch/{ch2[4]})\n*Figure 2.5: Mapping tokens to integer vocabulary indices.*")

md.append("""
### 2.3 Handling Out-of-Vocabulary Tokens
When encountering unknown words, simple tokenizers fail or insert `<|unk|>`. Advanced algorithms like Byte Pair Encoding (BPE) split unknown words into characters and subword tokens.
""")
md.append(f"\n![BPE Unknown Word Decomposition](plots/llm_from_scratch/{ch2[5]})\n*Figure 2.6: BPE tokenizing an out-of-vocabulary word by splitting it into characters and known subwords.*")
md.append(f"\n![BPE Tiktoken Tiktokenization](plots/llm_from_scratch/{ch2[6]})\n*Figure 2.7: BPE tiktoken tokenization mapping characters to a dense list of token IDs.*")
md.append(f"\n![Tiktoken Code Example](plots/llm_from_scratch/{ch2[7]})\n*Figure 2.8: Code snippets demonstrating tiktoken vocabulary size and tokenization execution.*")
md.append(f"\n![Concatenation with EoT Markers](plots/llm_from_scratch/{ch2[8]})\n*Figure 2.9: Prepend/append `<|endoftext|>` tokens between multiple independent documents.*")
md.append(f"\n![Tiktoken Special Tokens Code](plots/llm_from_scratch/{ch2[9]})\n*Figure 2.10: Instantiating BPE tokenizers with special boundaries.*")

md.append("""
### 2.4 Sliding Bins and Context Window Shifts
To train on next-token prediction, we define a sliding context window of length $T$. For each step, the inputs are $x_{1:T}$ and the targets are $y_{1:T} = x_{2:T+1}$, representing the input sequence shifted by one token.
""")
md.append(f"\n![Sliding Window Input-Target Shifts](plots/llm_from_scratch/{ch2[10]})\n*Figure 2.11: Shifted target sequences for next-word training prediction.*")
md.append(f"\n![Context Window Shifts Frame 2](plots/llm_from_scratch/{ch2[11]})\n*Figure 2.12: Slide frame showing input token IDs and their corresponding target labels.*")
md.append(f"\n![PyTorch DataLoader Dataset Batching](plots/llm_from_scratch/{ch2[12]})\n*Figure 2.13: Packaging dataset into standard PyTorch tensor batches.*")
md.append(f"\n![Embedding Lookup Weight Matrix](plots/llm_from_scratch/{ch2[13]})\n*Figure 2.14: Retrieving rows corresponding to incoming token index values.*")
md.append(f"\n![Lookup Vectors Dimensions](plots/llm_from_scratch/{ch2[14]})\n*Figure 2.15: Mapping Token IDs to vectors of embedding dimension.*")
md.append(f"\n![Positional Embedding Addition](plots/llm_from_scratch/{ch2[15]})\n*Figure 2.16: Adding positional coordinates (absolute positional embeddings) to token embeddings.*")
md.append(f"\n![Continuous Vector Assembly](plots/llm_from_scratch/{ch2[16]})\n*Figure 2.17: Complete visual summary of text processing from characters to final vector tokens.*")

# Chapter 3: Coding Attention Mechanisms
ch3 = image_groups[3]
md.append(r"""
---

## Section 3: Coding Attention Mechanisms

Attention mechanisms compute dynamic weights representing the pairwise relationships between all tokens in a sequence, allowing the model to focus on contextually relevant words.

### 3.1 Attention Basics and Weight Computation
A simple attention mechanism calculates attention weights based on vector similarity (dot products) without parameter weights.
""")
md.append(f"\n![Self-Attention Context Vector Calculation](plots/llm_from_scratch/{ch3[0]})\n*Figure 3.1: Visualizing how a token builds its context vector from other tokens.*")
md.append(f"\n![Attention Scores Similarity Dot Product](plots/llm_from_scratch/{ch3[1]})\n*Figure 3.2: Computing attention scores using vector dot products.*")
md.append(f"\n![Softmax Normalization of Weights](plots/llm_from_scratch/{ch3[2]})\n*Figure 3.3: Softmax function scaling attention scores to sum to 1.0 (probabilities).*")
md.append(f"\n![Weighted Value Addition](plots/llm_from_scratch/{ch3[3]})\n*Figure 3.4: Multiplying value tokens by normalized attention weights.*")
md.append(f"\n![Weight Multiplication Matrix Visualization](plots/llm_from_scratch/{ch3[4]})\n*Figure 3.5: Step-by-step matrix representation of context vector calculation.*")

md.append(r"""
### 3.2 Parameterized Self-Attention: Queries, Keys, and Values
We parameterize self-attention by projecting input tokens $X$ into Query ($Q$), Key ($K$), and Value ($V$) matrices using three learned projection weight matrices:

$$Q = X W_q \quad K = X W_k \quad V = X W_v$$
""")
md.append(f"\n![Query Key Value Projections](plots/llm_from_scratch/{ch3[5]})\n*Figure 3.6: Projecting inputs into Query, Key, and Value vector representations.*")
md.append(f"\n![Query Key Dot Product Scores](plots/llm_from_scratch/{ch3[6]})\n*Figure 3.7: Query-Key similarity dot products.*")
md.append(f"\n![Query Vector Row Matrix Multiplication](plots/llm_from_scratch/{ch3[7]})\n*Figure 3.8: Matrix multiplication layout of Queries and Keys.*")
md.append(f"\n![Attention Score Matrix Mapping](plots/llm_from_scratch/{ch3[8]})\n*Figure 3.9: Softmax attention map showing pairwise scores.*")
md.append(f"\n![Query Key Value Matrix Product](plots/llm_from_scratch/{ch3[9]})\n*Figure 3.10: The complete query, key, value matrix pipeline.*")

md.append(r"""
### 3.3 Scaled Dot-Product Attention
We divide dot product scores by the scaling factor $\sqrt{d_k}$ (square root of the key projection dimension) to maintain vector magnitude and prevent vanishing gradients during softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
""")
md.append(f"\n![Scaled Similarity Multiplier](plots/llm_from_scratch/{ch3[10]})\n*Figure 3.11: Scaling scores to stabilize training variance.*")
md.append(f"\n![Self-Attention Class Code](plots/llm_from_scratch/{ch3[11]})\n*Figure 3.12: Code implementing query, key, value projections and context vector assembly.*")
md.append(f"\n![Scaled Dot-Product Formula Illustration](plots/llm_from_scratch/{ch3[12]})\n*Figure 3.13: Step-by-step visual of the scaled dot product equation.*")
md.append(f"\n![Causal Mask Multiplier](plots/llm_from_scratch/{ch3[13]})\n*Figure 3.14: Causal masking to prevent model from looking at future words.*")
md.append(f"\n![Causal Mask Matrix Representation](plots/llm_from_scratch/{ch3[14]})\n*Figure 3.15: Setting upper triangle matrix values to -\infty.*")
md.append(f"\n![Softmax Mask Mapping](plots/llm_from_scratch/{ch3[15]})\n*Figure 3.16: Softmax converting masked values to 0.0 attention scores.*")
md.append(f"\n![Causal Self-Attention Code](plots/llm_from_scratch/{ch3[16]})\n*Figure 3.17: Code implementing causal masking in PyTorch.*")
md.append(f"\n![Dropout Normalization](plots/llm_from_scratch/{ch3[17]})\n*Figure 3.18: Applying dropout to attention matrices to prevent co-adaptation.*")
md.append(f"\n![Dropout Visual Diagram](plots/llm_from_scratch/{ch3[18]})\n*Figure 3.19: Randomly zeroing out attention matrix values during training.*")
md.append(f"\n![Causal Mask Self-Attention Final Summary](plots/llm_from_scratch/{ch3[19]})\n*Figure 3.20: Complete causal self-attention workflow.*")

md.append(r"""
### 3.4 Multi-Head Attention (MHA)
Instead of computing attention once, Multi-Head Attention splits the Queries, Keys, and Values into $H$ heads, computes attention in parallel, and concatenates the outputs:
""")
md.append(f"\n![Multi-Head Splitting](plots/llm_from_scratch/{ch3[20]})\n*Figure 3.21: Splitting token dimensions into multiple attention heads.*")
md.append(f"\n![Parallel Heads Computation](plots/llm_from_scratch/{ch3[21]})\n*Figure 3.22: Processing parallel attention weights.*")
md.append(f"\n![Heads Concatenation](plots/llm_from_scratch/{ch3[22]})\n*Figure 3.23: Concatenating head outputs back to original token dimension.*")
md.append(f"\n![Multi-Head Attention Code](plots/llm_from_scratch/{ch3[23]})\n*Figure 3.24: Code implementing Multi-Head Attention.*")
md.append(f"\n![Multi-Head Attention Diagram](plots/llm_from_scratch/{ch3[24]})\n*Figure 3.25: Layout of Multi-Head Attention layer.*")
md.append(f"\n![Multi-Head Attention Final Matrix Output](plots/llm_from_scratch/{ch3[25]})\n*Figure 3.26: Matrix pipeline of Multi-Head Attention.*")

# Chapter 4: Implementing a GPT Model
ch4 = image_groups[4]
md.append(r"""
---

## Section 4: Implementing a GPT Model from Scratch

GPT models compose stacked Transformer blocks. This section details layer normalization, activations, skip connections, and token decoding configurations.

### 4.1 Layer Normalization (LayerNorm)
LayerNorm computes mean and variance across the feature dimension for each token independently, stabilizing scale distributions:
""")
md.append(f"\n![LayerNorm vs BatchNorm Dimensions](plots/llm_from_scratch/{ch4[0]})\n*Figure 4.1: Normalization dimensions: LayerNorm (across features) vs. BatchNorm (across batch).*")
md.append(f"\n![LayerNorm Execution Math](plots/llm_from_scratch/{ch4[1]})\n*Figure 4.2: Normalizing token features to zero mean and unit variance.*")
md.append(f"\n![LayerNorm PyTorch Code](plots/llm_from_scratch/{ch4[2]})\n*Figure 4.3: Custom LayerNorm implementation.*")

md.append(r"""
### 4.2 GELU Activation Function
GPT blocks use Gaussian Error Linear Units (GELU) in the MLP block:

$$\text{GELU}(x) = 0.5x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$
""")
md.append(f"\n![GELU Activation Curve](plots/llm_from_scratch/{ch4[3]})\n*Figure 4.4: GELU activation function graph: smooth curve preventing dead neurons.*")
md.append(f"\n![GELU Code Snippet](plots/llm_from_scratch/{ch4[4]})\n*Figure 4.5: PyTorch GELU activation implementation.*")

md.append(r"""
### 4.3 GPT Block Assembly
Each GPT block contains LayerNorm, Multi-Head Attention, residual connections, and Feed-Forward Networks (MLP blocks).
""")
md.append(f"\n![MLP Block Code](plots/llm_from_scratch/{ch4[5]})\n*Figure 4.6: Feed-forward network (MLP) block construction.*")
md.append(f"\n![GPT Block Code Structure](plots/llm_from_scratch/{ch4[6]})\n*Figure 4.7: GPT Block code putting together MHA and MLP.*")
md.append(f"\n![Residual Connection Mechanics](plots/llm_from_scratch/{ch4[7]})\n*Figure 4.8: Adding input shortcuts directly to layer outputs.*")
md.append(f"\n![Residual Skip Code](plots/llm_from_scratch/{ch4[8]})\n*Figure 4.9: Code implementing residual connections.*")
md.append(f"\n![Transformer Block Bounding Connections](plots/llm_from_scratch/{ch4[9]})\n*Figure 4.10: Visual overview of a single GPT Transformer block.*")

md.append(r"""
### 4.4 GPT Model Stack
We stack multiple Transformer blocks to construct the complete GPT model:
""")
md.append(f"\n![GPT Model Code](plots/llm_from_scratch/{ch4[10]})\n*Figure 4.11: Custom GPTModel class implementing embedding, stacked blocks, and final linear head.*")
md.append(f"\n![GPT Model Bounding Layers](plots/llm_from_scratch/{ch4[11]})\n*Figure 4.12: Dense layer diagram of the stacked GPT architecture.*")
md.append(f"\n![GPT Parameters Count Code](plots/llm_from_scratch/{ch4[12]})\n*Figure 4.13: Calculating the total trainable parameter counts.*")
md.append(f"\n![Logits Output Projection Head](plots/llm_from_scratch/{ch4[13]})\n*Figure 4.14: Logits projection mapping final output dimension back to vocabulary size.*")
md.append(f"\n![Next Word Logits Indexing](plots/llm_from_scratch/{ch4[14]})\n*Figure 4.15: Selecting logits at the final token position to predict the next word.*")

md.append(r"""
### 4.5 Decoding Strategies
To generate text, we sample from output probabilities. We configure decoding behaviors:
*   **Greedy Search**: Always select the token with the highest probability.
*   **Temperature Scaling**: Scale logits by $T$ before softmax to adjust randomness.
*   **Top-k Sampling**: Keep only the top $k$ highest probability tokens, redistribute softmax.
""")
md.append(f"\n![Text Generation Mechanics](plots/llm_from_scratch/{ch4[15]})\n*Figure 4.16: Flow showing token IDs mapped to logits, scaled, mapped to probabilities, and sampled.*")
md.append(f"\n![Text Generation Execution Pipeline](plots/llm_from_scratch/{ch4[16]})\n*Figure 4.17: Sequence showing next-token predictions iteratively feeding back into the inputs.*")
md.append(f"\n![Temperature Scaling Graph](plots/llm_from_scratch/{ch4[17]})\n*Figure 4.18: Impact of temperature scaling on probability distribution shapes.*")
md.append(f"\n![Top-k Sampling Graph](plots/llm_from_scratch/{ch4[18]})\n*Figure 4.19: Filtering out low-probability tails via Top-k.*")
md.append(f"\n![Text Generation Python Code](plots/llm_from_scratch/{ch4[19]})\n*Figure 4.20: Complete text generation decoding function.*")

# Chapter 5: Pretraining on Unlabeled Data
ch5 = image_groups[5]
md.append(r"""
---

## Section 5: Pretraining on Unlabeled Data

This section details batching raw text, tracking loss, calculating perplexity, scheduling learning rates, and checkpointing weights.

### 5.1 Training Batches and Logits
We batch inputs $x$ and targets $y$, pass inputs through the model, and align logits to target tokens to calculate Cross-Entropy loss.
""")
md.append(f"\n![PyTorch DataLoader Inputs Targets](plots/llm_from_scratch/{ch5[0]})\n*Figure 5.1: DataLoader outputting batches of token inputs and target outputs.*")
md.append(f"\n![Aligned Logits Targets Loss](plots/llm_from_scratch/{ch5[1]})\n*Figure 5.2: Aligning outputs to target indices for loss calculation.*")
md.append(f"\n![Cross Entropy Loss Code](plots/llm_from_scratch/{ch5[2]})\n*Figure 5.3: PyTorch cross-entropy evaluation code.*")
md.append(f"\n![Model Training Loop Code](plots/llm_from_scratch/{ch5[3]})\n*Figure 5.4: Custom training loop tracking batch loss.*")

md.append(r"""
### 5.2 Validation Loss Curves and Perplexity
We calculate validation loss on held-out text. Perplexity (PPL) evaluates next-token predictions:

$$\text{PPL} = e^{\mathcal{L}}$$
""")
md.append(f"\n![Loss Curves Plot](plots/llm_from_scratch/{ch5[4]})\n*Figure 5.5: Training vs. Validation loss curve plot showing convergence.*")
md.append(f"\n![Loss Values Printout](plots/llm_from_scratch/{ch5[5]})\n*Figure 5.6: Logging outputs showing loss and perplexity.*")
md.append(f"\n![Perplexity Metric Printout](plots/llm_from_scratch/{ch5[6]})\n*Figure 5.7: Detailed validation log showing perplexity values.*")

md.append(r"""
### 5.3 Learning Rate Scheduling and Warmup
To optimize deep training convergence, we use Cosine Annealing learning rate schedules with a linear warmup phase.
""")
md.append(f"\n![Cosine Learning Rate Schedule Plot](plots/llm_from_scratch/{ch5[7]})\n*Figure 5.8: Learning rate decay schedule plot over steps.*")
md.append(f"\n![Cosine Schedule Code](plots/llm_from_scratch/{ch5[8]})\n*Figure 5.9: Cosine annealing learning rate scheduling implementation.*")
md.append(f"\n![Training Iteration Code](plots/llm_from_scratch/{ch5[9]})\n*Figure 5.10: Incorporating scheduler updates in training loops.*")

md.append(r"""
### 5.4 Saving Checkpoints and Loading Weight Files
We serialize model weights (parameters) to disk and load them back for evaluation or HuggingFace/OpenAI weight translation.
""")
md.append(f"\n![Saving Weights PyTorch Code](plots/llm_from_scratch/{ch5[10]})\n*Figure 5.11: Serialization saving weights file.*")
md.append(f"\n![Loading Weights PyTorch Code](plots/llm_from_scratch/{ch5[11]})\n*Figure 5.12: Loading weights file back to model.*")
md.append(f"\n![Weight Translation Code](plots/llm_from_scratch/{ch5[12]})\n*Figure 5.13: Translating checkpoint parameters from OpenAI formats.*")
md.append(f"\n![Load OpenAI Weight Maps Code](plots/llm_from_scratch/{ch5[13]})\n*Figure 5.14: Code mapping keys from standard GPT-2 models.*")
md.append(f"\n![Checkpoint Evaluation Printout](plots/llm_from_scratch/{ch5[14]})\n*Figure 5.15: Printout showing generation output from loaded checkpoints.*")
md.append(f"\n![HuggingFace GPT2 Model Integration](plots/llm_from_scratch/{ch5[15]})\n*Figure 5.16: Model validation prints matching HuggingFace GPT-2 parameters.*")

# Chapter 6: Fine-Tuning for Classification
ch6 = image_groups[6]
md.append(r"""
---

## Section 6: Fine-Tuning for Classification

To convert a generative foundation model into a text classifier (e.g. classifying messages as spam vs. ham), we modify its architecture and train it using supervised data.

### 6.1 Classification Dataset Loading and Padded Batches
Incoming messages have variable sequence lengths. We pad shorter sequences with padding tokens (e.g. `<|endoftext|>`) to a uniform length to allow parallel batch operations.
""")
md.append(f"\n![Padded Token ID Batches](plots/llm_from_scratch/{ch6[0]})\n*Figure 6.1: Padded token IDs and corresponding class labels array.*")
md.append(f"\n![Variable Length Messages Padding](plots/llm_from_scratch/{ch6[1]})\n*Figure 6.2: Padding variable text inputs to uniform token length.*")
md.append(f"\n![PyTorch Classification DataLoader Code](plots/llm_from_scratch/{ch6[2]})\n*Figure 6.3: Custom DataLoader class executing sequence padding.*")
md.append(f"\n![DataLoader Batches Output Print](plots/llm_from_scratch/{ch6[3]})\n*Figure 6.4: Log prints showing batched token ID tensor shape.*")
md.append(f"\n![Supervised Dataset Splits Table](plots/llm_from_scratch/{ch6[4]})\n*Figure 6.5: Splitting classification dataset into Train, Validation, and Test.*")

md.append(r"""
### 6.2 Classification Head Replacement
We replace the vocabulary-sized language model head (decoder output projection) with a classification head $W_c \in \mathbb{R}^{D \times C}$, where $C$ is the number of target classes.
""")
md.append(f"\n![Output Head Linear Projection](plots/llm_from_scratch/{ch6[5]})\n*Figure 6.6: Swapping next-token head with linear classification projection.*")
md.append(f"\n![Final Token Index Extraction](plots/llm_from_scratch/{ch6[6]})\n*Figure 6.7: Bounding output representation at the final token position.*")
md.append(f"\n![Linear Output Class Projection Head](plots/llm_from_scratch/{ch6[7]})\n*Figure 6.8: Extracting final token representations for input to the classification head.*")
md.append(f"\n![Classification Model Code Class](plots/llm_from_scratch/{ch6[8]})\n*Figure 6.9: Custom GPTClassifier model class implementation.*")

md.append(r"""
### 6.3 Classifier Training and Evaluation Metrics
We optimize the model using classification cross-entropy loss, and evaluate accuracy, precision, recall, and F1-score.
""")
md.append(f"\n![Accuracy Evaluation Code](plots/llm_from_scratch/{ch6[9]})\n*Figure 6.10: Code computing prediction accuracy.*")
md.append(f"\n![Batch Classification Loss Code](plots/llm_from_scratch/{ch6[10]})\n*Figure 6.11: Loss calculation over classification batches.*")
md.append(f"\n![Classification Loss Curves Plot](plots/llm_from_scratch/{ch6[11]})\n*Figure 6.12: Classifier Train vs. Validation loss convergence curve.*")
md.append(f"\n![Classifier Accuracy Curve Plot](plots/llm_from_scratch/{ch6[12]})\n*Figure 6.13: Classifier Train vs. Validation accuracy growth curve.*")
md.append(f"\n![Spam Prediction Examples Print](plots/llm_from_scratch/{ch6[13]})\n*Figure 6.14: Sample predictions output logs.*")
md.append(f"\n![Confusion Matrix Visual Chart](plots/llm_from_scratch/{ch6[14]})\n*Figure 6.15: Confusion matrix showing True Positives, True Negatives, False Positives, False Negatives.*")
md.append(f"\n![Classification Metrics Summary](plots/llm_from_scratch/{ch6[15]})\n*Figure 6.16: Final F1-score evaluation metrics log.*")

# Chapter 7: Fine-Tuning to Follow Instructions
ch7 = image_groups[7]
md.append(r"""
---

## Section 7: Fine-Tuning to Follow Instructions

Instruction fine-tuning trains a foundation model to behave as a helpful personal assistant. We format prompt-response sequences, mask inputs in the loss function, and evaluate conversational outputs.

### 7.1 Instruction Dataset Formats and Alpaca Style
Instruction datasets structure samples into instructions, inputs, and responses. We format samples using prompts templates:
""")
md.append(f"\n![Prompt Template Layout](plots/llm_from_scratch/{ch7[0]})\n*Figure 7.1: Visualizing template structures wrapping instruction and response text.*")
md.append(f"\n![Formatted Prompt Text Sample](plots/llm_from_scratch/{ch7[1]})\n*Figure 7.2: Text prompt showing instruction, input context, and target response.*")
md.append(f"\n![Dataset Sample Representation Table](plots/llm_from_scratch/{ch7[2]})\n*Figure 7.3: Table showing instruction, input, output values.*")
md.append(f"\n![Instruction Dataset Class Code](plots/llm_from_scratch/{ch7[3]})\n*Figure 7.4: Custom Dataset class processing instruction strings.*")
md.append(f"\n![DataLoader Padded Instruction Batch](plots/llm_from_scratch/{ch7[4]})\n*Figure 7.5: Padding prompt-response token sequences to uniform length.*")

md.append(r"""
### 7.2 Loss Masking on Prompts
To prevent the model from learning to copy instructions, we apply a mask to the input prompt tokens during loss calculation. Cross-entropy loss is computed only on the target response tokens.
""")
md.append(f"\n![Instruction Loss Masking Concept](plots/llm_from_scratch/{ch7[5]})\n*Figure 7.6: Masking out prompt token logits (setting loss weight to zero) and computing loss on response tokens.*")
md.append(f"\n![Prompt Mask Targets Realignment](plots/llm_from_scratch/{ch7[6]})\n*Figure 7.7: Aligning target tensor IDs: prompt tokens are replaced with -100.*")
md.append(f"\n![PyTorch Cross-Entropy Index Masking](plots/llm_from_scratch/{ch7[7]})\n*Figure 7.8: Setting ignore_index=-100 in cross-entropy loss function.*")
md.append(f"\n![Loss Masking Code Implementation](plots/llm_from_scratch/{ch7[8]})\n*Figure 7.9: Code implementing custom masking inside DataLoader collation.*")
md.append(f"\n![DataLoader Mask Batches Print](plots/llm_from_scratch/{ch7[9]})\n*Figure 7.10: Printout showing target ID arrays with -100 mask values.*")

md.append(r"""
### 7.3 Instruction Training Loop and Evaluation
We load pretrained foundation weights, compile masked loss functions, run the optimizer, and evaluate qualitatively and quantitatively.
""")
md.append(f"\n![Model Loading Pretrained Weights](plots/llm_from_scratch/{ch7[10]})\n*Figure 7.11: Initializing foundation GPT model and loading parameters.*")
md.append(f"\n![Instruction Training Loop Code](plots/llm_from_scratch/{ch7[11]})\n*Figure 7.12: Custom SFT training loop.*")
md.append(f"\n![SFT Training Loss Curve Plot](plots/llm_from_scratch/{ch7[12]})\n*Figure 7.13: SFT training convergence plot.*")
md.append(f"\n![Qualitative Evaluation Code](plots/llm_from_scratch/{ch7[13]})\n*Figure 7.14: Code generating conversational responses from prompts.*")
md.append(f"\n![Three Stage Pipeline Summary](plots/llm_from_scratch/{ch7[14]})\n*Figure 7.15: Visual summary: Preparing dataset, fine-tuning model, scoring responses.*")
md.append(f"\n![Qualitative Response Output Logs](plots/llm_from_scratch/{ch7[15]})\n*Figure 7.16: Sample assistant answers logs.*")
md.append(f"\n![Assistant Model scoring criteria](plots/llm_from_scratch/{ch7[16]})\n*Figure 7.17: Scoring assistant response quality.*")
md.append(f"\n![MMLU Benchmark scoring table](plots/llm_from_scratch/{ch7[17]})\n*Figure 7.18: Model performance scores across standard benchmarks.*")
md.append(f"\n![LLM as a Judge Scoring Interface](plots/llm_from_scratch/{ch7[18]})\n*Figure 7.19: Using another model (LLM-as-a-judge) to grade generated response quality.*")

# Join the markdown elements
full_markdown = "\n".join(md)

# Write to file
with open("/Users/donthireddy/code/ai-course/llm_scratch_guide.md", "w") as f:
    f.write(full_markdown)

print("llm_scratch_guide.md written successfully!")
