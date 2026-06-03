# Build a Large Language Model (from Scratch)
### The Complete Illustrated Developer & Mathematical Guide

This comprehensive, step-by-step developer guide details the theoretical, mechanical, and mathematical foundations of building a generative GPT-like Large Language Model from scratch. This document incorporates **all 122 content diagrams** extracted from the course materials, organized sequentially by poster layout coordinates.

---

## Section 1: Understanding Large Language Models

This section covers the high-level roadmap of building Large Language Models, their nested relationship with other AI fields, BERT vs. GPT architecture modules, and zero/few-shot paradigms.

### 1.1 The Roadmap of Building an LLM
The pipeline of constructing an LLM contains three primary phases: data preparation/sampling, next-token pretraining, and task-specific or instruction-based fine-tuning.


![Building Stages Pipeline](../plots/llm_from_scratch/image_3_Im4.jpg)
*Figure 1.1: The building blocks of LLM development: Data Prep, Pretraining, and Fine-Tuning.*

![AI ML DL LLM Hierarchy](../plots/llm_from_scratch/image_5_Im6.jpg)
*Figure 1.2: Bounding relationship between Artificial Intelligence, Machine Learning, Deep Learning, and GenAI/LLMs.*

![Pretraining vs Fine-Tuning](../plots/llm_from_scratch/image_7_Im8.jpg)
*Figure 1.3: Contrast between Pretraining on unlabeled text and Fine-Tuning on task-specific labeled text.*

### 1.2 Transformer Architectures: Encoder vs. Decoder
Modern Transformers are split into submodules: BERT-style Encoders process bidirectional text for mask-prediction, while GPT-style Decoders generate text autoregressively (left-to-right).


![BERT vs GPT submodules](../plots/llm_from_scratch/image_9_Im10.jpg)
*Figure 1.4: Submodule comparison showing Bidirectional Encoder representations (BERT) and Left-to-Right Decoder representations (GPT).*

![Original Transformer Architecture](../plots/llm_from_scratch/image_11_Im12.jpg)
*Figure 1.5: The original Encoder-Decoder translation structure.*

### 1.3 Few-Shot Learning and Datasets
Emergent abilities are demonstrated by Zero-shot, One-shot, and Few-shot prompting, allowing models to perform tasks without parameter updates by learning in-context.


![In-Context Prompting](../plots/llm_from_scratch/image_13_Im14.jpg)
*Figure 1.6: Visual demonstration of zero-shot, zero-shot with instructions, and few-shot in-context learning.*

![GPT-3 Pretraining Dataset](../plots/llm_from_scratch/image_15_Im16.jpg)
*Figure 1.7: Overview table of the GPT-3 pretraining corpus tokens and proportions.*

![Iterative Text Generation Loop](../plots/llm_from_scratch/image_17_Im18.jpg)
*Figure 1.8: Loop showing how the model predicts the next word, appends it, and repeats.*

---

## Section 2: Working with Text Data

To feed text into deep learning architectures, we must convert raw characters into subword tokens, vocabulary indices, and finally into continuous dense vectors containing semantic and positional coordinates.

### 2.1 Text Embedding Workflows
Deep learning models are natively numerical and cannot process raw strings. We map text to token arrays and embed them in low-dimensional continuous vector space.


![Multimodal Embeddings](../plots/llm_from_scratch/image_19_Im20.jpg)
*Figure 2.1: Converting video, audio, and text samples into dense numerical vectors.*

![Word Embedding Scatterplot](../plots/llm_from_scratch/image_21_Im22.jpg)
*Figure 2.2: 2D scatterplot demonstrating concept clustering: similar words reside close to each other.*

![Data Sampling Pipeline Highlight](../plots/llm_from_scratch/image_23_Im24.jpg)
*Figure 2.3: Highlighting step 1 of Stage 1: The data preparation and sampling pipeline.*

### 2.2 Tokenization Algorithms and Vocabulary Mapping
We convert raw strings to tokens using tokenizers. Vocabulary maps every token to a unique integer index (Token ID).


![Word Level Tokenizer](../plots/llm_from_scratch/image_25_Im26.jpg)
*Figure 2.4: Tokenizing input text into individual words and mapping them to vocabulary indices.*

![Token ID array mapping](../plots/llm_from_scratch/image_27_Im28.jpg)
*Figure 2.5: Mapping tokens to integer vocabulary indices.*

### 2.3 Handling Out-of-Vocabulary Tokens
When encountering unknown words, simple tokenizers fail or insert `<|unk|>`. Advanced algorithms like Byte Pair Encoding (BPE) split unknown words into characters and subword tokens.


![BPE Unknown Word Decomposition](../plots/llm_from_scratch/image_29_Im30.jpg)
*Figure 2.6: BPE tokenizing an out-of-vocabulary word by splitting it into characters and known subwords.*

![BPE Tiktoken Tiktokenization](../plots/llm_from_scratch/image_31_Im32.jpg)
*Figure 2.7: BPE tiktoken tokenization mapping characters to a dense list of token IDs.*

![Tiktoken Code Example](../plots/llm_from_scratch/image_33_Im34.jpg)
*Figure 2.8: Code snippets demonstrating tiktoken vocabulary size and tokenization execution.*

![Concatenation with EoT Markers](../plots/llm_from_scratch/image_35_Im36.jpg)
*Figure 2.9: Prepend/append `<|endoftext|>` tokens between multiple independent documents.*

![Tiktoken Special Tokens Code](../plots/llm_from_scratch/image_37_Im38.jpg)
*Figure 2.10: Instantiating BPE tokenizers with special boundaries.*

### 2.4 Sliding Bins and Context Window Shifts
To train on next-token prediction, we define a sliding context window of length $T$. For each step, the inputs are $x_{1:T}$ and the targets are $y_{1:T} = x_{2:T+1}$, representing the input sequence shifted by one token.


![Sliding Window Input-Target Shifts](../plots/llm_from_scratch/image_39_Im40.jpg)
*Figure 2.11: Shifted target sequences for next-word training prediction.*

![Context Window Shifts Frame 2](../plots/llm_from_scratch/image_41_Im42.jpg)
*Figure 2.12: Slide frame showing input token IDs and their corresponding target labels.*

![PyTorch DataLoader Dataset Batching](../plots/llm_from_scratch/image_43_Im44.jpg)
*Figure 2.13: Packaging dataset into standard PyTorch tensor batches.*

![Embedding Lookup Weight Matrix](../plots/llm_from_scratch/image_45_Im46.jpg)
*Figure 2.14: Retrieving rows corresponding to incoming token index values.*

![Lookup Vectors Dimensions](../plots/llm_from_scratch/image_47_Im48.jpg)
*Figure 2.15: Mapping Token IDs to vectors of embedding dimension.*

![Positional Embedding Addition](../plots/llm_from_scratch/image_49_Im50.jpg)
*Figure 2.16: Adding positional coordinates (absolute positional embeddings) to token embeddings.*

![Continuous Vector Assembly](../plots/llm_from_scratch/image_51_Im52.jpg)
*Figure 2.17: Complete visual summary of text processing from characters to final vector tokens.*

---

## Section 3: Coding Attention Mechanisms

Attention mechanisms compute dynamic weights representing the pairwise relationships between all tokens in a sequence, allowing the model to focus on contextually relevant words.

### 3.1 Attention Basics and Weight Computation
A simple attention mechanism calculates attention weights based on vector similarity (dot products) without parameter weights.


![Self-Attention Context Vector Calculation](../plots/llm_from_scratch/image_55_Im56.jpg)
*Figure 3.1: Visualizing how a token builds its context vector from other tokens.*

![Attention Scores Similarity Dot Product](../plots/llm_from_scratch/image_57_Im58.jpg)
*Figure 3.2: Computing attention scores using vector dot products.*

![Softmax Normalization of Weights](../plots/llm_from_scratch/image_59_Im60.jpg)
*Figure 3.3: Softmax function scaling attention scores to sum to 1.0 (probabilities).*

![Weighted Value Addition](../plots/llm_from_scratch/image_61_Im62.jpg)
*Figure 3.4: Multiplying value tokens by normalized attention weights.*

![Weight Multiplication Matrix Visualization](../plots/llm_from_scratch/image_53_Im54.jpg)
*Figure 3.5: Step-by-step matrix representation of context vector calculation.*

### 3.2 Parameterized Self-Attention: Queries, Keys, and Values
We parameterize self-attention by projecting input tokens $X$ into Query ($Q$), Key ($K$), and Value ($V$) matrices using three learned projection weight matrices:

$$Q = X W_q \quad K = X W_k \quad V = X W_v$$


![Query Key Value Projections](../plots/llm_from_scratch/image_63_Im64.jpg)
*Figure 3.6: Projecting inputs into Query, Key, and Value vector representations.*

![Query Key Dot Product Scores](../plots/llm_from_scratch/image_65_Im66.jpg)
*Figure 3.7: Query-Key similarity dot products.*

![Query Vector Row Matrix Multiplication](../plots/llm_from_scratch/image_67_Im68.jpg)
*Figure 3.8: Matrix multiplication layout of Queries and Keys.*

![Attention Score Matrix Mapping](../plots/llm_from_scratch/image_69_Im70.jpg)
*Figure 3.9: Softmax attention map showing pairwise scores.*

![Query Key Value Matrix Product](../plots/llm_from_scratch/image_71_Im72.jpg)
*Figure 3.10: The complete query, key, value matrix pipeline.*

### 3.3 Scaled Dot-Product Attention
We divide dot product scores by the scaling factor $\sqrt{d_k}$ (square root of the key projection dimension) to maintain vector magnitude and prevent vanishing gradients during softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$


![Scaled Similarity Multiplier](../plots/llm_from_scratch/image_73_Im74.jpg)
*Figure 3.11: Scaling scores to stabilize training variance.*

![Self-Attention Class Code](../plots/llm_from_scratch/image_75_Im76.jpg)
*Figure 3.12: Code implementing query, key, value projections and context vector assembly.*

![Scaled Dot-Product Formula Illustration](../plots/llm_from_scratch/image_77_Im78.jpg)
*Figure 3.13: Step-by-step visual of the scaled dot product equation.*

![Causal Mask Multiplier](../plots/llm_from_scratch/image_79_Im80.jpg)
*Figure 3.14: Causal masking to prevent model from looking at future words.*

![Causal Mask Matrix Representation](../plots/llm_from_scratch/image_81_Im82.jpg)
*Figure 3.15: Setting upper triangle matrix values to -\infty.*

![Softmax Mask Mapping](../plots/llm_from_scratch/image_89_Im90.jpg)
*Figure 3.16: Softmax converting masked values to 0.0 attention scores.*

![Causal Self-Attention Code](../plots/llm_from_scratch/image_91_Im92.jpg)
*Figure 3.17: Code implementing causal masking in PyTorch.*

![Dropout Normalization](../plots/llm_from_scratch/image_83_Im84.jpg)
*Figure 3.18: Applying dropout to attention matrices to prevent co-adaptation.*

![Dropout Visual Diagram](../plots/llm_from_scratch/image_243_Im244.png)
*Figure 3.19: Randomly zeroing out attention matrix values during training.*

![Causal Mask Self-Attention Final Summary](../plots/llm_from_scratch/image_85_Im86.jpg)
*Figure 3.20: Complete causal self-attention workflow.*

### 3.4 Multi-Head Attention (MHA)
Instead of computing attention once, Multi-Head Attention splits the Queries, Keys, and Values into $H$ heads, computes attention in parallel, and concatenates the outputs:


![Multi-Head Splitting](../plots/llm_from_scratch/image_87_Im88.jpg)
*Figure 3.21: Splitting token dimensions into multiple attention heads.*

![Parallel Heads Computation](../plots/llm_from_scratch/image_95_Im96.jpg)
*Figure 3.22: Processing parallel attention weights.*

![Heads Concatenation](../plots/llm_from_scratch/image_99_Im100.jpg)
*Figure 3.23: Concatenating head outputs back to original token dimension.*

![Multi-Head Attention Code](../plots/llm_from_scratch/image_101_Im102.jpg)
*Figure 3.24: Code implementing Multi-Head Attention.*

![Multi-Head Attention Diagram](../plots/llm_from_scratch/image_93_Im94.jpg)
*Figure 3.25: Layout of Multi-Head Attention layer.*

![Multi-Head Attention Final Matrix Output](../plots/llm_from_scratch/image_97_Im98.jpg)
*Figure 3.26: Matrix pipeline of Multi-Head Attention.*

---

## Section 4: Implementing a GPT Model from Scratch

GPT models compose stacked Transformer blocks. This section details layer normalization, activations, skip connections, and token decoding configurations.

### 4.1 Layer Normalization (LayerNorm)
LayerNorm computes mean and variance across the feature dimension for each token independently, stabilizing scale distributions:


![LayerNorm vs BatchNorm Dimensions](../plots/llm_from_scratch/image_113_Im114.jpg)
*Figure 4.1: Normalization dimensions: LayerNorm (across features) vs. BatchNorm (across batch).*

![LayerNorm Execution Math](../plots/llm_from_scratch/image_115_Im116.jpg)
*Figure 4.2: Normalizing token features to zero mean and unit variance.*

![LayerNorm PyTorch Code](../plots/llm_from_scratch/image_117_Im118.jpg)
*Figure 4.3: Custom LayerNorm implementation.*

### 4.2 GELU Activation Function
GPT blocks use Gaussian Error Linear Units (GELU) in the MLP block:

$$\text{GELU}(x) = 0.5x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$


![GELU Activation Curve](../plots/llm_from_scratch/image_119_Im120.jpg)
*Figure 4.4: GELU activation function graph: smooth curve preventing dead neurons.*

![GELU Code Snippet](../plots/llm_from_scratch/image_1_Im2.png)
*Figure 4.5: PyTorch GELU activation implementation.*

### 4.3 GPT Block Assembly
Each GPT block contains LayerNorm, Multi-Head Attention, residual connections, and Feed-Forward Networks (MLP blocks).


![MLP Block Code](../plots/llm_from_scratch/image_109_Im110.jpg)
*Figure 4.6: Feed-forward network (MLP) block construction.*

![GPT Block Code Structure](../plots/llm_from_scratch/image_103_Im104.jpg)
*Figure 4.7: GPT Block code putting together MHA and MLP.*

![Residual Connection Mechanics](../plots/llm_from_scratch/image_105_Im106.jpg)
*Figure 4.8: Adding input shortcuts directly to layer outputs.*

![Residual Skip Code](../plots/llm_from_scratch/image_107_Im108.jpg)
*Figure 4.9: Code implementing residual connections.*

![Transformer Block Bounding Connections](../plots/llm_from_scratch/image_111_Im112.jpg)
*Figure 4.10: Visual overview of a single GPT Transformer block.*

### 4.4 GPT Model Stack
We stack multiple Transformer blocks to construct the complete GPT model:


![GPT Model Code](../plots/llm_from_scratch/image_121_Im122.jpg)
*Figure 4.11: Custom GPTModel class implementing embedding, stacked blocks, and final linear head.*

![GPT Model Bounding Layers](../plots/llm_from_scratch/image_123_Im124.jpg)
*Figure 4.12: Dense layer diagram of the stacked GPT architecture.*

![GPT Parameters Count Code](../plots/llm_from_scratch/image_125_Im126.jpg)
*Figure 4.13: Calculating the total trainable parameter counts.*

![Logits Output Projection Head](../plots/llm_from_scratch/image_127_Im128.jpg)
*Figure 4.14: Logits projection mapping final output dimension back to vocabulary size.*

![Next Word Logits Indexing](../plots/llm_from_scratch/image_129_Im130.jpg)
*Figure 4.15: Selecting logits at the final token position to predict the next word.*

### 4.5 Decoding Strategies
To generate text, we sample from output probabilities. We configure decoding behaviors:
*   **Greedy Search**: Always select the token with the highest probability.
*   **Temperature Scaling**: Scale logits by $T$ before softmax to adjust randomness.
*   **Top-k Sampling**: Keep only the top $k$ highest probability tokens, redistribute softmax.


![Text Generation Mechanics](../plots/llm_from_scratch/image_131_Im132.jpg)
*Figure 4.16: Flow showing token IDs mapped to logits, scaled, mapped to probabilities, and sampled.*

![Text Generation Execution Pipeline](../plots/llm_from_scratch/image_135_Im136.png)
*Figure 4.17: Sequence showing next-token predictions iteratively feeding back into the inputs.*

![Temperature Scaling Graph](../plots/llm_from_scratch/image_137_Im138.jpg)
*Figure 4.18: Impact of temperature scaling on probability distribution shapes.*

![Top-k Sampling Graph](../plots/llm_from_scratch/image_139_Im140.jpg)
*Figure 4.19: Filtering out low-probability tails via Top-k.*

![Text Generation Python Code](../plots/llm_from_scratch/image_133_Im134.jpg)
*Figure 4.20: Complete text generation decoding function.*

---

## Section 5: Pretraining on Unlabeled Data

This section details batching raw text, tracking loss, calculating perplexity, scheduling learning rates, and checkpointing weights.

### 5.1 Training Batches and Logits
We batch inputs $x$ and targets $y$, pass inputs through the model, and align logits to target tokens to calculate Cross-Entropy loss.


![PyTorch DataLoader Inputs Targets](../plots/llm_from_scratch/image_141_Im142.jpg)
*Figure 5.1: DataLoader outputting batches of token inputs and target outputs.*

![Aligned Logits Targets Loss](../plots/llm_from_scratch/image_143_Im144.jpg)
*Figure 5.2: Aligning outputs to target indices for loss calculation.*

![Cross Entropy Loss Code](../plots/llm_from_scratch/image_145_Im146.jpg)
*Figure 5.3: PyTorch cross-entropy evaluation code.*

![Model Training Loop Code](../plots/llm_from_scratch/image_147_Im148.png)
*Figure 5.4: Custom training loop tracking batch loss.*

### 5.2 Validation Loss Curves and Perplexity
We calculate validation loss on held-out text. Perplexity (PPL) evaluates next-token predictions:

$$\text{PPL} = e^{\mathcal{L}}$$


![Loss Curves Plot](../plots/llm_from_scratch/image_149_Im150.png)
*Figure 5.5: Training vs. Validation loss curve plot showing convergence.*

![Loss Values Printout](../plots/llm_from_scratch/image_151_Im152.jpg)
*Figure 5.6: Logging outputs showing loss and perplexity.*

![Perplexity Metric Printout](../plots/llm_from_scratch/image_153_Im154.jpg)
*Figure 5.7: Detailed validation log showing perplexity values.*

### 5.3 Learning Rate Scheduling and Warmup
To optimize deep training convergence, we use Cosine Annealing learning rate schedules with a linear warmup phase.


![Cosine Learning Rate Schedule Plot](../plots/llm_from_scratch/image_157_Im158.jpg)
*Figure 5.8: Learning rate decay schedule plot over steps.*

![Cosine Schedule Code](../plots/llm_from_scratch/image_155_Im156.jpg)
*Figure 5.9: Cosine annealing learning rate scheduling implementation.*

![Training Iteration Code](../plots/llm_from_scratch/image_159_Im160.jpg)
*Figure 5.10: Incorporating scheduler updates in training loops.*

### 5.4 Saving Checkpoints and Loading Weight Files
We serialize model weights (parameters) to disk and load them back for evaluation or HuggingFace/OpenAI weight translation.


![Saving Weights PyTorch Code](../plots/llm_from_scratch/image_161_Im162.jpg)
*Figure 5.11: Serialization saving weights file.*

![Loading Weights PyTorch Code](../plots/llm_from_scratch/image_163_Im164.jpg)
*Figure 5.12: Loading weights file back to model.*

![Weight Translation Code](../plots/llm_from_scratch/image_165_Im166.jpg)
*Figure 5.13: Translating checkpoint parameters from OpenAI formats.*

![Load OpenAI Weight Maps Code](../plots/llm_from_scratch/image_167_Im168.jpg)
*Figure 5.14: Code mapping keys from standard GPT-2 models.*

![Checkpoint Evaluation Printout](../plots/llm_from_scratch/image_169_Im170.png)
*Figure 5.15: Printout showing generation output from loaded checkpoints.*

![HuggingFace GPT2 Model Integration](../plots/llm_from_scratch/image_171_Im172.png)
*Figure 5.16: Model validation prints matching HuggingFace GPT-2 parameters.*

---

## Section 6: Fine-Tuning for Classification

To convert a generative foundation model into a text classifier (e.g. classifying messages as spam vs. ham), we modify its architecture and train it using supervised data.

### 6.1 Classification Dataset Loading and Padded Batches
Incoming messages have variable sequence lengths. We pad shorter sequences with padding tokens (e.g. `<|endoftext|>`) to a uniform length to allow parallel batch operations.


![Padded Token ID Batches](../plots/llm_from_scratch/image_193_Im194.jpg)
*Figure 6.1: Padded token IDs and corresponding class labels array.*

![Variable Length Messages Padding](../plots/llm_from_scratch/image_195_Im196.jpg)
*Figure 6.2: Padding variable text inputs to uniform token length.*

![PyTorch Classification DataLoader Code](../plots/llm_from_scratch/image_197_Im198.png)
*Figure 6.3: Custom DataLoader class executing sequence padding.*

![DataLoader Batches Output Print](../plots/llm_from_scratch/image_173_Im174.jpg)
*Figure 6.4: Log prints showing batched token ID tensor shape.*

![Supervised Dataset Splits Table](../plots/llm_from_scratch/image_175_Im176.jpg)
*Figure 6.5: Splitting classification dataset into Train, Validation, and Test.*

### 6.2 Classification Head Replacement
We replace the vocabulary-sized language model head (decoder output projection) with a classification head $W_c \in \mathbb{R}^{D \times C}$, where $C$ is the number of target classes.


![Output Head Linear Projection](../plots/llm_from_scratch/image_185_Im186.png)
*Figure 6.6: Swapping next-token head with linear classification projection.*

![Final Token Index Extraction](../plots/llm_from_scratch/image_177_Im178.png)
*Figure 6.7: Bounding output representation at the final token position.*

![Linear Output Class Projection Head](../plots/llm_from_scratch/image_179_Im180.png)
*Figure 6.8: Extracting final token representations for input to the classification head.*

![Classification Model Code Class](../plots/llm_from_scratch/image_181_Im182.jpg)
*Figure 6.9: Custom GPTClassifier model class implementation.*

### 6.3 Classifier Training and Evaluation Metrics
We optimize the model using classification cross-entropy loss, and evaluate accuracy, precision, recall, and F1-score.


![Accuracy Evaluation Code](../plots/llm_from_scratch/image_183_Im184.jpg)
*Figure 6.10: Code computing prediction accuracy.*

![Batch Classification Loss Code](../plots/llm_from_scratch/image_189_Im190.jpg)
*Figure 6.11: Loss calculation over classification batches.*

![Classification Loss Curves Plot](../plots/llm_from_scratch/image_199_Im200.png)
*Figure 6.12: Classifier Train vs. Validation loss convergence curve.*

![Classifier Accuracy Curve Plot](../plots/llm_from_scratch/image_187_Im188.jpg)
*Figure 6.13: Classifier Train vs. Validation accuracy growth curve.*

![Spam Prediction Examples Print](../plots/llm_from_scratch/image_191_Im192.jpg)
*Figure 6.14: Sample predictions output logs.*

![Confusion Matrix Visual Chart](../plots/llm_from_scratch/image_201_Im202.png)
*Figure 6.15: Confusion matrix showing True Positives, True Negatives, False Positives, False Negatives.*

![Classification Metrics Summary](../plots/llm_from_scratch/image_203_Im204.png)
*Figure 6.16: Final F1-score evaluation metrics log.*

---

## Section 7: Fine-Tuning to Follow Instructions

Instruction fine-tuning trains a foundation model to behave as a helpful personal assistant. We format prompt-response sequences, mask inputs in the loss function, and evaluate conversational outputs.

### 7.1 Instruction Dataset Formats and Alpaca Style
Instruction datasets structure samples into instructions, inputs, and responses. We format samples using prompts templates:


![Prompt Template Layout](../plots/llm_from_scratch/image_205_Im206.png)
*Figure 7.1: Visualizing template structures wrapping instruction and response text.*

![Formatted Prompt Text Sample](../plots/llm_from_scratch/image_207_Im208.png)
*Figure 7.2: Text prompt showing instruction, input context, and target response.*

![Dataset Sample Representation Table](../plots/llm_from_scratch/image_209_Im210.png)
*Figure 7.3: Table showing instruction, input, output values.*

![Instruction Dataset Class Code](../plots/llm_from_scratch/image_211_Im212.png)
*Figure 7.4: Custom Dataset class processing instruction strings.*

![DataLoader Padded Instruction Batch](../plots/llm_from_scratch/image_213_Im214.jpg)
*Figure 7.5: Padding prompt-response token sequences to uniform length.*

### 7.2 Loss Masking on Prompts
To prevent the model from learning to copy instructions, we apply a mask to the input prompt tokens during loss calculation. Cross-entropy loss is computed only on the target response tokens.


![Instruction Loss Masking Concept](../plots/llm_from_scratch/image_215_Im216.jpg)
*Figure 7.6: Masking out prompt token logits (setting loss weight to zero) and computing loss on response tokens.*

![Prompt Mask Targets Realignment](../plots/llm_from_scratch/image_217_Im218.jpg)
*Figure 7.7: Aligning target tensor IDs: prompt tokens are replaced with -100.*

![PyTorch Cross-Entropy Index Masking](../plots/llm_from_scratch/image_219_Im220.png)
*Figure 7.8: Setting ignore_index=-100 in cross-entropy loss function.*

![Loss Masking Code Implementation](../plots/llm_from_scratch/image_221_Im222.png)
*Figure 7.9: Code implementing custom masking inside DataLoader collation.*

![DataLoader Mask Batches Print](../plots/llm_from_scratch/image_223_Im224.jpg)
*Figure 7.10: Printout showing target ID arrays with -100 mask values.*

### 7.3 Instruction Training Loop and Evaluation
We load pretrained foundation weights, compile masked loss functions, run the optimizer, and evaluate qualitatively and quantitatively.


![Model Loading Pretrained Weights](../plots/llm_from_scratch/image_225_Im226.jpg)
*Figure 7.11: Initializing foundation GPT model and loading parameters.*

![Instruction Training Loop Code](../plots/llm_from_scratch/image_227_Im228.png)
*Figure 7.12: Custom SFT training loop.*

![SFT Training Loss Curve Plot](../plots/llm_from_scratch/image_229_Im230.png)
*Figure 7.13: SFT training convergence plot.*

![Qualitative Evaluation Code](../plots/llm_from_scratch/image_231_Im232.png)
*Figure 7.14: Code generating conversational responses from prompts.*

![Three Stage Pipeline Summary](../plots/llm_from_scratch/image_241_Im242.png)
*Figure 7.15: Visual summary: Preparing dataset, fine-tuning model, scoring responses.*

![Qualitative Response Output Logs](../plots/llm_from_scratch/image_233_Im234.png)
*Figure 7.16: Sample assistant answers logs.*

![Assistant Model scoring criteria](../plots/llm_from_scratch/image_235_Im236.png)
*Figure 7.17: Scoring assistant response quality.*

![MMLU Benchmark scoring table](../plots/llm_from_scratch/image_237_Im238.png)
*Figure 7.18: Model performance scores across standard benchmarks.*

![LLM as a Judge Scoring Interface](../plots/llm_from_scratch/image_239_Im240.png)
*Figure 7.19: Using another model (LLM-as-a-judge) to grade generated response quality.*