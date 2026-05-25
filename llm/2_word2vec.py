#!/usr/bin/env python3
import math
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Word2Vec Skip-gram with Negative Sampling from Scratch
======================================================

Introduction:
-------------
Word2Vec learns continuous, low-dimensional vector representations of words such 
that words sharing semantic contexts are mapped to close coordinates in space. 
In the Skip-gram architecture, we use a target word to predict surrounding 
context words. To do this efficiently, Negative Sampling transforms the multi-class 
softmax prediction into multiple binary classification tasks: predicting whether 
a word pair is a real context pair (label 1) or a randomly sampled negative pair (label 0).

Mathematical Logic:
-------------------
1. **Sigmoid Probability**:
   $$P(\text{context}=1 | w, c) = \sigma(v_c'^T v_w) = \frac{1}{1 + \exp(-v_c'^T v_w)}$$
2. **Objective/Loss (Negative Sampling)**:
   $$\mathcal{L} = -\log \sigma(v_{c_{pos}}'^T v_w) - \sum_{i=1}^k \log \sigma(-v_{c_{neg_i}}'^T v_w)$$
3. **Parameter Updates**:
   Using Gradient Descent, we update the input vector $v_w$ and candidate output vectors $v_c'$:
   $$v_w \leftarrow v_w - \eta \cdot (\sigma(v_c'^T v_w) - t) v_c'$$
   $$v_c' \leftarrow v_c' - \eta \cdot (\sigma(v_c'^T v_w) - t) v_w$$
   where $t=1$ for positive samples and $t=0$ for negative samples.
"""

print("====================================================")
print("Word2Vec Skip-gram with Negative Sampling from Scratch")

# Set random seed for reproducibility
random.seed(42)

# 1. Prepare Corpus
corpus = [
    "machine learning is standard",
    "deep learning runs neural networks",
    "transformer models train language",
    "neural networks process inputs",
    "language models predict words"
]

# Tokenize and build vocabulary
words_list = []
for sentence in corpus:
    words_list.extend(sentence.lower().split())

vocab = sorted(list(set(words_list)))
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

print(f"\nVocabulary size: {vocab_size} unique words.")
print("Vocabulary:", vocab)

# 2. Generate Training Pairs (Skip-gram Window = 1)
window_size = 1
training_pairs = []

for sentence in corpus:
    tokens = sentence.lower().split()
    for i, target in enumerate(tokens):
        target_idx = word_to_idx[target]
        # Context window boundaries
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                context_idx = word_to_idx[tokens[j]]
                training_pairs.append((target_idx, context_idx))

print(f"Generated {len(training_pairs)} positive Skip-gram training pairs.")

# 3. Model Parameters Initialization (Embedding size d=2 for direct plotting!)
embedding_dim = 2
# Input embeddings matrix W (vocab_size x embedding_dim)
W_in = [[random.uniform(-0.5, 0.5) for _ in range(embedding_dim)] for _ in range(vocab_size)]
# Output context embeddings matrix W' (vocab_size x embedding_dim)
W_out = [[random.uniform(-0.5, 0.5) for _ in range(embedding_dim)] for _ in range(vocab_size)]

# Hyperparameters
learning_rate = 0.2
epochs = 500
num_neg_samples = 4

# Sigmoid activation helper
def sigmoid(x):
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

# Dot product helper
def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# 4. Training Loop
loss_history = []
print(f"\nTraining Word2Vec model ({epochs} epochs)...")

for epoch in range(epochs):
    epoch_loss = 0.0
    # Shuffle training pairs
    random.shuffle(training_pairs)
    
    for target_idx, pos_context_idx in training_pairs:
        # Target word vector v_w
        v_w = W_in[target_idx]
        
        # We collect all updates to add at the end of negative samples loop
        v_w_update = [0.0] * embedding_dim
        
        # --- Positive Sample (target context word, label = 1) ---
        v_pos = W_out[pos_context_idx]
        score_pos = dot_product(v_w, v_pos)
        prob_pos = sigmoid(score_pos)
        
        # positive loss: -log(sigmoid(score))
        eps = 1e-15
        epoch_loss += -math.log(max(eps, prob_pos))
        
        error_pos = prob_pos - 1.0
        # Accumulate updates for W_in[target_idx]
        for d in range(embedding_dim):
            v_w_update[d] += error_pos * v_pos[d]
            # Update W_out[pos_context_idx] directly
            W_out[pos_context_idx][d] -= learning_rate * error_pos * v_w[d]
            
        # --- Negative Samples (random words, label = 0) ---
        neg_samples = []
        while len(neg_samples) < num_neg_samples:
            neg_idx = random.randint(0, vocab_size - 1)
            # Cannot sample target word or actual context word as negative sample
            if neg_idx != target_idx and neg_idx != pos_context_idx:
                neg_samples.append(neg_idx)
                
        for neg_idx in neg_samples:
            v_neg = W_out[neg_idx]
            score_neg = dot_product(v_w, v_neg)
            prob_neg = sigmoid(score_neg)
            
            # negative loss: -log(1 - sigmoid(score))
            epoch_loss += -math.log(max(eps, 1.0 - prob_neg))
            
            error_neg = prob_neg - 0.0
            for d in range(embedding_dim):
                v_w_update[d] += error_neg * v_neg[d]
                # Update W_out[neg_idx] directly
                W_out[neg_idx][d] -= learning_rate * error_neg * v_w[d]
                
        # --- Update Target Input Vector W_in[target_idx] ---
        for d in range(embedding_dim):
            W_in[target_idx][d] -= learning_rate * v_w_update[d]
            
    avg_loss = epoch_loss / len(training_pairs)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f" - Epoch {epoch+1:3d}/{epochs} | Avg Loss: {avg_loss:.5f}")

print("\nLearned 2D Word Embeddings:")
for i in range(vocab_size):
    word = idx_to_word[i]
    coords = W_in[i]
    print(f" - '{word}': ({coords[0]:.4f}, {coords[1]:.4f})")

# 5. Generate and save visualization plots (Subplots: Left = Loss Curve, Right = Embeddings Map)
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Word2Vec Loss curve
ax1.plot(range(1, epochs + 1), loss_history, color="#EF4444", linewidth=2.5, label="Skip-gram Loss")
ax1.set_xlabel("Epochs", fontsize=11, labelpad=10)
ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, labelpad=10)
ax1.set_title("Word2Vec Negative Sampling Convergence", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right")

# Subplot 2: 2D Word Embedding Scatter Map
x_coords = [W_in[i][0] for i in range(vocab_size)]
y_coords = [W_in[i][1] for i in range(vocab_size)]

ax2.scatter(x_coords, y_coords, color="#4F46E5", s=120, edgecolor="black", zorder=3)
# Annotate each word in coordinate map
for i in range(vocab_size):
    word = idx_to_word[i]
    # Highlight specific groups of words
    if word in ["machine", "learning", "neural", "networks", "deep", "transformer", "models"]:
        color = "#10B981" # Green for models/ml jargon
    elif word in ["inputs", "words", "language"]:
        color = "#EC4899" # Pink for outputs/data
    else:
        color = "#374151" # Slate for utilities
    ax2.text(W_in[i][0] + 0.03, W_in[i][1] + 0.03, word, fontsize=10, color=color, fontweight="bold")

ax2.set_xlabel("Dimension 1", fontsize=11, labelpad=10)
ax2.set_ylabel("Dimension 2", fontsize=11, labelpad=10)
ax2.set_title("Learned 2D Semantic Feature Map", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.3)

plt.suptitle("Word2Vec Skip-gram Embeddings from Scratch", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/llm_2_word2vec.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")
