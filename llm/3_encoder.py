#!/usr/bin/env python3
import math
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Transformer Encoder Layer from Scratch
======================================

Introduction:
-------------
The Transformer Encoder processes an input sequence of token embeddings in parallel. 
It uses self-attention to determine how much weight/attention each token should pay 
to every other token in the sequence. To retain sequence order (since self-attention 
has no inherent sequence order awareness), we add a Positional Encoding vector to 
each input embedding.

Mathematical Logic:
-------------------
1. **Positional Encoding (PE)**:
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
   $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
2. **Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
3. **Layer Normalization (LayerNorm)**:
   $$\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
4. **Feed-Forward Network (FFN)**:
   $$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$
"""

print("====================================================")
print("Transformer Encoder Layer from Scratch")

random.seed(42)

# 1. Math and Matrix Helper Functions
def matmul(A, B):
    """Matrix multiplication: A (m x n) x B (n x p) -> (m x p)"""
    m, n = len(A), len(A[0])
    n_check, p = len(B), len(B[0])
    assert n == n_check, f"Dimension mismatch: {n} != {n_check}"
    
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return result

def transpose(A):
    """Transpose matrix A (m x n) -> (n x m)"""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax(vector):
    """Softmax activation for a 1D vector."""
    max_val = max(vector) # subtract max to prevent overflow
    exps = [math.exp(x - max_val) for x in vector]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def layernorm(X, gamma=1.0, beta=0.0, eps=1e-5):
    """Layer Normalization applied row-by-row on a 2D matrix X."""
    n_rows = len(X)
    n_cols = len(X[0])
    X_norm = [[0.0] * n_cols for _ in range(n_rows)]
    
    for i in range(n_rows):
        mean = sum(X[i]) / n_cols
        variance = sum((x - mean)**2 for x in X[i]) / n_cols
        std = math.sqrt(variance + eps)
        for j in range(n_cols):
            X_norm[i][j] = gamma * ((X[i][j] - mean) / std) + beta
    return X_norm

def relu(x):
    return max(0.0, x)

# 2. Positional Encoding Generator
def get_positional_encoding(seq_len, d_model):
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Sine wave for even indices
            div_term = math.exp(i * -math.log(10000.0) / d_model)
            pe[pos][i] = math.sin(pos * div_term)
            if i + 1 < d_model:
                # Cosine wave for odd indices
                pe[pos][i+1] = math.cos(pos * div_term)
    return pe

# Initialize input parameters
seq_len = 6      # E.g. "deep learning runs neural networks" (tokenized)
d_model = 8      # Dimension of embeddings
num_heads = 2    # Heads in multi-head attention
d_k = d_model // num_heads # 4 dimensions per head

# Generate dummy input sequence embeddings (X)
X_embeddings = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(seq_len)]

print(f"\n1. Inputs dimensions:")
print(f" - Sequence Length: {seq_len} tokens")
print(f" - Embeddings Dim (d_model): {d_model}")
print(f" - Multi-head attention heads: {num_heads} (d_k: {d_k})")

# Generate and add Positional Encodings
PE = get_positional_encoding(seq_len, d_model)
X_input = [[X_embeddings[i][j] + PE[i][j] for j in range(d_model)] for i in range(seq_len)]
print("\nAdded Positional Encodings to input token embeddings.")

# 3. Multi-Head Attention forward pass
# Initialize projection weights Q, K, V (d_model x d_model)
W_Q = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_K = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_V = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_O = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]

# Compute Q, K, V projections
Q = matmul(X_input, W_Q)
K = matmul(X_input, W_K)
V = matmul(X_input, W_V)

# Split heads and perform attention calculations
head_outputs = []
for h in range(num_heads):
    # Extract slices for each head (seq_len x d_k)
    q_h = [[row[h * d_k + d] for d in range(d_k)] for row in Q]
    k_h = [[row[h * d_k + d] for d in range(d_k)] for row in K]
    v_h = [[row[h * d_k + d] for d in range(d_k)] for row in V]
    
    # Compute attention scores: Q * K^T / sqrt(d_k)
    k_h_T = transpose(k_h)
    scores = matmul(q_h, k_h_T)
    # Scale scores
    scale = math.sqrt(d_k)
    scaled_scores = [[scores[i][j] / scale for j in range(seq_len)] for i in range(seq_len)]
    
    # Softmax row-by-row to get attention weights
    attn_weights = [softmax(row) for row in scaled_scores]
    
    # Compute attention output: Weights * V
    output_h = matmul(attn_weights, v_h)
    head_outputs.append(output_h)
    
# Concatenate head outputs (reconstruct seq_len x d_model matrix)
concat_attn = [[0.0] * d_model for _ in range(seq_len)]
for i in range(seq_len):
    for h in range(num_heads):
        for d in range(d_k):
            concat_attn[i][h * d_k + d] = head_outputs[h][i][d]

# Final linear projection
self_attention_out = matmul(concat_attn, W_O)

# LayerNorm & Residual Connection 1
X_attn_norm = layernorm([[X_input[i][j] + self_attention_out[i][j] for j in range(d_model)] for i in range(seq_len)])

print("\n2. Self-Attention layer finished:")
print(f" - Attention output shape: {len(self_attention_out)} x {len(self_attention_out[0])}")

# 4. Feed-Forward Network Layer
# Hidden layer dimension = 16 (d_ff = 2 * d_model)
d_ff = 16
W_1 = [[random.uniform(-0.1, 0.1) for _ in range(d_ff)] for _ in range(d_model)]
b_1 = [random.uniform(-0.1, 0.1) for _ in range(d_ff)]
W_2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_ff)]
b_2 = [random.uniform(-0.1, 0.1) for _ in range(d_model)]

ffn_out = [[0.0] * d_model for _ in range(seq_len)]
for i in range(seq_len):
    # Linear 1: x * W_1 + b_1
    h_layer = [sum(X_attn_norm[i][k] * W_1[k][j] for k in range(d_model)) + b_1[j] for j in range(d_ff)]
    # Activation: ReLU
    h_relu = [relu(val) for val in h_layer]
    # Linear 2: relu_out * W_2 + b_2
    for j in range(d_model):
        ffn_out[i][j] = sum(h_relu[k] * W_2[k][j] for k in range(d_ff)) + b_2[j]

# LayerNorm & Residual Connection 2
encoder_output = layernorm([[X_attn_norm[i][j] + ffn_out[i][j] for j in range(d_model)] for i in range(seq_len)])

print("\n3. Feed-Forward Network layer finished:")
print(f" - Final Encoder Output shape: {len(encoder_output)} x {len(encoder_output[0])}")
print(" - Sample output (first token representation):")
print(f"   { [round(v, 4) for v in encoder_output[0]] }")

# 5. Generate and save visualization plot (Positional Encoding Waveforms Heatmap)
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 6))

# Generate larger Positional Encoding matrix for visualization clarity
viz_seq_len = 32
viz_d_model = 16
PE_viz = get_positional_encoding(viz_seq_len, viz_d_model)

plt.imshow(PE_viz, cmap="RdYlBu", aspect="auto", interpolation="nearest")
plt.colorbar(label="Encoding Value")
plt.title("Transformer Positional Encodings (Sine & Cosine Patterns)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Embedding Dimensions (d_model)", fontsize=11, labelpad=10)
plt.ylabel("Token Sequence Position (pos)", fontsize=11, labelpad=10)
plt.tight_layout()

# Save the plot
plot_path = "plots/llm_3_positional_encoding.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")
