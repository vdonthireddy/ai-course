#!/usr/bin/env python3
import math
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Transformer Decoder Layer from Scratch
======================================

Introduction:
-------------
The Transformer Decoder generates target sequences tokens sequentially. It features two 
kinds of Multi-Head Attention:
1. **Causal Masked Self-Attention**: Processes target tokens but masks out future positions 
   (lower-triangular masking) so the model cannot "cheat" during training by looking ahead.
2. **Encoder-Decoder Cross-Attention**: Connects the decoder to the encoder's output. The 
   Queries ($Q$) come from the decoder's causal output, while Keys ($K$) and Values ($V$) 
   come from the encoder representations, enabling alignment between source and target words.

Mathematical Logic:
-------------------
1. **Causal Masking**:
   $$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
"""

print("====================================================")
print("Transformer Decoder Layer from Scratch")

random.seed(42)

# 1. Math and Matrix Helpers
def matmul(A, B):
    m, n = len(A), len(A[0])
    n_check, p = len(B), len(B[0])
    assert n == n_check, f"Dimension mismatch: {n} != {n_check}"
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return result

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax(vector):
    max_val = max(vector)
    exps = [math.exp(x - max_val) for x in vector]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def layernorm(X, gamma=1.0, beta=0.0, eps=1e-5):
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

# 2. Dimensions Initialization
seq_len_tgt = 4   # Target sequence length (e.g. "hola", "aprendizaje", "automatico", "</w>")
seq_len_src = 6   # Encoder source sequence length
d_model = 8
num_heads = 2
d_k = d_model // num_heads

# Dummy target input (X_tgt) and encoder output (Y_enc)
X_tgt = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(seq_len_tgt)]
Y_enc = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(seq_len_src)]

print(f"\n1. Sequence configurations:")
print(f" - Target sequence length (Decoder): {seq_len_tgt} tokens")
print(f" - Source sequence length (Encoder): {seq_len_src} tokens")
print(f" - Dimension size (d_model): {d_model}")

# 3. Causal Masked Self-Attention
W_Q1 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_K1 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_V1 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_O1 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]

Q1 = matmul(X_tgt, W_Q1)
K1 = matmul(X_tgt, W_K1)
V1 = matmul(X_tgt, W_V1)

# Causal Mask (lower triangular matrix)
# Future tokens have value -1e9 (acting as -infinity)
mask_val = -1e9
causal_mask = [[0.0 if j <= i else mask_val for j in range(seq_len_tgt)] for i in range(seq_len_tgt)]

head_outputs_1 = []
for h in range(num_heads):
    q_h = [[row[h * d_k + d] for d in range(d_k)] for row in Q1]
    k_h = [[row[h * d_k + d] for d in range(d_k)] for row in K1]
    v_h = [[row[h * d_k + d] for d in range(d_k)] for row in V1]
    
    # Q * K^T / sqrt(d_k)
    scores = matmul(q_h, transpose(k_h))
    scale = math.sqrt(d_k)
    
    # Add Causal Mask to scores
    masked_scores = [[(scores[i][j] / scale) + causal_mask[i][j] for j in range(seq_len_tgt)] for i in range(seq_len_tgt)]
    
    # Softmax row-by-row
    attn_weights = [softmax(row) for row in masked_scores]
    output_h = matmul(attn_weights, v_h)
    head_outputs_1.append(output_h)

# Concatenate & Project
concat_attn_1 = [[0.0] * d_model for _ in range(seq_len_tgt)]
for i in range(seq_len_tgt):
    for h in range(num_heads):
        for d in range(d_k):
            concat_attn_1[i][h * d_k + d] = head_outputs_1[h][i][d]

self_attn_out = matmul(concat_attn_1, W_O1)
# LayerNorm + Residual 1
X_attn_norm_1 = layernorm([[X_tgt[i][j] + self_attn_out[i][j] for j in range(d_model)] for i in range(seq_len_tgt)])

print("\n2. Causal Self-Attention finished.")
print(" - Causal Attention Weight sample row 2 (position 2 can attend only to positions 0, 1, 2):")
print(f"   { [round(w, 4) for w in attn_weights[2]] }")

# 4. Encoder-Decoder Cross-Attention
# Query (Q) comes from target (X_attn_norm_1)
# Keys (K) and Values (V) come from encoder (Y_enc)
W_Q2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_K2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_V2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
W_O2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]

Q2 = matmul(X_attn_norm_1, W_Q2)
K2 = matmul(Y_enc, W_K2)
V2 = matmul(Y_enc, W_V2)

head_outputs_2 = []
for h in range(num_heads):
    q_h = [[row[h * d_k + d] for d in range(d_k)] for row in Q2]
    k_h = [[row[h * d_k + d] for d in range(d_k)] for row in K2]
    v_h = [[row[h * d_k + d] for d in range(d_k)] for row in V2]
    
    # Compute Cross-attention (No causal masking here!)
    scores = matmul(q_h, transpose(k_h))
    scale = math.sqrt(d_k)
    scaled_scores = [[scores[i][j] / scale for j in range(seq_len_src)] for i in range(seq_len_tgt)]
    
    attn_weights_cross = [softmax(row) for row in scaled_scores]
    output_h = matmul(attn_weights_cross, v_h)
    head_outputs_2.append(output_h)

# Concatenate & Project
concat_attn_2 = [[0.0] * d_model for _ in range(seq_len_tgt)]
for i in range(seq_len_tgt):
    for h in range(num_heads):
        for d in range(d_k):
            concat_attn_2[i][h * d_k + d] = head_outputs_2[h][i][d]

cross_attn_out = matmul(concat_attn_2, W_O2)
# LayerNorm + Residual 2
X_attn_norm_2 = layernorm([[X_attn_norm_1[i][j] + cross_attn_out[i][j] for j in range(d_model)] for i in range(seq_len_tgt)])

print("\n3. Cross-Attention layer finished:")
print(f" - Cross-Attention output shape: {len(cross_attn_out)} x {len(cross_attn_out[0])}")

# 5. Feed-Forward Layer & LayerNorm + Residual 3
d_ff = 16
W_1 = [[random.uniform(-0.1, 0.1) for _ in range(d_ff)] for _ in range(d_model)]
b_1 = [random.uniform(-0.1, 0.1) for _ in range(d_ff)]
W_2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_ff)]
b_2 = [random.uniform(-0.1, 0.1) for _ in range(d_model)]

ffn_out = [[0.0] * d_model for _ in range(seq_len_tgt)]
for i in range(seq_len_tgt):
    h_layer = [sum(X_attn_norm_2[i][k] * W_1[k][j] for k in range(d_model)) + b_1[j] for j in range(d_ff)]
    h_relu = [relu(val) for val in h_layer]
    for j in range(d_model):
        ffn_out[i][j] = sum(h_relu[k] * W_2[k][j] for k in range(d_ff)) + b_2[j]

decoder_output = layernorm([[X_attn_norm_2[i][j] + ffn_out[i][j] for j in range(d_model)] for i in range(seq_len_tgt)])

print("\n4. Feed-Forward layer finished:")
print(f" - Final Decoder Output shape: {len(decoder_output)} x {len(decoder_output[0])}")
print(" - Sample output (first token representation):")
print(f"   { [round(v, 4) for v in decoder_output[0]] }")

# 6. Generate and save visualization plot (Causal Mask Heatmap)
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(7, 6))

# Let's create a larger causal mask matrix for visualization (e.g. 8x8 tokens)
viz_len = 8
mask_grid = [[1.0 if j <= i else 0.0 for j in range(viz_len)] for i in range(viz_len)]

plt.imshow(mask_grid, cmap="Blues", interpolation="nearest")
plt.colorbar(label="Attention Allowed (1 = Yes, 0 = Masked)")
plt.title("Causal Self-Attention Lower-Triangular Mask", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Key/Value Token Indices (Past)", fontsize=11, labelpad=10)
plt.ylabel("Query Token Indices (Present)", fontsize=11, labelpad=10)

# Set exact ticks
plt.xticks(range(viz_len))
plt.yticks(range(viz_len))

plt.tight_layout()

# Save the plot
plot_path = "plots/llm_4_causal_mask.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")
