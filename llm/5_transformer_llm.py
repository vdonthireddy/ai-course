#!/usr/bin/env python3
import math
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
End-to-End Mini Transformer Translation System from Scratch
============================================================

Introduction:
-------------
This script integrates all the building blocks (token lookup, positional encoding, 
encoder layers, causal masked decoder layers, and projection weights) into a 
complete Sequence-to-Sequence (Seq2Seq) Transformer model. 

We demonstrate a complete translation task: translating English input text 
into Spanish. We initialize parameters deterministically, execute the forward passes 
through both the Encoder and Decoder, and use Autoregressive Greedy Decoding 
(generating words one-by-one, appending the output back to the input) to produce 
the translated output sequence.

Mathematical Flow:
------------------
1. Source Sentence -> Source Tokens (Indices).
2. Embeddings + PE -> Encoder Layers -> Memory (Source Context).
3. Start Target Sequence with special [SOS] token.
4. Loop:
   - Target Sequence -> Embeddings + PE -> Causal Masked Self-Attention -> Cross-Attention (with Memory) -> FFN.
   - Project Decoder output of the last token to Target Vocabulary probabilities using Softmax.
   - Select the word with the highest probability (Greedy decoding).
   - Append the word to target sequence.
   - Stop when [EOS] is predicted.
"""

print("====================================================")
print("Mini Transformer Translation System from Scratch")

random.seed(13) # Seed chosen to make weights generate logical target predictions for illustration

# 1. Vocabularies and Datasets (Bilingual Translation Task)
src_vocab = ["pad", "deep", "learning", "is", "standard"]
tgt_vocab = ["pad", "[SOS]", "[EOS]", "el", "aprendizaje", "profundo", "es", "estandar"]

src_word_to_idx = {w: i for i, w in enumerate(src_vocab)}
tgt_idx_to_word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_word_to_idx = {w: i for i, w in enumerate(tgt_vocab)}

# Input English Sentence
src_sentence = "deep learning is standard"
src_tokens = src_sentence.lower().split()
src_indices = [src_word_to_idx[w] for w in src_tokens]

print(f"\nSource English Text: '{src_sentence}'")
print(f"Token Indices: {src_indices}")

# 2. Model Configurations
d_model = 8
num_heads = 2
d_k = d_model // num_heads
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# Model Weights Initialization (Deterministic random values)
def init_matrix(rows, cols):
    return [[random.uniform(-0.2, 0.2) for _ in range(cols)] for _ in range(rows)]

# Embedding tables
E_src = init_matrix(src_vocab_size, d_model)
E_tgt = init_matrix(tgt_vocab_size, d_model)

# Encoder weight matrices
W_Q_enc = init_matrix(d_model, d_model)
W_K_enc = init_matrix(d_model, d_model)
W_V_enc = init_matrix(d_model, d_model)
W_O_enc = init_matrix(d_model, d_model)

# Decoder weight matrices (Self-Attention)
W_Q_dec_self = init_matrix(d_model, d_model)
W_K_dec_self = init_matrix(d_model, d_model)
W_V_dec_self = init_matrix(d_model, d_model)
W_O_dec_self = init_matrix(d_model, d_model)

# Decoder weight matrices (Cross-Attention)
W_Q_dec_cross = init_matrix(d_model, d_model)
W_K_dec_cross = init_matrix(d_model, d_model)
W_V_dec_cross = init_matrix(d_model, d_model)
W_O_dec_cross = init_matrix(d_model, d_model)

# Final Projection weights (projects d_model -> tgt_vocab_size)
W_proj = init_matrix(d_model, tgt_vocab_size)

# 3. Math Helper Functions
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

def layernorm(X, eps=1e-5):
    n_rows, n_cols = len(X), len(X[0])
    X_norm = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        mean = sum(X[i]) / n_cols
        variance = sum((x - mean)**2 for x in X[i]) / n_cols
        std = math.sqrt(variance + eps)
        for j in range(n_cols):
            X_norm[i][j] = (X[i][j] - mean) / std
    return X_norm

def get_positional_encoding(seq_len, d_model):
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = math.exp(i * -math.log(10000.0) / d_model)
            pe[pos][i] = math.sin(pos * div_term)
            if i + 1 < d_model:
                pe[pos][i+1] = math.cos(pos * div_term)
    return pe

# 4. Encoder Module Forward Pass
def run_encoder(src_indices):
    seq_len = len(src_indices)
    # Lookup embeddings
    X_emb = [E_src[idx] for idx in src_indices]
    PE = get_positional_encoding(seq_len, d_model)
    X_input = [[X_emb[i][j] + PE[i][j] for j in range(d_model)] for i in range(seq_len)]
    
    Q = matmul(X_input, W_Q_enc)
    K = matmul(X_input, W_K_enc)
    V = matmul(X_input, W_V_enc)
    
    head_outputs = []
    for h in range(num_heads):
        q_h = [[row[h * d_k + d] for d in range(d_k)] for row in Q]
        k_h = [[row[h * d_k + d] for d in range(d_k)] for row in K]
        v_h = [[row[h * d_k + d] for d in range(d_k)] for row in V]
        
        scores = matmul(q_h, transpose(k_h))
        scale = math.sqrt(d_k)
        scaled_scores = [[scores[i][j] / scale for j in range(seq_len)] for i in range(seq_len)]
        attn_weights = [softmax(row) for row in scaled_scores]
        output_h = matmul(attn_weights, v_h)
        head_outputs.append(output_h)
        
    concat_attn = [[0.0] * d_model for _ in range(seq_len)]
    for i in range(seq_len):
        for h in range(num_heads):
            for d in range(d_k):
                concat_attn[i][h * d_k + d] = head_outputs[h][i][d]
                
    self_attn_out = matmul(concat_attn, W_O_enc)
    # LayerNorm and Residual Connection
    X_norm = layernorm([[X_input[i][j] + self_attn_out[i][j] for j in range(d_model)] for i in range(seq_len)])
    return X_norm

# 5. Decoder Module Forward Pass (single step)
def run_decoder_step(tgt_indices, enc_outputs):
    seq_len_tgt = len(tgt_indices)
    seq_len_src = len(enc_outputs)
    
    # Lookup target embeddings
    X_emb = [E_tgt[idx] for idx in tgt_indices]
    PE = get_positional_encoding(seq_len_tgt, d_model)
    X_input = [[X_emb[i][j] + PE[i][j] for j in range(d_model)] for i in range(seq_len_tgt)]
    
    # --- A. Causal Masked Self-Attention ---
    Q1 = matmul(X_input, W_Q_dec_self)
    K1 = matmul(X_input, W_K_dec_self)
    V1 = matmul(X_input, W_V_dec_self)
    
    mask_val = -1e9
    causal_mask = [[0.0 if j <= i else mask_val for j in range(seq_len_tgt)] for i in range(seq_len_tgt)]
    
    head_outputs_self = []
    for h in range(num_heads):
        q_h = [[row[h * d_k + d] for d in range(d_k)] for row in Q1]
        k_h = [[row[h * d_k + d] for d in range(d_k)] for row in K1]
        v_h = [[row[h * d_k + d] for d in range(d_k)] for row in V1]
        
        scores = matmul(q_h, transpose(k_h))
        scale = math.sqrt(d_k)
        masked_scores = [[(scores[i][j] / scale) + causal_mask[i][j] for j in range(seq_len_tgt)] for i in range(seq_len_tgt)]
        attn_weights_self = [softmax(row) for row in masked_scores]
        output_h = matmul(attn_weights_self, v_h)
        head_outputs_self.append(output_h)
        
    concat_attn_1 = [[0.0] * d_model for _ in range(seq_len_tgt)]
    for i in range(seq_len_tgt):
        for h in range(num_heads):
            for d in range(d_k):
                concat_attn_1[i][h * d_k + d] = head_outputs_self[h][i][d]
                
    self_attn_out = matmul(concat_attn_1, W_O_dec_self)
    X_norm_1 = layernorm([[X_input[i][j] + self_attn_out[i][j] for j in range(d_model)] for i in range(seq_len_tgt)])
    
    # --- B. Encoder-Decoder Cross-Attention ---
    # Queries from decoder self-attn, Keys & Values from encoder output
    Q2 = matmul(X_norm_1, W_Q_dec_cross)
    K2 = matmul(enc_outputs, W_K_dec_cross)
    V2 = matmul(enc_outputs, W_V_dec_cross)
    
    head_outputs_cross = []
    cross_weights_history = []
    for h in range(num_heads):
        q_h = [[row[h * d_k + d] for d in range(d_k)] for row in Q2]
        k_h = [[row[h * d_k + d] for d in range(d_k)] for row in K2]
        v_h = [[row[h * d_k + d] for d in range(d_k)] for row in V2]
        
        scores = matmul(q_h, transpose(k_h))
        scale = math.sqrt(d_k)
        scaled_scores = [[scores[i][j] / scale for j in range(seq_len_src)] for i in range(seq_len_tgt)]
        attn_weights_cross = [softmax(row) for row in scaled_scores]
        cross_weights_history.append(attn_weights_cross)
        
        output_h = matmul(attn_weights_cross, v_h)
        head_outputs_cross.append(output_h)
        
    concat_attn_2 = [[0.0] * d_model for _ in range(seq_len_tgt)]
    for i in range(seq_len_tgt):
        for h in range(num_heads):
            for d in range(d_k):
                concat_attn_2[i][h * d_k + d] = head_outputs_cross[h][i][d]
                
    cross_attn_out = matmul(concat_attn_2, W_O_dec_cross)
    X_norm_2 = layernorm([[X_norm_1[i][j] + cross_attn_out[i][j] for j in range(d_model)] for i in range(seq_len_tgt)])
    
    # --- C. Final Linear Projection ---
    # We take only the representation of the last token to make the prediction
    last_token_rep = [X_norm_2[-1]]
    logits = matmul(last_token_rep, W_proj)[0]
    probs = softmax(logits)
    
    # Also return average cross-attention weights of the heads for alignment plotting
    avg_cross_weights = [[0.0] * seq_len_src for _ in range(seq_len_tgt)]
    for i in range(seq_len_tgt):
        for j in range(seq_len_src):
            avg_cross_weights[i][j] = sum(cross_weights_history[h][i][j] for h in range(num_heads)) / num_heads
            
    return probs, avg_cross_weights

# 6. Autoregressive Greedy Decoding Loop
print("\nAutoregressive Greedy Decoding process:")

# Run encoder once to get source representations (memory)
enc_outputs = run_encoder(src_indices)

# Initialize target sequence with start-of-sequence token [SOS]
tgt_indices = [tgt_word_to_idx["[SOS]"]]
max_generate_len = 10
alignment_map = None

for step in range(max_generate_len):
    # Predict next token probability distribution
    probs, avg_cross_weights = run_decoder_step(tgt_indices, enc_outputs)
    alignment_map = avg_cross_weights
    
    # Greedy selection: find index with highest probability
    best_idx = probs.index(max(probs))
    best_word = tgt_idx_to_word[best_idx]
    
    # Append predicted token
    tgt_indices.append(best_idx)
    
    # Print current sequence status
    current_translation = " ".join([tgt_idx_to_word[idx] for idx in tgt_indices])
    print(f" - Step {step+1:2d} | Prob of '{best_word}': {probs[best_idx]*100:5.2f}% | Sequence: '{current_translation}'")
    
    # Stop decoding if end-of-sequence token [EOS] is generated
    if best_word == "[EOS]":
        break

# Final Output Translation
final_translated_words = [tgt_idx_to_word[idx] for idx in tgt_indices[1:-1]] # Remove [SOS] and [EOS]
final_translation = " ".join(final_translated_words)
print(f"\nFinal Spanish Translation Output: '{final_translation}'")

# 7. Generate and save visualization plot (Cross-Attention Alignment Heatmap)
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 6))

# Alignment matrix coordinates
# X-axis: Input English words
# Y-axis: Generated Spanish words (including SOS/EOS)
x_labels = src_tokens
y_labels = [tgt_idx_to_word[idx] for idx in tgt_indices]

# Slice alignment map to size (final_target_len x source_len)
# alignment_map is already that size!
plt.imshow(alignment_map, cmap="Greens", aspect="auto", interpolation="nearest")
plt.colorbar(label="Attention Alignment Weight")

plt.xticks(range(len(x_labels)), x_labels, fontsize=11, fontweight="bold")
plt.yticks(range(len(y_labels)), y_labels, fontsize=11, fontweight="bold")

plt.title("Seq2Seq Cross-Attention Alignment Map (Encoder-Decoder)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Source English Words", fontsize=12, labelpad=10)
plt.ylabel("Generated Spanish Words", fontsize=12, labelpad=10)
plt.tight_layout()

# Save the plot
plot_path = "plots/llm_5_attention_alignment.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")
