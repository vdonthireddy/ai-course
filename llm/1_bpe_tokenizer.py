#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Byte Pair Encoding (BPE) Tokenizer from Scratch
==============================================

Introduction:
-------------
BPE is a subword tokenization algorithm widely used in LLMs (e.g., GPT, LLaMA). 
It starts with a vocabulary of individual characters and iteratively merges the 
most frequent adjacent token pairs in a text corpus to form new subwords. This 
allows the tokenizer to handle rare and out-of-vocabulary words by breaking them 
into subword chunks, balancing vocabulary size and sequence length.

Mathematical Logic:
-------------------
1. **Initial Vocabulary**: All individual characters present in the training corpus + end-of-word marker `</w>`.
2. **Frequency Count**: Count occurrences of adjacent symbol pairs in the word-frequency dictionary.
3. **Merger**: Select the pair $(s_1, s_2)$ with maximum frequency and merge it into a new token $s_{1\_2}$.
4. **Encoding (Tokenization)**: Apply the learned merge rules in order to segment new words.
"""

print("====================================================")
print("Byte Pair Encoding (BPE) Tokenizer from Scratch")

# 1. Define Training Corpus and Word Dictionary
# We represent words with spaces between characters and ending with '</w>'
corpus = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

# Extract initial vocabulary (all unique characters)
vocab = set()
for word in corpus.keys():
    for char in word.split():
        vocab.add(char)
vocab = list(vocab)

print(f"\nInitial Vocabulary ({len(vocab)} tokens):")
print(sorted(vocab))

# 2. BPE Helper Functions
def get_stats(vocab_dict):
    """Count frequencies of all adjacent symbol pairs."""
    pairs = {}
    for word, freq in vocab_dict.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs

def merge_vocab(pair, vocab_dict):
    """Merge all occurrences of the specified pair in the dictionary."""
    v_out = {}
    bigram = ' '.join(pair)
    # Escape special characters for exact matching
    replacement = ''.join(pair)
    for word in vocab_dict:
        # Find raw sequence of symbols
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = vocab_dict[word]
    return v_out

# 3. Training Loop (Merge Iterations)
num_merges = 10
merge_rules = [] # Stores learned merge operations in order
vocab_sizes = [len(vocab)]

print(f"\nStarting BPE Training ({num_merges} merge iterations)...")
current_corpus = dict(corpus)

for i in range(num_merges):
    pairs = get_stats(current_corpus)
    if not pairs:
        break
    # Find the most frequent pair
    best_pair = max(pairs, key=pairs.get)
    best_freq = pairs[best_pair]
    
    # Store merge rule
    merge_rules.append(best_pair)
    
    # Merge the pair in our corpus representation
    current_corpus = merge_vocab(best_pair, current_corpus)
    
    # Add new merged token to vocabulary
    new_token = ''.join(best_pair)
    if new_token not in vocab:
        vocab.append(new_token)
        
    vocab_sizes.append(len(vocab))
    print(f" - Iteration {i+1:2d} | Best Pair: {best_pair} (Freq: {best_freq}) -> Merged into: '{new_token}' | Vocab Size: {len(vocab)}")

print(f"\nFinal Vocabulary ({len(vocab)} tokens):")
print(sorted(vocab))

# 4. Tokenizer Encoding & Decoding Functions
def tokenize_word(word, learned_rules):
    """Tokenize a single word using BPE merge rules."""
    # Split word into characters and append end-of-word marker
    symbols = list(word) + ['</w>']
    
    # Apply merge rules in the exact order they were learned
    for pair in learned_rules:
        s_1, s_2 = pair
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == s_1 and symbols[i+1] == s_2:
                # Merge symbols
                symbols[i] = s_1 + s_2
                del symbols[i+1]
            else:
                i += 1
    return symbols

def encode(text, learned_rules):
    """Encode a sentence into BPE tokens."""
    words = text.strip().split()
    tokens = []
    for w in words:
        tokens.extend(tokenize_word(w, learned_rules))
    return tokens

def decode(tokens):
    """Decode BPE tokens back to standard text."""
    # Joining tokens and replacing '</w>' marker with space
    text = ''.join(tokens).replace('</w>', ' ')
    return text.strip()

# 5. Test Tokenizer on Out-of-Vocabulary (OOV) Words
test_words = "lowest newer wide"
print(f"\nEncoding test sentence: '{test_words}'")
encoded_tokens = encode(test_words, merge_rules)
print(f" - BPE Encoded Tokens: {encoded_tokens}")

decoded_text = decode(encoded_tokens)
print(f" - BPE Decoded Text  : '{decoded_text}'")

# 6. Generate and save visualization plot (Vocab Growth Curve)
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 5))
plt.plot(range(0, len(vocab_sizes)), vocab_sizes, marker='o', color='#4F46E5', linewidth=2.5, label="Vocabulary Size")
plt.title("BPE Vocabulary Growth Curve", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Merge Iterations", fontsize=11, labelpad=10)
plt.ylabel("Vocabulary Size (Tokens)", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(range(0, len(vocab_sizes)))
plt.legend(loc="upper left")
plt.tight_layout()

# Save the plot
plot_path = "plots/llm_1_bpe_vocab.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")
