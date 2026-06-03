# Implementation Plan: Adam Optimizer from Scratch in LLM Module

This plan details the addition of the **Adam (Adaptive Moment Estimation) optimizer** written from scratch in pure Python for the LLM module (`llm/2_word2vec.py`). This will replace SGD with Adam, ensuring faster, more stable convergence of the 2D word embeddings, and matching modern optimization practices in Large Language Models.

---

## User Review Required

> [!IMPORTANT]
> - **From-Scratch Adam Math**: The Adam algorithm will be written in pure Python without using libraries. We will explicitly initialize the first ($m$) and second ($v$) moment tables for all input and output embedding weights and compute updates using standard loop calculations.
> - **Hyperparameters**: We will set standard Adam parameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, learning rate $\eta = 0.02$ (tuned for 2D convergence), and $\epsilon = 10^{-8}$.
> - **Visual Validation**: The output plot `plots/llm_2_word2vec.png` will still show the loss curve and learned clusters, but now showing faster convergence.
> - **Slide Deck & README Updates**: The slide deck generator and `README.md` will be updated to include Adam optimizer equations and description.

---

## Proposed Changes

We will modify:

### 1. `llm/2_word2vec.py`
#### [MODIFY] [2_word2vec.py](file:///Users/donthireddy/code/ai-course/llm/2_word2vec.py)
- Update the intro comment equations and parameter explanations to reflect Adam updates.
- Initialize `m_in`, `v_in`, `m_out`, `v_out` as tables of zeros corresponding to input/output embedding shapes.
- Track step counter $t$ inside the training loop.
- Perform Adam moment accumulations, bias-corrections, and parameter updates for positive context word weights, negative sample weights, and target input word weights.
- Change learning rate default to `0.02` (suitable for Adam here).

### 2. `README.md`
#### [MODIFY] [README.md](file:///Users/donthireddy/code/ai-course/README.md)
- Add a new subsection `#### Optimization: The Adam Optimizer from Scratch` under section `9.2 Vector Semantics: Word Embeddings (Word2Vec)`. Show Adam's equations for moments, bias-correction, and weight update.
- Update reference to `llm/2_word2vec.py` at line 1144.
- Add a new term `16. Adam Optimizer` to the Appendix Glossary with its mathematical definition, step-by-step calculation example, and visual diagram.

### 3. `generate_pptx.py`
#### [MODIFY] [generate_pptx.py](file:///Users/donthireddy/code/ai-course/generate_pptx.py)
- Update Slide 13 (Module 9.2: Word2Vec) bullets to explain that embeddings are trained using the Adam optimizer.
- Add Slide 21 (Appendix: Optimization & Adam) to display Adam equations and explanation, adjusting Slide 21/22 indexing as needed.

---

## Verification Plan

### Automated Tests
- Run `python3 llm/2_word2vec.py` to verify that the Adam training loop executes correctly, prints descending BCE losses, clusters semantically related words, and outputs the updated plots.
- Run `python3 generate_pptx.py` to regenerate the presentation.
- Run validation commands on all python files to ensure they build without errors.

### Manual Verification
- Review the generated slide deck `machine_learning_fundamentals.pptx` to verify that the Adam optimizer content is properly laid out.
