# Walkthrough: Pure Python Adam Optimizer from Scratch

This walkthrough documents the integration of the **Adaptive Moment Estimation (Adam)** optimization algorithm implemented from scratch into the LLM Word2Vec embedding script, project documentation, and slide deck.

---

## Technical Details

### 1. From-Scratch Adam Optimizer Implementation
In `llm/2_word2vec.py`, we replaced the Stochastic Gradient Descent (SGD) update logic with a pure Python implementation of the Adam optimizer.
- **State Initialization**: Declared first moment tables (`m_in`, `m_out`) and second moment tables (`v_in`, `v_out`) as lists of zeros matching the shape of the embeddings `(vocab_size, embedding_dim)`.
- **Step Tracking**: Maintained a time step counter $t$ that increments with each parameter update.
- **Decay & Correction**: Implemented standard exponential moving averages for the gradients (first moment $m_t$) and squared gradients (second moment $v_t$), followed by bias-correction computations:
  
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

- **Parameter Updates**: Applied updates to the embedding parameters coordinate-by-coordinate:

  $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 2. Documentation Updates in `README.md`
- **Module 9.2 Section**: Added a mathematical explanation of the Adam optimizer, outlining the formulas for moving averages, bias-correction, and the parameters update step.
- **Appendix Glossary**: Added **16. Adam Optimizer** to the glossary of key terminologies. Provided a mathematical definition, a complete step-by-step example calculating values for $t=1$, and visual references to the generated training plots.
- **Summary Tables**: Updated descriptions to mention the Word2Vec script uses the Adam optimizer.

### 3. Presentation Slides Compilation
- **Slide 13 (Word2Vec)**: Updated the bullet points to show optimization is powered by the from-scratch Adam optimizer.
- **Slide 19 (Optimization & Bias)**: Added a bullet point defining the Adam optimizer and its formula, contrasting it with traditional Gradient Descent.

---

## Validation Results

- **Word2Vec Convergence**: Re-running `python3 llm/2_word2vec.py` runs successfully, prints declining losses over 500 epochs, and outputs the updated semantic word cluster plot [plots/llm_2_word2vec.png](file:///Users/donthireddy/code/ai-course/plots/llm_2_word2vec.png) with no syntax or runtime warnings.
- **Slide Deck Generation**: Re-running `python3 generate_pptx.py` generates the presentation [machine_learning_fundamentals.pptx](file:///Users/donthireddy/code/ai-course/machine_learning_fundamentals.pptx) successfully.
- **Git Sync**: Code modifications, updated slide deck, and training plot have been committed and pushed to git remote (Commit: `8dd252c`).
