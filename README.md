# Fundamentals of Machine Learning: A Comprehensive Curriculum

Welcome to the **Fundamentals of Machine Learning** course materials. This document serves as a complete, step-by-step teaching guide designed to introduce students to the core paradigms, algorithms, mathematics, and practical implementations of machine learning.

## Table of Contents
- [Course Roadmap](#course-roadmap)
- [Runnable Code Examples](#runnable-code-examples)
  - [1. Framework Examples](#1-framework-examples-using-scikit-learn--xgboost)
  - [2. From-Scratch Examples](#2-from-scratch-examples-pure-python-no-frameworks)
- [Module 1: Introduction to Machine Learning](#module-1-introduction-to-machine-learning)
  - [1.1 What is Machine Learning?](#11-what-is-machine-learning)
  - [1.2 The Machine Learning Pipeline](#12-the-machine-learning-pipeline)
- [Module 2: The Four Learning Paradigms](#module-2-the-four-learning-paradigms)
  - [2.2 Case Study: Where do Large Language Models (LLMs) fit?](#22-case-study-where-do-large-language-models-llms-fit)
- [Module 3: Supervised Learning - Regression](#module-3-supervised-learning---regression)
  - [3.1 Simple and Multiple Linear Regression](#31-simple-and-multiple-linear-regression)
  - [3.2 Regularization: Ridge & Lasso](#32-regularization-ridge--lasso)
  - [3.3 Python Implementation](#33-python-implementation)
- [Module 4: Supervised Learning - Classification](#module-4-supervised-learning---classification)
  - [4.1 Logistic Regression](#41-logistic-regression)
  - [4.2 K-Nearest Neighbors (KNN)](#42-k-nearest-neighbors-knn)
  - [4.3 Support Vector Machines (SVM)](#43-support-vector-machines-svm)
  - [4.4 Naive Bayes](#44-naive-bayes)
  - [4.5 Decision Trees & Random Forests](#45-decision-trees--random-forests)
  - [4.6 Python Implementation](#46-python-implementation)
- [Module 5: Ensemble Learning & XGBoost](#module-5-ensemble-learning--xgboost)
  - [5.1 Bagging vs. Boosting](#51-bagging-vs-boosting)
  - [5.2 XGBoost (Extreme Gradient Boosting)](#52-xgboost-extreme-gradient-boosting)
  - [5.3 Python Implementation](#53-python-implementation)
- [Module 6: Unsupervised Learning - Clustering](#module-6-unsupervised-learning---clustering)
  - [6.1 K-Means Clustering](#61-k-means-clustering)
  - [6.2 Hierarchical Clustering](#62-hierarchical-clustering)
  - [6.3 DBSCAN](#63-dbscan)
  - [6.4 Python Implementation](#64-python-implementation)
- [Module 7: Evaluation & Validation](#module-7-evaluation--validation)
  - [7.1 Bias-Variance Tradeoff](#71-bias-variance-tradeoff)
  - [7.2 Cross-Validation](#72-cross-validation)
  - [7.3 Performance Metrics](#73-performance-metrics)
- [Module 8: Code Walkthroughs & Implementation Steps](#module-8-code-walkthroughs--implementation-steps)
  - [8.1 Regression (Housing Prices Example)](#81-regression-housing-prices-example)
  - [8.2 Classification (Student Exam Pass/Fail)](#82-classification-student-exam-passfail)
  - [8.3 K-Nearest Neighbors (Offer Acceptance)](#83-k-nearest-neighbors-offer-acceptance)
  - [8.4 Decision Trees & Random Forests (Churn)](#84-decision-trees--random-forests-churn)
  - [8.5 Ensemble Boosting (XGBoost Loan Defaults)](#85-ensemble-boosting-xgboost-loan-defaults)
  - [8.6 K-Means Clustering (Customer Segments)](#86-kmeans-clustering-customer-segments)
  - [8.7 DBSCAN Density Clustering (Outlier Detection)](#87-dbscan-density-clustering-outlier-detection)
  - [8.8 Model Evaluation & Performance Metrics](#88-model-evaluation--performance-metrics)
- [Module 9: Introduction to Large Language Models (LLMs) & Transformers](#module-9-introduction-to-large-language-models-llms--transformers)
  - [9.1 Tokenization: Byte Pair Encoding (BPE)](#91-tokenization-byte-pair-encoding-bpe)
  - [9.2 Vector Semantics: Word Embeddings (Word2Vec)](#92-vector-semantics-word-embeddings-word2vec)
  - [9.3 The Encoder: Self-Attention & Positional Encoding](#93-the-encoder-self-attention--positional-encoding)
  - [9.4 The Decoder: Causal Masking & Cross-Attention](#94-the-decoder-causal-masking--cross-attention)
  - [9.5 End-to-End LLM Python Implementation](#95-end-to-end-llm-python-implementation)
- [Appendix: Glossary of Key Terminologies](#appendix-glossary-of-key-terminologies)

---

## Course Roadmap

```mermaid
graph TD
    A[Introduction & ML Pipeline] --> B[ML Paradigms]
    B --> C[Supervised Learning]
    B --> D[Unsupervised Learning]
    B --> E[Semi-Supervised & Reinforcement]
    
    C --> C1[Regression: Linear, Ridge, Lasso]
    C --> C2[Classification: Logistic, KNN, SVM, Naive Bayes, Trees]
    C --> C3[Ensembles & XGBoost]
    
    D --> D1[Clustering: K-Means, Hierarchical, DBSCAN]
    
    C1 & C2 & C3 & D1 --> F[Model Evaluation & Validation]
    style A fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#FFF
    style B fill:#0EA5E9,stroke:#0369A1,stroke-width:2px,color:#FFF
    style C fill:#10B981,stroke:#047857,stroke-width:2px,color:#FFF
    style D fill:#F59E0B,stroke:#B45309,stroke-width:2px,color:#FFF
    style F fill:#EF4444,stroke:#B91C1C,stroke-width:2px,color:#FFF
```

---

## Runnable Code Examples

This course features two sets of runnable, self-contained Python examples for each core concept:

1. **Framework Examples (`examples/`)**: Standard implementations using industry-standard libraries like `scikit-learn` and `xgboost`.
2. **From-Scratch Examples (`scratch/`)**: Pure Python implementations of the same subjects written without *any* external ML libraries to teach the mathematical inner workings step-by-step.

### 1. Framework Examples (using Scikit-Learn / XGBoost)
You can run them in your terminal using:
```bash
python3 examples/<script_name>.py
```

| Lesson / Topic | Concept | Python Example Link | Output Plot Link |
| :--- | :--- | :--- | :--- |
| **Module 3: Regression** | House Value per Sq Ft | [1_regression_housing.py](file:///Users/donthireddy/code/ai-course/examples/1_regression_housing.py) | [examples_1_regression.png](file:///Users/donthireddy/code/ai-course/plots/examples_1_regression.png) |
| **Module 4: Classification** | Student Exam Pass/Fail | [2_classification_exam.py](file:///Users/donthireddy/code/ai-course/examples/2_classification_exam.py) | [examples_2_classification.png](file:///Users/donthireddy/code/ai-course/plots/examples_2_classification.png) |
| **Module 4: KNN** | Credit Card Offer Acceptance | [3_knn_classification.py](file:///Users/donthireddy/code/ai-course/examples/3_knn_classification.py) | [examples_3_knn.png](file:///Users/donthireddy/code/ai-course/plots/examples_3_knn.png) |
| **Module 4: Decision Trees** | Customer Churn tree rules & Random Forest | [4_decision_tree_rf.py](file:///Users/donthireddy/code/ai-course/examples/4_decision_tree_rf.py) | [examples_4_decision_tree.png](file:///Users/donthireddy/code/ai-course/plots/examples_4_decision_tree.png) |
| **Module 5: Ensemble** | Loan Default risk forecasting with XGBoost | [5_xgboost_ensemble.py](file:///Users/donthireddy/code/ai-course/examples/5_xgboost_ensemble.py) | [examples_5_xgboost.png](file:///Users/donthireddy/code/ai-course/plots/examples_5_xgboost.png) |
| **Module 6: K-Means** | Customer Income & Spend Segmentation | [6_kmeans_clustering.py](file:///Users/donthireddy/code/ai-course/examples/6_kmeans_clustering.py) | [examples_6_kmeans.png](file:///Users/donthireddy/code/ai-course/plots/examples_6_kmeans.png) |
| **Module 6: DBSCAN** | Density Coordinate Point Classification | [7_dbscan_clustering.py](file:///Users/donthireddy/code/ai-course/examples/7_dbscan_clustering.py) | [examples_7_dbscan.png](file:///Users/donthireddy/code/ai-course/plots/examples_7_dbscan.png) |
| **Module 7: Evaluation** | MAE/MSE/RMSE, Confusion Matrix, Precision/Recall | [8_model_evaluation.py](file:///Users/donthireddy/code/ai-course/examples/8_model_evaluation.py) | [examples_8_evaluation.png](file:///Users/donthireddy/code/ai-course/plots/examples_8_evaluation.png) |

### 2. From-Scratch Examples (Pure Python, No Frameworks)
You can run them in your terminal using:
```bash
python3 scratch/<script_name>.py
```

| Lesson / Topic | Concept | From-Scratch Python Link | Output Plot Link |
| :--- | :--- | :--- | :--- |
| **Module 3: Regression** | Gradient Descent Line Fitting | [1_linear_regression.py](file:///Users/donthireddy/code/ai-course/scratch/1_linear_regression.py) | [scratch_1_regression.png](file:///Users/donthireddy/code/ai-course/plots/scratch_1_regression.png) |
| **Module 4: Classification** | Sigmoid & Log Loss updates | [2_logistic_regression.py](file:///Users/donthireddy/code/ai-course/scratch/2_logistic_regression.py) | [scratch_2_logistic.png](file:///Users/donthireddy/code/ai-course/plots/scratch_2_logistic.png) |
| **Module 4: KNN** | Standard Scaling & Euclidean distances | [3_knn_classification.py](file:///Users/donthireddy/code/ai-course/scratch/3_knn_classification.py) | [scratch_3_knn.png](file:///Users/donthireddy/code/ai-course/plots/scratch_3_knn.png) |
| **Module 4: Decision Trees** | Gini Impurity split searches | [4_decision_tree.py](file:///Users/donthireddy/code/ai-course/scratch/4_decision_tree.py) | [scratch_4_decision_tree.png](file:///Users/donthireddy/code/ai-course/plots/scratch_4_decision_tree.png) |
| **Module 6: K-Means** | Distance centroid updates | [5_kmeans_clustering.py](file:///Users/donthireddy/code/ai-course/scratch/5_kmeans_clustering.py) | [scratch_5_kmeans.png](file:///Users/donthireddy/code/ai-course/plots/scratch_5_kmeans.png) |
| **Module 6: DBSCAN** | Density point neighborhood & BFS | [6_dbscan_clustering.py](file:///Users/donthireddy/code/ai-course/scratch/6_dbscan_clustering.py) | [scratch_6_dbscan.png](file:///Users/donthireddy/code/ai-course/plots/scratch_6_dbscan.png) |
| **Module 7: Evaluation** | Metrics loop logic (F1, MSE, $R^2$) | [7_model_evaluation.py](file:///Users/donthireddy/code/ai-course/scratch/7_model_evaluation.py) | [scratch_7_evaluation.png](file:///Users/donthireddy/code/ai-course/plots/scratch_7_evaluation.png) |

### 3. Large Language Model (LLM) Examples
You can run them in your terminal using:
```bash
python3 llm/<script_name>.py
```

| Lesson / Topic | Concept | Python Script Link | Output Plot Link |
| :--- | :--- | :--- | :--- |
| **Module 9.1: Tokenizer** | Byte Pair Encoding (BPE) from scratch | [1_bpe_tokenizer.py](file:///Users/donthireddy/code/ai-course/llm/1_bpe_tokenizer.py) | [llm_1_bpe_vocab.png](file:///Users/donthireddy/code/ai-course/plots/llm_1_bpe_vocab.png) |
| **Module 9.2: Embeddings** | Skip-gram Word2Vec with Negative Sampling | [2_word2vec.py](file:///Users/donthireddy/code/ai-course/llm/2_word2vec.py) | [llm_2_word2vec.png](file:///Users/donthireddy/code/ai-course/plots/llm_2_word2vec.png) |
| **Module 9.3: Encoder** | Multi-Head Self-Attention & Positional Encoding | [3_encoder.py](file:///Users/donthireddy/code/ai-course/llm/3_encoder.py) | [llm_3_positional_encoding.png](file:///Users/donthireddy/code/ai-course/plots/llm_3_positional_encoding.png) |
| **Module 9.4: Decoder** | Causal Masked Self-Attention & Cross-Attention | [4_decoder.py](file:///Users/donthireddy/code/ai-course/llm/4_decoder.py) | [llm_4_causal_mask.png](file:///Users/donthireddy/code/ai-course/plots/llm_4_causal_mask.png) |
| **Module 9.5: Transformer** | End-to-End Translator (Seq2Seq Model) | [5_transformer_llm.py](file:///Users/donthireddy/code/ai-course/llm/5_transformer_llm.py) | [llm_5_attention_alignment.png](file:///Users/donthireddy/code/ai-course/plots/llm_5_attention_alignment.png) |

---

## Module 1: Introduction to Machine Learning

### 1.1 What is Machine Learning?
Traditional programming requires a human developer to write explicit rules (code) that process inputs to produce outputs. In contrast, **Machine Learning (ML)** is a paradigm where an algorithm learns the underlying rules directly from data.

```
Traditional Programming:
[Data] + [Rules/Algorithms] ---> (Computer) ---> [Results]

Machine Learning:
[Data] + [Results/Labels]     ---> (Computer) ---> [Rules/Model]
```

### 1.2 The Machine Learning Pipeline
Every ML project follows a structured workflow:

```mermaid
flowchart LR
    A[(Data Source)] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    E --> F{Performance OK?}
    F -- Yes --> G[Model Deployment]
    F -- No --> C
    
    style A fill:#D1FAE5,stroke:#059669
    style D fill:#DBEAFE,stroke:#2563EB
    style G fill:#FEE2E2,stroke:#DC2626
```

1. **Data Acquisition**: Collecting raw data (structured tables, text, images).
2. **Preprocessing**: Handling missing values, removing duplicates, and scaling numerical features.
3. **Feature Engineering**: Creating new variables or transforming existing ones (e.g., one-hot encoding categorical variables) to help the model learn.
4. **Model Training**: Feeding data to the algorithm to optimize its internal weights.
5. **Model Evaluation**: Testing the model on unseen data using metrics like Accuracy or Mean Squared Error.
6. **Deployment**: Integrating the trained model into a production environment.

---

## Module 2: The Four Learning Paradigms

Machine learning is broadly categorized into four paradigms based on the presence and nature of feedback during training.

| Paradigm | Data Input | Core Goal | Common Algorithms | Real-World Example |
| :--- | :--- | :--- | :--- | :--- |
| **Supervised** | Labeled Data $(X, y)$ | Predict target $y$ from features $X$ | Linear Regression, SVM, Random Forest | Credit scoring, Spam filtering |
| **Unsupervised** | Unlabeled Data $(X)$ | Find hidden patterns or groupings | K-Means, PCA, DBSCAN | Customer segmentation |
| **Semi-Supervised** | Tiny Labeled + Large Unlabeled | Leverage unlabeled data to boost learning | Label Propagation, Self-Training | Medical imaging annotation |
| **Reinforcement** | Environment & Rewards | Learn optimal policy via trial-and-error | Q-Learning, Deep Q-Networks (DQN) | Autonomous driving, AlphaGo |

```mermaid
flowchart TD
    subgraph Paradigms [Learning Paradigms]
        direction LR
        A["Supervised Learning<br/>(Labeled instances guide model)<br/>e.g., Spam Filtering"]
        B["Unsupervised Learning<br/>(Discovers structure automatically)<br/>e.g., Customer Clustering"]
        C["Semi-Supervised Learning<br/>(Blends labeled & unlabeled data)<br/>e.g., Medical Image Labeling"]
        D["Reinforcement Learning<br/>(Policy optimization via rewards)<br/>e.g., Autonomous Driving / Chess AI"]
    end
    style A fill:#4F46E5,stroke:#312E81,stroke-width:1px,color:#FFF
    style B fill:#0EA5E9,stroke:#0369A1,stroke-width:1px,color:#FFF
    style C fill:#10B981,stroke:#047857,stroke-width:1px,color:#FFF
    style D fill:#F59E0B,stroke:#B45309,stroke-width:1px,color:#FFF
```

### 2.2 Case Study: Where do Large Language Models (LLMs) fit?
Modern LLMs (like GPT, Claude, Llama, and Gemini) do not fit neatly into a single learning paradigm. Instead, they are trained using a **multi-paradigm pipeline** that combines self-supervised learning, supervised learning, and reinforcement learning:

```mermaid
flowchart TD
    A[(Raw Unlabeled Web Text)] --> B[Stage 1: Pre-training<br/>Self-Supervised Learning]
    B --> C[(Instruction-Response Pairs)]
    C --> D[Stage 2: Supervised Fine-Tuning SFT]
    D --> E[(Human Preference Comparisons)]
    E --> F[Stage 3: Alignment<br/>Reinforcement Learning RLHF]
    F --> G[Production LLM]
    
    style B fill:#4F46E5,stroke:#312E81,color:#FFF
    style D fill:#10B981,stroke:#047857,color:#FFF
    style F fill:#F59E0B,stroke:#B45309,color:#FFF
```

1. **Pre-Training (Self-Supervised / Unsupervised)**:
   * **Process**: The model reads billions of words of raw, unlabeled text and learns to predict the next word.
   * **Connection**: While the dataset is unlabeled (unsupervised), the training creates target labels ($y$) automatically from the next word in the text (self-supervised).
2. **Supervised Fine-Tuning (SFT)**:
   * **Process**: The model is trained on human-curated datasets of instruction-answer pairs.
   * **Connection**: This is a direct application of **supervised learning**, training the model to align its outputs with high-quality labeled examples.
3. **Reinforcement Learning from Human Feedback (RLHF)**:
   * **Process**: The model generates responses, and a reward model scores them based on human preferences. The LLM updates its weights to maximize this reward.
   * **Connection**: This is **reinforcement learning**, where the LLM acts as the agent, the reward model acts as the environment, and the human preference criteria act as the reward signal.

---

## Module 3: Supervised Learning - Regression

Regression models predict a continuous numeric output (e.g., salary, temperature, house prices).

### 3.1 Simple and Multiple Linear Regression
Linear Regression assumes a linear relationship between the input features $X$ and the target $y$.

#### Mathematical Formulation
The prediction equation is:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T X$$

Where:
* $\hat{y}$ is the predicted value.
* $\theta_0$ is the intercept (bias).
* $\theta_j$ are the feature weights (coefficients).
* $x_j$ are the feature values.

The model is optimized by minimizing the **Mean Squared Error (MSE)** cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

##### Parameter Breakdown:
* **$J(\theta)$**: The **Cost (or Loss) Function** value. It measures how much the model's predictions deviate from the actual targets. The goal of training is to find the parameters $\theta$ that minimize this value.
* **$\theta$ (Theta)**: The vector of model parameters (coefficients/weights and bias).
* **$m$**: The total number of **training examples** (data samples) in the dataset.
* **$\frac{1}{2m}$**: A scaling factor. The $m$ in the denominator averages the errors across all samples. The $2$ in the denominator is a mathematical convenience: during derivative calculation in gradient descent, it cancels out the exponent $2$ from the squared term, leaving a simpler derivative.
* **$\sum_{i=1}^{m}$**: The summation symbol, instructing to compute the squared error for each training sample from $1$ to $m$ and sum them all together.
* **$h_\theta(x^{(i)})$**: The **Hypothesis function** (predicted value $\hat{y}^{(i)}$) for the $i$-th training sample, computed using the current parameters $\theta$ and inputs $x^{(i)}$.
* **$x^{(i)}$**: The input feature vector for the **$i$-th training example** (the superscript $(i)$ is an index, not an exponent).
* **$y^{(i)}$**: The actual target value (ground truth label) for the **$i$-th training example**.
* **$(h_\theta(x^{(i)}) - y^{(i)})^2$**: The **squared prediction error** for the $i$-th sample. Squaring the error ensures that negative and positive errors don't cancel each other out, and heavily penalizes larger errors.

Using **Gradient Descent**, parameters are updated iteratively:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

### 3.2 Regularization: Ridge & Lasso
When a model has too many parameters, it might overfit. **Regularization** adds a penalty term to the cost function to constrain the weights.

```mermaid
graph LR
    A[Loss Function] --> B[L2 Penalty: Ridge Regression]
    A --> C[L1 Penalty: Lasso Regression]
    
    B --> B1["J(θ) + λ ∑ θ² (Shrinks weights close to 0)"]
    C --> C1["J(θ) + λ ∑ |θ| (Drives weights exactly to 0 - Feature Selection)"]
```

#### Ridge Regression ($L_2$ Regularization)
Adds the sum of squared weights as a penalty:

$$J_{\text{Ridge}}(\theta) = J(\theta) + \lambda \sum_{j=1}^{n} \theta_j^2$$

* **Effect**: Shrinks coefficients toward zero, reducing variance, but never makes them exactly zero.

#### Lasso Regression ($L_1$ Regularization)
Adds the sum of absolute weights as a penalty:

$$J_{\text{Lasso}}(\theta) = J(\theta) + \lambda \sum_{j=1}^{n} |\theta_j|$$

* **Effect**: Drives less important feature weights to exactly zero. Useful for **feature selection**.

#### Conceptual Example: The "Biased Word" in Spam Detection
Suppose we are training a model to predict the probability that an email is spam, using features $x_j$ representing the frequency of specific words (e.g., $x_1 = \text{"win"}$, $x_2 = \text{"free"}$, $x_3 = \text{"hello"}$).

* **The Problem (Overfitting & High Variance)**:
  In a small training dataset of 20 emails, the word **"win"** only appears in spam messages. An unregularized model (OLS) will assign an excessively high, biased positive weight to "win" (e.g., $\theta_{\text{win}} = +15.0$). Later, if a user receives a legitimate email like *"Did our team win the game?"*, the model will incorrectly flag it as spam because of this massive weight.
* **How Ridge ($L_2$) Solves It**:
  Ridge penalizes squared weights. A weight of $+15.0$ contributes a massive penalty ($15^2 = 225.0$). Ridge forces this weight to shrink drastically (e.g., to $\theta_{\text{win}} = +1.2$). "Win" remains a positive spam indicator but is no longer large enough to override standard vocabulary.
* **How Lasso ($L_1$) Solves It**:
  Lasso penalizes absolute weights. It identifies that common words like "hello" ($\theta_{\text{hello}} = +0.1$) add noise but have almost zero actual predictive capability. Lasso drives their weights to **exactly zero** ($\theta_{\text{hello}} = 0$), performing feature selection and removing noise features completely from the equation.

### 3.3 Python Implementation
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
X = np.random.rand(100, 5)
y = 3 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(100) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Ordinary Least Squares Linear Regression
ols = LinearRegression()
ols.fit(X_train, y_train)
print(f"OLS MSE: {mean_squared_error(y_test, ols.predict(X_test)):.4f}")

# Fit Ridge Regression (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge MSE: {mean_squared_error(y_test, ridge.predict(X_test)):.4f}")

# Fit Lasso Regression (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Lasso MSE: {mean_squared_error(y_test, lasso.predict(X_test)):.4f}")
```

---

## Module 4: Supervised Learning - Classification

Classification models predict discrete class labels (e.g., Spam vs. Not Spam, Cat vs. Dog vs. Bird).

### 4.1 Logistic Regression
Despite its name, Logistic Regression is used for binary classification. It maps real-valued outputs to probabilities between $0$ and $1$ using the **Sigmoid function**.

```mermaid
graph LR
    A["Linear Combination: z = θᵀX"] --> B["Sigmoid Function: σ(z)"]
    B --> C["Probability Output: P(y=1|X) ∈ [0,1]"]
```

#### Mathematical Formulation
The Sigmoid function $\sigma(z)$ is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The objective is to minimize the **Binary Cross-Entropy Loss (Log Loss)**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 4.2 K-Nearest Neighbors (KNN)
KNN is a **lazy learning** algorithm. It does not learn an explicit model; instead, it stores the dataset and classifies new samples by voting among the $K$ closest instances.

```mermaid
graph TD
    A[New Data Point] --> B[Calculate Distance to all training points]
    B --> C[Find the 'K' nearest points]
    C --> D{Classification or Regression?}
    D -- Classification --> E[Majority Vote of labels]
    D -- Regression --> F[Average value of labels]
```

#### Distance Metrics
* **Euclidean Distance**:

  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

* **Manhattan Distance**:

  $$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

### 4.3 Support Vector Machines (SVM)
SVM finds the optimal hyperplane that separates classes with the maximum **margin**.

```mermaid
graph TD
    A[Data Space] --> B[Find Margin boundaries]
    B --> C[Maximize margin between Support Vectors]
    C --> D[Kernel Trick: Project to higher dimensions if linearly inseparable]
```

* **Support Vectors**: The data points closest to the hyperplane.
* **Kernel Trick**: Projects non-linearly separable data into a higher-dimensional space where it becomes linearly separable (e.g., Radial Basis Function `RBF` kernel).

### 4.4 Naive Bayes
Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**, applying the "naive" assumption of conditional independence between features.

#### Mathematical Formulation

$$P(C_k \mid x) = \frac{P(x \mid C_k) P(C_k)}{P(x)}$$

With the naive conditional independence assumption:

$$P(C_k \mid x_1, \dots, x_n) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$

### 4.5 Decision Trees & Random Forests
* **Decision Trees**: Split data recursively based on feature thresholds that maximize homogeneity in resulting child nodes.
  * **Splitting Criteria**:
    * **Entropy (Information Gain)**:

      $$H(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

    * **Gini Impurity**:

      $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
* **Random Forests**: An ensemble of independent Decision Trees trained on random subsets of features and bootstrap data samples (a method called **Bagging**).

### 4.6 Python Implementation
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris Dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN (K=3)": KNeighborsClassifier(n_neighbors=3),
    "SVM (RBF Kernel)": SVC(kernel='rbf'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## Module 5: Ensemble Learning & XGBoost

Ensemble methods combine predictions from multiple base models to produce a more robust generalizer.

```mermaid
graph TD
    A[Ensemble Learning] --> B[Bagging: Bootstrap Aggregating]
    A --> C[Boosting: Sequential Learning]
    
    B --> B1[Models trained in parallel on random data subsets. e.g., Random Forest]
    C --> C1[Models trained sequentially; errors of prior model are weighted higher. e.g., AdaBoost, Gradient Boosting, XGBoost]
```

### 5.1 Bagging vs. Boosting
* **Bagging**: Trains independent base learners in parallel. Reduces **Variance** (reduces overfitting).
* **Boosting**: Trains base learners sequentially. Each new model corrects the residual errors of its predecessor. Reduces **Bias** (improves fit).

### 5.2 XGBoost (Extreme Gradient Boosting)
XGBoost is an optimized, highly efficient implementation of gradient boosted decision trees.

* **Key Strengths**:
  * **Regularization**: Includes L1 ($L_1$) and L2 ($L_2$) regularization inside the objective function to control overfitting.
  * **Parallel/Distributed Computing**: Highly optimized node splitting.
  * **Handling Sparsity**: Automatic direction routing for missing values.

#### Mathematical Formulation
The objective function at step $t$:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{m} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

Where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ is the tree complexity regularization.

### 5.3 Python Implementation
```python
import xgboost as xgb
from sklearn.datasets import make_classification

# Generate binary dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")
```

---

## Module 6: Unsupervised Learning - Clustering

Clustering groups unlabeled data points so that items in the same cluster are similar, and items in different clusters are distinct.

### 6.1 K-Means Clustering
K-Means partitions data into $K$ distinct clusters by minimizing the distance between points and their assigned cluster centroid.

```mermaid
graph TD
    A[Specify K clusters] --> B[Initialize centroids randomly]
    B --> C[Assign points to nearest centroid]
    C --> D[Recalculate centroids as cluster means]
    D --> E{Centroids shifted?}
    E -- Yes --> C
    E -- No --> F[Convergence reached]
```

#### Algorithm Steps
1. Randomly place $K$ centroids.
2. Assign each data point to its closest centroid.
3. Compute the mean of all points in each cluster and move the centroid there.
4. Repeat steps 2 and 3 until centroids stabilize (convergence).

#### Choosing K
* **Elbow Method**: Plot the Sum of Squared Errors (SSE/Inertia) vs. $K$ and find the "elbow" point.
* **Silhouette Analysis**: Measures how close each point in one cluster is to points in the neighboring clusters.

### 6.2 Hierarchical Clustering
Constructs a tree-like structure (dendrogram) of clusters.
* **Agglomerative**: Bottom-up approach. Start with individual points and merge the closest clusters.
* **Linkage Criteria**: Defines how distance between clusters is measured:
  * *Single*: Minimum distance between points.
  * *Complete*: Maximum distance between points.
  * *Average*: Mean distance between all pairs.
  * *Ward*: Minimizes variance increase when merging.

### 6.3 DBSCAN
**Density-Based Spatial Clustering of Applications with Noise** groups points based on density, identifying outliers as noise.

```mermaid
graph TD
    A[DBSCAN Points] --> B["Core Point (>= min_samples in Eps radius)"]
    A --> C["Border Point (< min_samples in Eps, but close to Core)"]
    A --> D["Noise (Neither Core nor Border)"]
```

* **Core Points**: Have at least `min_samples` within distance `eps`.
* **Border Points**: Have fewer than `min_samples` in `eps` distance, but reside close to a core point.
* **Noise**: Neither core nor border points.

### 6.4 Python Implementation
```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate 2D clusters
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
X = StandardScaler().fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Agglomerative
hierarchical = AgglomerativeClustering(n_clusters=4)
hc_labels = hierarchical.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

print("Clustering completed successfully.")
```

---

## Module 7: Evaluation & Validation

To ensure our models generalize to new data, we must evaluate them carefully.

### 7.1 Bias-Variance Tradeoff
* **Bias**: Error due to simplistic assumptions. High bias leads to **Underfitting** (model is too simple).
* **Variance**: Error due to capturing noise in the training data. High variance leads to **Overfitting** (model fits training data perfectly but fails on test data).

```
   Low Bias, High Variance          High Bias, Low Variance          Low Bias, Low Variance
        (Overfitting)                    (Underfitting)               (Optimal Generalization)
   +-----------------------+        +-----------------------+        +-----------------------+
   |   *   x     *         |        |         /             |        |   *   x *             |
   |     *   *   x    x    |        |        /              |        |    * * x  x           |
   |  x      *      x      |        |       /               |        |  x   *   x           |
   +-----------------------+        +-----------------------+        +-----------------------+
```

### 7.2 Cross-Validation
Instead of a single train/test split, **K-Fold Cross-Validation** splits the data into $K$ subsets. The model is trained on $K-1$ subsets and evaluated on the remaining subset. This is repeated $K$ times.

```
Iteration 1: [ Test ] [ Train ] [ Train ] [ Train ]
Iteration 2: [ Train ] [ Test ] [ Train ] [ Train ]
Iteration 3: [ Train ] [ Train ] [ Test ] [ Train ]
Iteration 4: [ Train ] [ Train ] [ Train ] [ Test ]
```

### 7.3 Performance Metrics

#### Regression Metrics
* **Mean Absolute Error (MAE)**:

  $$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|$$

* **Mean Squared Error (MSE)**:

  $$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

* **R-squared ($R^2$)**:

  $$R^2 = 1 - \frac{\sum (y^{(i)} - \hat{y}^{(i)})^2}{\sum (y^{(i)} - \bar{y})^2}$$

#### Classification Metrics (Confusion Matrix)
```
                  Predicted Positive    Predicted Negative
Actual Positive         TP                    FN (Type II Error)
Actual Negative   FP (Type I Error)           TN
```

* **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
* **Precision**: $\frac{TP}{TP + FP}$ (Quality of positive predictions)
* **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (Model's coverage of actual positives)
* **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean)
* **ROC-AUC**: Graphing True Positive Rate vs. False Positive Rate; Area Under the Curve (AUC) measures class separation capability.

---

## Module 8: Code Walkthroughs & Implementation Steps

This module breaks down the code structure, execution steps, and logic for both the **Framework (Scikit-Learn/XGBoost)** and **From-Scratch (Pure Python)** scripts.

---

### 8.1 Regression (Housing Prices Example)
* **Target Concept**: Predict continuous house prices based on square footage.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/1_regression_housing.py](file:///Users/donthireddy/code/ai-course/examples/1_regression_housing.py) &rarr; Output Plot: [examples_1_regression.png](file:///Users/donthireddy/code/ai-course/plots/examples_1_regression.png)
  * **From-Scratch**: [scratch/1_linear_regression.py](file:///Users/donthireddy/code/ai-course/scratch/1_linear_regression.py) &rarr; Output Plot: [scratch_1_regression.png](file:///Users/donthireddy/code/ai-course/plots/scratch_1_regression.png)

#### Step-by-Step Logic Breakdown:
1. **Data Prep**: Define size array ($X$) and price array ($y$).
   * *From-Scratch Difference*: To prevent gradient descent number overflow, size features are normalized by dividing by 1000 ($1.5$ instead of $1500$).
2. **Train/Test Splitting**: The framework uses `train_test_split(X, y, test_size=0.2)` to set aside 2 samples for validation.
3. **Model Initialization & Fitting**:
   * *Framework*: Calls `.fit(X_train, y_train)` on `LinearRegression()`, `Ridge()`, and `Lasso()`.
   * *From-Scratch*: Manually initializes `weight = 0.0` and `bias = 0.0`. Runs a loop for `1000` epochs. Inside the loop, it calculates the prediction error ($h_\theta(x^{(i)}) - y^{(i)}$) for each sample, aggregates the gradients (sum of error $\times$ input size, and sum of errors), updates parameters, and prints MSE loss.
4. **Predicting**:
   * *Framework*: Calls `.predict([[2200]])` on all models.
   * *From-Scratch*: Computes prediction directly: `price = weight * (2200/1000) + bias`.

---

### 8.2 Classification (Student Exam Pass/Fail)
* **Target Concept**: Predict binary pass (1) or fail (0) status based on study hours.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/2_classification_exam.py](file:///Users/donthireddy/code/ai-course/examples/2_classification_exam.py) &rarr; Output Plot: [examples_2_classification.png](file:///Users/donthireddy/code/ai-course/plots/examples_2_classification.png)
  * **From-Scratch**: [scratch/2_logistic_regression.py](file:///Users/donthireddy/code/ai-course/scratch/2_logistic_regression.py) &rarr; Output Plot: [scratch_2_logistic.png](file:///Users/donthireddy/code/ai-course/plots/scratch_2_logistic.png)

#### Step-by-Step Logic Breakdown:
1. **Data Prep**: Inputs hours studied $1$ to $10$; targets are $0$ for hours $\le 5$, and $1$ for hours $\ge 6$.
2. **Probability Mapping (Sigmoid)**:
   * *Framework*: Handled internally by `LogisticRegression()`.
   * *From-Scratch*: Implements `sigmoid(z) = 1 / (1 + exp(-z))` with bounds checks to prevent mathematical float overflow.
3. **Training Updates**:
   * *Framework*: Fits coefficients using coordinate descent solver.
   * *From-Scratch*: Runs a `2000` epoch loop. Computes $z = weight \times hours + bias$, passes it to `sigmoid()`, calculates prediction error, updates parameters via negative gradients, and computes Log Loss to monitor convergence.
4. **Decision Boundaries**:
   * *Both*: Classify as Pass (1) if probability $\ge 0.5$ ($z \ge 0$), otherwise Fail (0).

---

### 8.3 K-Nearest Neighbors (Offer Acceptance)
* **Target Concept**: Classify card offer acceptance based on [Age, Income].
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/3_knn_classification.py](file:///Users/donthireddy/code/ai-course/examples/3_knn_classification.py) &rarr; Output Plot: [examples_3_knn.png](file:///Users/donthireddy/code/ai-course/plots/examples_3_knn.png)
  * **From-Scratch**: [scratch/3_knn_classification.py](file:///Users/donthireddy/code/ai-course/scratch/3_knn_classification.py) &rarr; Output Plot: [scratch_3_knn.png](file:///Users/donthireddy/code/ai-course/plots/scratch_3_knn.png)

#### Step-by-Step Logic Breakdown:
1. **Feature Scaling (Standardization)**:
   * *Framework*: Uses `StandardScaler().fit_transform(X)`.
   * *From-Scratch*: Computes the exact mean and standard deviation of Age and Income across the dataset. Standardizes points using: `scaled = (val - mean) / std`.
2. **Distance Matrix Calculation**:
   * *Framework*: Handled internally by `KNeighborsClassifier()`.
   * *From-Scratch*: For the query point, calculates the Euclidean distance to every training point: $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$.
3. **Sorting & Voting**:
   * *Both*: Select the $K$ points with the smallest distances.
   * *From-Scratch*: Sorts a list of `(distance, index)` tuples using `.sort()`, retrieves the top $K$ items, tallies classification votes, and outputs the majority class.

---

### 8.4 Decision Trees & Random Forests (Churn)
* **Target Concept**: Predict customer churn based on Age, Support Calls, and Tenure.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/4_decision_tree_rf.py](file:///Users/donthireddy/code/ai-course/examples/4_decision_tree_rf.py) &rarr; Output Plot: [examples_4_decision_tree.png](file:///Users/donthireddy/code/ai-course/plots/examples_4_decision_tree.png)
  * **From-Scratch**: [scratch/4_decision_tree.py](file:///Users/donthireddy/code/ai-course/scratch/4_decision_tree.py) &rarr; Output Plot: [scratch_4_decision_tree.png](file:///Users/donthireddy/code/ai-course/plots/scratch_4_decision_tree.png)

#### Step-by-Step Logic Breakdown:
1. **Splitting Metric (Gini Impurity)**:
   * *Framework*: Evaluates splits using internal C libraries.
   * *From-Scratch*: Implements `calculate_gini(labels)`: counts zeros and ones, calculates probability proportions ($p_0, p_1$), and computes $1 - (p_0^2 + p_1^2)$.
2. **Exhaustive Threshold Search**:
   * *Framework*: Automatically tests splits to build a full multi-level tree.
   * *From-Scratch*: Loops through each feature column. For each feature, sorts the values, identifies unique midpoints, splits the dataset, calculates the weighted Gini impurity of the left and right child nodes, and selects the feature and threshold that minimize Gini impurity.
3. **Ensemble Aggregation**:
   * *Framework*: Fits `RandomForestClassifier` which trains multiple independent trees on bootstrapped samples and random features, returning averaged votes.

---

### 8.5 Ensemble Boosting (XGBoost Loan Defaults)
* **Target Concept**: Predict default probability using gradient boosted trees.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/5_xgboost_ensemble.py](file:///Users/donthireddy/code/ai-course/examples/5_xgboost_ensemble.py) &rarr; Output Plot: [examples_5_xgboost.png](file:///Users/donthireddy/code/ai-course/plots/examples_5_xgboost.png)

#### Step-by-Step Logic Breakdown:
1. **Dataset Scaling**: Replicates sample indices to produce a robust $100$-sample training set.
2. **Sequential Fitting**: `XGBClassifier` fits trees sequentially. The first tree makes baseline predictions, calculates residuals, and subsequent trees are trained to fit these residuals.
3. **Regularized Complexity**: XGBoost checks leaves count ($T$) and weights ($w$) to penalize model complexity directly during node splits.

---

### 8.6 K-Means Clustering (Customer Segments)
* **Target Concept**: Group customers based on Income and Spending Score.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/6_kmeans_clustering.py](file:///Users/donthireddy/code/ai-course/examples/6_kmeans_clustering.py) &rarr; Output Plot: [examples_6_kmeans.png](file:///Users/donthireddy/code/ai-course/plots/examples_6_kmeans.png)
  * **From-Scratch**: [scratch/5_kmeans_clustering.py](file:///Users/donthireddy/code/ai-course/scratch/5_kmeans_clustering.py) &rarr; Output Plot: [scratch_5_kmeans.png](file:///Users/donthireddy/code/ai-course/plots/scratch_5_kmeans.png)

#### Step-by-Step Logic Breakdown:
1. **Centroid Initialization**: Set initial coordinates for $K=3$ cluster centers.
2. **Iterative Updates**:
   * **Distance & Assignment**: Compute distance from each coordinate point to all 3 centroids. Assign the point to the cluster index of the closest centroid.
   * **Centroid Moving**: Group points by cluster index. Recalculate the centroid coordinate as the average (mean) of all points in that cluster.
   * **Convergence Check**:
     * *Framework*: Continues until coordinate shifts fall below tolerance.
     * *From-Scratch*: Compares new centroid coordinates with previous coordinates. If they are identical, it breaks the loop early.

---

### 8.7 DBSCAN Density Clustering (Outlier Detection)
* **Target Concept**: Density-based clustering that automatically flags outlier points as noise.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/7_dbscan_clustering.py](file:///Users/donthireddy/code/ai-course/examples/7_dbscan_clustering.py) &rarr; Output Plot: [examples_7_dbscan.png](file:///Users/donthireddy/code/ai-course/plots/examples_7_dbscan.png)
  * **From-Scratch**: [scratch/6_dbscan_clustering.py](file:///Users/donthireddy/code/ai-course/scratch/6_dbscan_clustering.py) &rarr; Output Plot: [scratch_6_dbscan.png](file:///Users/donthireddy/code/ai-course/plots/scratch_6_dbscan.png)

#### Step-by-Step Logic Breakdown:
1. **Neighborhood Discovery**:
   * *From-Scratch*: Calculates Euclidean distance between all coordinate pairs. Builds a list of indices representing neighbors within distance `eps = 0.5`.
2. **Core Point Identification**:
   * *Both*: If a point has at least `min_samples = 3` neighbors, it is classified as a **Core Point**.
3. **Queue Expansion (BFS)**:
   * *From-Scratch*: Loops through the points. When an unvisited core point is found, it initializes a new cluster and uses a list-based queue to traverse its neighbors. If any neighbor in the queue is also a core point, its own neighbors are appended to the search queue.
4. **Outlier Labeling**:
   * *Both*: Any point not reachable from a core point remains labeled as `-1` (Noise/Outlier).

---

### 8.8 Model Evaluation & Performance Metrics
* **Target Concept**: Compute metrics for classification and regression models.
* **Runnable Files & Output Plots**:
  * **Framework**: [examples/8_model_evaluation.py](file:///Users/donthireddy/code/ai-course/examples/8_model_evaluation.py) &rarr; Output Plot: [examples_8_evaluation.png](file:///Users/donthireddy/code/ai-course/plots/examples_8_evaluation.png)
  * **From-Scratch**: [scratch/7_model_evaluation.py](file:///Users/donthireddy/code/ai-course/scratch/7_model_evaluation.py) &rarr; Output Plot: [scratch_7_evaluation.png](file:///Users/donthireddy/code/ai-course/plots/scratch_7_evaluation.png)

#### Step-by-Step Logic Breakdown:
1. **Classification Metrics (Confusion Matrix)**:
   * *From-Scratch*: Sets counters `tp, fp, tn, fn = 0`. Loops through actual and predicted pairs, updating counters.
   * *Ratios*: Compute Accuracy: `(tp+tn)/total`, Precision: `tp/(tp+fp)`, Recall: `tp/(tp+fn)`, and F1-Score: `2 * p * r / (p + r)`.
2. **Regression Metrics**:
   * *From-Scratch*: Loops through actual and predicted prices. Aggregates absolute differences (`abs(act - pred)`) and squared differences (`(act - pred)**2`).
   * *Ratios*: Compute MAE: `sum_abs/n`, MSE: `sum_sq/n`, RMSE: `sqrt(MSE)`.
   * **R-squared ($R^2$)*: Computes the mean price, calculates total variance sum (`(act - mean)**2`), and returns `1.0 - (sum_squared_error / sum_total_variance)`.

---

## Module 9: Introduction to Large Language Models (LLMs) & Transformers

This module covers the core components of modern generative AI and Large Language Models (LLMs), moving from subword tokenization to continuous word semantics, and finally to the complete Sequence-to-Sequence (Seq2Seq) Transformer architecture.

### 9.1 Tokenization: Byte Pair Encoding (BPE)
Traditional tokenizers split text by whitespace or punctuation, leading to two major flaws: massive vocabulary sizes (which demand huge embedding memory) and an inability to handle unseen or misspelled words (out-of-vocabulary). 

**Byte Pair Encoding (BPE)** solves this by breaking words down into dynamic subwords. It is trained as follows:
1. Initialize the vocabulary ($V$) with all individual characters present in the training corpus, plus a special end-of-word marker `</w>`.
2. Segment the training corpus into space-separated characters.
3. Count frequencies of all adjacent token pairs (bigrams).
4. Identify the most frequent bigram $(s_1, s_2)$ and merge it into a single new subword token $s_{1\_2}$. Add $s_{1\_2}$ to $V$.
5. Repeat steps 3 and 4 for a pre-determined number of merge iterations.

When tokenizing new text, the learned merge rules are applied in the exact order they were trained, slicing words into the largest possible subwords found in the vocabulary.

---

### 9.2 Vector Semantics: Word Embeddings (Word2Vec)
Computers cannot process raw strings; words must be represented as continuous, dense numerical vectors where semantic similarity translates to geometric closeness (e.g. cosine similarity).

The **Word2Vec Skip-gram** model learns these embeddings by training a simple neural network to predict surrounding context words given a target input word:

$$\mathcal{L} = -\log \sigma({v'_{c_{pos}}}^T v_w) - \sum_{i=1}^k \log \sigma(-{v'_{c_{neg_i}}}^T v_w)$$

Where:
- $v_w \in \mathbb{R}^d$ is the input representation vector of the target word $w$.
- $v'_c \in \mathbb{R}^d$ is the output context representation vector of a candidate word $c$.
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function representing probability.
- $k$ is the number of randomly selected "negative samples" (words that are not in the target context) used to transform the expensive softmax calculation into efficient binary classifications.

During training, gradient updates adjust the word vectors directly, causing semantically related words to group together in the vector space.

---

### 9.3 The Encoder: Self-Attention & Positional Encoding
The Transformer Encoder processes all token embeddings in parallel. To preserve sequence order, we add a wave-like **Positional Encoding (PE)** vector to each token embedding before the self-attention layer:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where $pos$ is the token position in the sequence, and $i$ represents the dimension index.

#### Scaled Dot-Product Self-Attention
For a sequence of input vectors $X \in \mathbb{R}^{T \times d_{model}}$, we project it into Queries ($Q$), Keys ($K$), and Values ($V$) using weight matrices $W^Q, W^K, W^V$:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

We compute attention alignment weights by taking the dot product of queries and keys, scaling by the dimension size $\sqrt{d_k}$ to prevent gradient vanishing, and applying softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $d_k$ is the dimension of keys per attention head.

#### Multi-Head Attention (MHA)
Rather than computing attention once, we split $Q, K, V$ into $H$ heads, compute attention on each head in parallel, concatenate the outputs, and project back using an output matrix $W^O$. This allows the model to attend to information from different representation subspaces simultaneously.

The encoder layer output is finalized using residual connections and Layer Normalization:

$$X_{attn} = \text{LayerNorm}(X + \text{MultiHeadAttention}(X))$$

$$X_{output} = \text{LayerNorm}(X_{attn} + \text{FeedForward}(X_{attn}))$$

Where the Feed-Forward Network is a simple two-layer MLP with ReLU activation:

$$\text{FeedForward}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

---

### 9.4 The Decoder: Causal Masking & Cross-Attention
The Transformer Decoder generates target tokens autoregressively. It features two special attention layers:

1. **Causal Masked Self-Attention**: To prevent the model from looking at future target tokens during training, we add a causal mask matrix $M$ to the dot-product attention scores before softmax:

   $$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

   $$\text{Attention}_{masked}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

   When softmax is applied, the $-\infty$ values become exactly 0, preventing attention to future positions.
2. **Encoder-Decoder Cross-Attention**: The Queries ($Q$) come from the decoder's masked self-attention representation, whereas the Keys ($K$) and Values ($V$) come from the encoder's final output representations. This maps target words directly back to the source words.

---

### 9.5 End-to-End LLM Python Implementation
We have created five pure Python scripts inside the `llm/` directory showing how these pieces operate together step-by-step:

1. **[llm/1_bpe_tokenizer.py](file:///Users/donthireddy/code/ai-course/llm/1_bpe_tokenizer.py)**: Trains BPE merge rules on a corpus, segments out-of-vocabulary words, and saves [llm_1_bpe_vocab.png](file:///Users/donthireddy/code/ai-course/plots/llm_1_bpe_vocab.png).
2. **[llm/2_word2vec.py](file:///Users/donthireddy/code/ai-course/llm/2_word2vec.py)**: Performs gradient descent on context pairs, trains semantic 2D embeddings, and saves [llm_2_word2vec.png](file:///Users/donthireddy/code/ai-course/plots/llm_2_word2vec.png) (showing semantic word clusters).
3. **[llm/3_encoder.py](file:///Users/donthireddy/code/ai-course/llm/3_encoder.py)**: Implements positional encoding math, Q/K/V projections, Multi-Head self-attention, LayerNorm, and FFN, saving the wave positional encoding heatmap to [llm_3_positional_encoding.png](file:///Users/donthireddy/code/ai-course/plots/llm_3_positional_encoding.png).
4. **[llm/4_decoder.py](file:///Users/donthireddy/code/ai-course/llm/4_decoder.py)**: Implements causal masking, cross-attention alignment, and saves the causal mask visual boundary heatmap to [llm_4_causal_mask.png](file:///Users/donthireddy/code/ai-course/plots/llm_4_causal_mask.png).
5. **[llm/5_transformer_llm.py](file:///Users/donthireddy/code/ai-course/llm/5_transformer_llm.py)**: Assembles the tokenizer, embeddings, encoder, and decoder layers. Translates English inputs to Spanish step-by-step using greedy autoregressive decoding, saving the cross-attention alignment map to [llm_5_attention_alignment.png](file:///Users/donthireddy/code/ai-course/plots/llm_5_attention_alignment.png).

---

## Appendix: Glossary of Key Terminologies

To help students quickly grasp machine learning jargon, here is a consolidated list of key terms with examples and illustrative diagrams:

### 1. Centroid
* **Definition**: The geometric center of a cluster, calculated as the arithmetic mean (average) of all coordinate vectors of the data points in that cluster. Used heavily in K-Means.
* **Example**: For three 2D points in a cluster: $P_1(1, 5)$, $P_2(2, 4)$, and $P_3(3, 1)$.

  $$\text{Centroid } C = \left( \frac{1+2+3}{3}, \frac{5+4+1}{3} \right) = (2, 3.33)$$

* **Diagram**:
```
     y-axis
     ^   * P1(1, 5)
     |          * P2(2, 4)
     |       C (2, 3.33)  <-- Centroid (mean coordinate)
     |   * P3(3, 1)
     +---------------------------> x-axis
```

---

### 2. Hyperplane, Margin, & Support Vectors (SVM Concepts)
* **Definitions**:
  * **Hyperplane**: A decision boundary that splits space into regions representing different classes (e.g., a line in 2D space, a flat plane in 3D, and a high-dimensional surface in 4D or higher).
  * **Margin**: The distance between the decision boundary (hyperplane) and the closest training points from either class. Maximizing this margin is the core objective of SVMs.
  * **Support Vectors**: The critical data points that lie closest to the decision boundary. Changing or removing these points would alter the position of the boundary.
* **Diagram**:
```
      Class A (Stars)                   Class B (Crosses)
        *      *           |         |       x      x
            *   [SV *] --->|         |<--- [SV x]     x
                           |         |
      - - - - - - - - - - -+---------+ - - - - - - - - - - - <-- Margin Boundary
                           | Hyperplane (Optimal separating line)
               <---------->|         |<---------->
                  Margin                 Margin
```

---

### 3. Weak Learner
* **Definition**: A simple classifier or regressor that performs only slightly better than random guessing. Boosting algorithms sequentially combine weak learners to construct a strong model.
* **Example**: A **Decision Stump** is a decision tree with a depth of exactly 1 (it makes a decision based on only a single feature threshold split).
* **Diagram**:
```mermaid
flowchart TD
    Start[New Lead] --> Split{"Salary > $80k?"}
    Split -- Yes --> High[Predict: Buy Product]
    Split -- No --> Low[Predict: Do Not Buy]
```

---

### 4. Residual
* **Definition**: The difference between the actual observed value and the value predicted by the model ($y - \hat{y}$). Boosting models focus on training subsequent trees to predict these residuals.
* **Example**: If a house sells for $400k ($y = 400$) and a linear model predicts it will sell for $380k ($\hat{y} = 380$):

  $$\text{Residual} = y - \hat{y} = 400 - 380 = 20$$

* **Diagram**:
```
      y-axis (Price)
      ^            * Actual Value (y = 400)
      |           /|
      |          / |  <-- Residual (Error = +20)
      |         /  |
      |--------/---v--- <-- Predicted Line (y_hat = 380)
      |       /
      |      /
      +------------------------------> x-axis (Size)
```

---

### 5. Regularization ($L_1$ / $L_2$)
* **Definition**: A technique that prevents overfitting by adding a mathematical penalty to the cost function to constrain the size of model parameters ($\theta$).
* **Comparison**:
  * **$L_1$ (Lasso)**: Adds $\lambda \sum |\theta_j|$ penalty. Drives coefficients to exactly zero, eliminating features (performs feature selection).
  * **$L_2$ (Ridge)**: Adds $\lambda \sum \theta_j^2$ penalty. Shrinks coefficients close to zero but keeps them active.

---

### 6. Inductive Bias
* **Definition**: The underlying assumptions an algorithm makes about the target function to generalize from training data to unseen data.
* **Example**: Linear Regression's inductive bias is that the relationship between features and target is strictly linear ($y = \theta X$). KNN's inductive bias is that nearby points are highly likely to share the same label.

---

### 7. Lazy Learning
* **Definition**: Algorithms that do not construct a generalized model during the training phase (which is fast), but instead defer computations until a query is made (which is slow).
* **Example**: **K-Nearest Neighbors (KNN)** is a lazy learner. Training simply stores the raw points. Prediction computes distances to all stored points, making prediction computationally expensive for large datasets.

---

### 8. Bootstrapping & Feature Subspacing (Ensemble Concepts)
* **Definitions**:
  * **Bootstrapping**: A statistical sampling technique that generates new datasets by drawing samples from the original dataset with replacement. Used in Bagging (e.g., Random Forests).
  * **Feature Subspacing**: A technique where only a random subset of features is considered at each split in a decision tree. This helps decorrelate individual trees in a Random Forest.

---

### 9. Gradient
* **Definition**: The vector of partial derivatives representing the direction of steepest ascent of a function. In optimization, we move in the opposite direction (gradient descent) to minimize the loss.
* **Analogy**: Imagine being blindfolded on a foggy hill. To find the valley (minimum loss), you feel the slope of the ground with your foot and take a step in the direction that slopes downward (negative gradient).

---

### 10. Token
* **Definition**: The basic atomic unit of text that a language model processes. Depending on the tokenizer, a token can represent a single character, a subword (like BPE), or a full word.
* **Example**: Under a BPE tokenizer, the word "learning" might be split into two tokens: `["learn", "ing"]`.

---

### 11. Self-Attention
* **Definition**: A mechanism that allows an algorithm to weigh the importance of different tokens in the same sequence when constructing a representation for any given token.
* **Example**: In the sentence "The animal didn't cross the street because **it** was too tired", a self-attention layer learns to connect the query token "**it**" with high attention weights to "**animal**" rather than "street".

---

### 12. Causal Mask
* **Definition**: A lower-triangular binary matrix used in Decoder Self-Attention layers that forces the model to attend only to current and past tokens, zeroing out attention scores for future positions.
* **Example**: When predicting the 3rd word in a sentence, the causal mask ensures the attention mechanism has mathematically zero access to the 4th, 5th, or subsequent tokens.

---

### 13. Autoregressive Decoding
* **Definition**: A text generation strategy where the model generates output sequences word-by-word (or token-by-token). The token output from the previous step is appended to the input sequence and fed back into the model to predict the next token.
* **Example**: Generating "the", then feeding "the" to get "machine", then feeding "the machine" to get "learning".

---

### 14. Temperature
* **Definition**: A scaling hyperparameter applied to the model's output logit values before the softmax function to control the randomness of the predictions.
* **Effect**: A lower temperature (< 0.5) makes predictions more deterministic and repetitive (concentrates probability on the argmax token), whereas a higher temperature (> 1.0) makes predictions more creative, random, and diverse (spreads out probability).



