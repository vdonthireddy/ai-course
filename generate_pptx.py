#!/usr/bin/env python3
import sys
import os
import subprocess

# Ensure python-pptx is installed
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.enum.shapes import MSO_SHAPE
except ImportError:
    print("python-pptx is not installed. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    # Initialize Presentation
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    
    # -------------------------------------------------------------
    # Premium Palette Configuration (Modern Developer Dark Mode)
    # -------------------------------------------------------------
    BG_COLOR = RGBColor(18, 18, 20)          # #121214 Deep Charcoal
    PANEL_BG = RGBColor(30, 30, 36)          # #1E1E24 Slightly lighter Gray
    TEXT_TITLE = RGBColor(248, 250, 252)     # #F8FAFC Off-white
    TEXT_BODY = RGBColor(161, 161, 170)      # #A1A1AA Soft Gray
    ACCENT_INDIGO = RGBColor(99, 102, 241)   # #6366F1 Royal Indigo
    ACCENT_TEAL = RGBColor(20, 184, 166)     # #14B8A6 Teal
    ACCENT_ORANGE = RGBColor(245, 158, 11)   # #F59E0B Warm Amber
    ACCENT_PURPLE = RGBColor(168, 85, 247)   # #A855F7 Purple

    # Helper function to apply dark background to slide
    def apply_bg(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    # Helper function to write standardized titles
    def add_slide_header(slide, title_text, category="MACHINE LEARNING FUNDAMENTALS"):
        # Category breadcrumb
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.3))
        cat_tf = cat_box.text_frame
        cat_tf.word_wrap = True
        cat_tf.margin_top = cat_tf.margin_bottom = cat_tf.margin_left = cat_tf.margin_right = 0
        p_cat = cat_tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_TEAL
        
        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11), Inches(0.8))
        title_tf = title_box.text_frame
        title_tf.word_wrap = True
        title_tf.margin_top = title_tf.margin_bottom = title_tf.margin_left = title_tf.margin_right = 0
        p_title = title_tf.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(30)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_TITLE

    # Helper to add standard panels
    def draw_panel(slide, left, top, width, height, bg_color=PANEL_BG, border_color=ACCENT_INDIGO):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    # Helper function to scale and center an image on the right
    def add_right_image(slide, img_path, left=Inches(7.2), top=Inches(1.8), max_width=Inches(5.3), max_height=Inches(4.8)):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at '{img_path}'. Creating placeholder.")
            # Draw placeholder instead
            draw_panel(slide, left, top, max_width, max_height, bg_color=PANEL_BG, border_color=ACCENT_ORANGE)
            box = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(1.8), max_width - Inches(0.4), Inches(1.5))
            tf = box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = f"Image File Missing:\n{os.path.basename(img_path)}"
            p.font.name = "Arial"
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = ACCENT_ORANGE
            p.alignment = PP_ALIGN.CENTER
            return None

        # Add image temporarily to inspect size
        pic = slide.shapes.add_picture(img_path, left, top)
        aspect_ratio = pic.width / pic.height
        
        # Calculate optimal size preserving aspect ratio
        if max_width / max_height > aspect_ratio:
            # Height is the limiting factor
            pic.height = int(max_height)
            pic.width = int(max_height * aspect_ratio)
            pic.left = int(left + (max_width - pic.width) / 2)
            pic.top = int(top)
        else:
            # Width is the limiting factor
            pic.width = int(max_width)
            pic.height = int(max_width / aspect_ratio)
            pic.left = int(left)
            pic.top = int(top + (max_height - pic.height) / 2)
        return pic

    # Helper function to generate standard layout slides (Left text, Right image)
    def create_standard_slide(title, category, panel_title, bullets, img_path, border_color=ACCENT_INDIGO):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        apply_bg(slide)
        add_slide_header(slide, title, category)
        
        # Left Panel Background
        draw_panel(slide, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8), border_color=border_color)
        
        # Text Frame
        box = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.6), Inches(4.4))
        tf = box.text_frame
        tf.word_wrap = True
        
        # Panel Title
        p_title = tf.paragraphs[0]
        p_title.text = panel_title
        p_title.font.name = "Arial"
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = ACCENT_TEAL
        p_title.space_after = Pt(12)
        
        # Bullets formatting
        for b in bullets:
            p = tf.add_paragraph()
            p.font.name = "Arial"
            p.space_after = Pt(8)
            
            if b.startswith("  -") or b.startswith("    "):
                p.text = "    " + b.strip()
                p.font.size = Pt(12)
                p.font.color.rgb = TEXT_BODY
            elif b.startswith("-") or b.startswith("•"):
                p.text = "• " + b.lstrip("-• ").strip()
                p.font.size = Pt(13)
                p.font.color.rgb = TEXT_BODY
            else:
                p.text = b
                p.font.size = Pt(13)
                p.font.color.rgb = TEXT_BODY
                
        # Right Image Placement
        add_right_image(slide, img_path, left=Inches(7.2), top=Inches(1.8), max_width=Inches(5.3), max_height=Inches(4.8))
        return slide

    # =============================================================
    # Slide 1: Title Slide (Sleek Landing)
    # =============================================================
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    
    highlight = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(7.5))
    highlight.fill.solid()
    highlight.fill.fore_color.rgb = ACCENT_INDIGO
    highlight.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(10.5), Inches(3.5))
    tf = title_box.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "Fundamentals of\nMachine Learning"
    p1.font.name = "Arial"
    p1.font.size = Pt(54)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_TITLE
    p1.space_after = Pt(20)
    
    p2 = tf.add_paragraph()
    p2.text = "A Step-by-Step Teaching Curriculum & Core Models Guide"
    p2.font.name = "Arial"
    p2.font.size = Pt(20)
    p2.font.color.rgb = ACCENT_TEAL
    p2.space_after = Pt(10)

    p3 = tf.add_paragraph()
    p3.text = "Instructor Slide Deck  •  Supervised, Unsupervised, Ensembles & LLMs"
    p3.font.name = "Arial"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_BODY

    # =============================================================
    # Slide 2: The Machine Learning Paradigm Shift
    # =============================================================
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "The Paradigm Shift: Rules vs. Data", "MODULE 1: INTRODUCTION TO ML")
    
    # Traditional Panel
    draw_panel(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    box_trad = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_trad = box_trad.text_frame
    tf_trad.word_wrap = True
    
    p = tf_trad.paragraphs[0]
    p.text = "Traditional Software Engineering"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(14)
    
    bullets_trad = [
        "Human developer codes logical rules explicitly.",
        "System applies rules directly to inputs to yield answers.",
        "Highly deterministic but difficult to scale for complex patterns like vision or natural language.",
        "Example:\n  If 'free' and 'money' in text: mark as spam."
    ]
    for b in bullets_trad:
        p = tf_trad.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)
        
    # Machine Learning Panel
    draw_panel(slide, Inches(7.0), Inches(1.8), Inches(5.5), Inches(4.8))
    box_ml = slide.shapes.add_textbox(Inches(7.3), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_ml = box_ml.text_frame
    tf_ml.word_wrap = True
    
    p = tf_ml.paragraphs[0]
    p.text = "Machine Learning Paradigm"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(14)
    
    bullets_ml = [
        "Algorithm accepts raw data paired with historical outcomes.",
        "Model self-optimizes internal parameters to map X to y.",
        "Excels at multi-variable high-dimensional datasets that defy simple rule creation.",
        "Example:\n  Analyze millions of emails to extract probabilistic spam indicators automatically."
    ]
    for b in bullets_ml:
        p = tf_ml.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)

    # =============================================================
    # Slide 3: The 4 Learning Paradigms
    # =============================================================
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "The Four Machine Learning Paradigms", "MODULE 2: LEARNING PARADIGMS")
    
    paradigms = [
        ("Supervised", "Labeled data drives training. Target labels (y) are paired with features (X).", "Credit Scoring, Pricing", ACCENT_INDIGO, Inches(0.8)),
        ("Unsupervised", "Unlabeled data. Discovers hidden structural groupings directly from features (X).", "Customer Clustering", ACCENT_TEAL, Inches(3.85)),
        ("Semi-Supervised", "Leverages small labeled dataset + huge unlabeled pool to reduce annotation costs.", "Image Annotation", ACCENT_ORANGE, Inches(6.9)),
        ("Reinforcement", "Interactive trial-and-error. Agent optimizes policy using environment rewards.", "Autonomous Driving", ACCENT_PURPLE, Inches(9.95))
    ]
    
    for title, desc, eg, color, left in paradigms:
        draw_panel(slide, left, Inches(1.8), Inches(2.6), Inches(4.8), border_color=color)
        
        box = slide.shapes.add_textbox(left + Inches(0.15), Inches(2.0), Inches(2.3), Inches(4.4))
        tf = box.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(12)
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(24)
        
        p = tf.add_paragraph()
        p.text = "Example Use Case:"
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_TITLE
        
        p = tf.add_paragraph()
        p.text = eg
        p.font.size = Pt(13)
        p.font.italic = True
        p.font.color.rgb = color

    # =============================================================
    # Slide 4: Supervised Regression
    # =============================================================
    create_standard_slide(
        title="Regression: Predicting Continuous Targets",
        category="MODULE 3: SUPERVISED REGRESSION",
        panel_title="Continuous Model Formulation",
        bullets=[
            "Ordinary Least Squares (OLS): Fits a linear equation: y_hat = theta_0 + theta_1 * x_1 + ...",
            "Cost Function: Minimizes Mean Squared Error (MSE) to align the regression line.",
            "L2 Regularization (Ridge): Adds penalty lambda * sum(theta^2). Shrinks weights close to zero, decreasing variance.",
            "L1 Regularization (Lasso): Adds penalty lambda * sum(|theta|). Drives coefficients to exactly zero (Soft-Thresholding), doing automatic feature selection.",
            "Visual: Framework OLS vs. Ridge vs. Lasso on housing per-sq-ft."
        ],
        img_path="plots/examples_1_regression.png",
        border_color=ACCENT_TEAL
    )

    # =============================================================
    # Slide 5: Supervised Classification
    # =============================================================
    create_standard_slide(
        title="Classification: Categorizing Observations",
        category="MODULE 4: SUPERVISED CLASSIFICATION",
        panel_title="Decision Boundary Approaches",
        bullets=[
            "Logistic Regression: Maps inputs to probabilities using the Sigmoid function: 1 / (1 + e^-z).",
            "K-Nearest Neighbors (KNN): Non-parametric model. Classifies a query point based on distance-weighted votes of its K closest neighbors.",
            "Support Vector Machines (SVM): Finds a hyperplane that maximizes the margin between classes. Employs the Kernel Trick for non-linear boundaries.",
            "Naive Bayes: Probabilistic classifier based on Bayes Theorem. Assumes strong conditional independence among features.",
            "Visual: Multi-class decision boundaries of Scikit-Learn models on the Iris dataset."
        ],
        img_path="plots/classification_decision_boundaries.png",
        border_color=ACCENT_INDIGO
    )

    # =============================================================
    # Slide 6: Distance Metrics in Machine Learning
    # =============================================================
    create_standard_slide(
        title="Distance Metrics & Vector Spatial Relations",
        category="MODULE 4: SUPERVISED CLASSIFICATION",
        panel_title="Quantifying Geometric Closeness",
        bullets=[
            "Euclidean Distance: Straight-line (L2 norm) distance. Sensitive to large coordinate deviations.",
            "Manhattan Distance: Grid-based (L1 norm) distance. Sum of absolute differences along coordinate axes.",
            "Cosine Similarity: Measures the angular alignment between two vectors. Independent of vector magnitude (widely used in text/embeddings).",
            "Chebyshev Distance: Maximum absolute coordinate difference (L-infinity norm).",
            "Visual: Distance metric bounds (Circle, Diamond, Square) in a 2D coordinate space."
        ],
        img_path="plots/distance_metrics.png",
        border_color=ACCENT_ORANGE
    )

    # =============================================================
    # Slide 7: Decision Trees & Random Forests
    # =============================================================
    create_standard_slide(
        title="Tree-Based Models & Bagging Ensembles",
        category="MODULE 4: SUPERVISED CLASSIFICATION",
        panel_title="Recursive Splits & Forest Aggregation",
        bullets=[
            "Decision Trees: Splits features recursively to maximize sample purity.",
            "Purity Metrics: Evaluated using Gini Impurity or Entropy (Information Gain). Highly prone to overfitting.",
            "Random Forest Ensemble: Trains multiple independent trees in parallel.",
            "Bootstrapping: Each tree is trained on a random subset of rows (drawn with replacement).",
            "Feature Subspacing: Selects a random subset of features (e.g. sqrt(D)) at each split to decorrelate tree errors.",
            "Visual: Decision Tree splits vs. Random Forest averaged boundary."
        ],
        img_path="plots/examples_4_decision_tree.png",
        border_color=ACCENT_TEAL
    )

    # =============================================================
    # Slide 8: Ensemble Boosting & XGBoost
    # =============================================================
    create_standard_slide(
        title="Boosting Paradigms & XGBoost",
        category="MODULE 5: ENSEMBLE LEARNING & XGBOOST",
        panel_title="Sequential Optimization",
        bullets=[
            "Sequential Learning: Weak learners (e.g. shallow decision trees) are trained sequentially rather than in parallel.",
            "Residual Fitting: Each successive tree is trained to predict the residual errors (gradients) of the cumulative ensemble.",
            "XGBoost (Extreme Gradient Boosting): Highly optimized library.",
            "Regularized Objective: Incorporates L1 & L2 parameter constraints directly into the split objective.",
            "Implementation: Employs parallel split searches, block structures, and handles missing/sparse data automatically."
        ],
        img_path="plots/bagging_vs_boosting.png",
        border_color=ACCENT_PURPLE
    )

    # =============================================================
    # Slide 9: Unsupervised Learning - Clustering
    # =============================================================
    create_standard_slide(
        title="Clustering: Grouping Unlabeled Points",
        category="MODULE 6: UNSUPERVISED LEARNING",
        panel_title="Unsupervised Spatial Grouping",
        bullets=[
            "K-Means: Partitions data into K clusters. Iteratively updates cluster centroids to minimize within-cluster sum of squares (Inertia).",
            "Hierarchical Clustering: Builds nested tree structures (dendrograms) via agglomerative (bottom-up) merges.",
            "DBSCAN: Density-based algorithm. Groups core points within radius Eps having min_samples. Isolates outliers as noise.",
            "Visual: Scikit-Learn K-Means customer segmentation showing centroids and color-mapped clusters."
        ],
        img_path="plots/examples_6_kmeans.png",
        border_color=ACCENT_TEAL
    )

    # =============================================================
    # Slide 10: Model Evaluation & Validation
    # =============================================================
    create_standard_slide(
        title="Model Evaluation & Generalization",
        category="MODULE 7: EVALUATION & VALIDATION",
        panel_title="Validation & Performance Metrics",
        bullets=[
            "Bias-Variance Tradeoff: Underfitting (High Bias) vs. Overfitting (High Variance). Goal is minimizing total test error.",
            "K-Fold Cross-Validation: Splits data into K folds, cycling train/test roles to obtain robust performance bounds.",
            "Classification Metrics: Derived from the Confusion Matrix.",
            "  - Precision: TP / (TP + FP) (Quality of positive predictions).",
            "  - Recall: TP / (TP + FN) (Coverage of actual positive samples).",
            "  - F1-Score: Harmonic mean of Precision and Recall.",
            "  - ROC-AUC: True Positive vs. False Positive rate probability area."
        ],
        img_path="plots/examples_8_evaluation.png",
        border_color=ACCENT_INDIGO
    )

    # =============================================================
    # Slide 11: Module 8: Code Walkthroughs
    # =============================================================
    create_standard_slide(
        title="Framework vs. From-Scratch Algorithms",
        category="MODULE 8: CODE WALKTHROUGHS",
        panel_title="Pedagogical Implementation Strategy",
        bullets=[
            "Dual-Path Curriculum: Every algorithm is explored via two implementations: Scikit-Learn and Pure Python.",
            "Framework Path: Teaches APIs, hyperparameter tuning, pipelines, and industry best practices.",
            "From-Scratch Path: Avoids libraries. Uses standard lists and math functions to write loops, gradients, and Gini calculations.",
            "De-mystifying the 'Black Box': Solidifies mathematical equations directly into concrete, readable code.",
            "Visual: Pure Python Gradient Descent line-fitting convergence over epochs."
        ],
        img_path="plots/scratch_1_regression.png",
        border_color=ACCENT_ORANGE
    )

    # =============================================================
    # Slide 12: Module 9.1: Tokenization (BPE)
    # =============================================================
    create_standard_slide(
        title="Tokenization: Byte Pair Encoding (BPE)",
        category="MODULE 9: LLMS & TRANSFORMERS",
        panel_title="Subword Text Segmentation",
        bullets=[
            "Flaws of Word/Char Tokenizers: Massive vocabulary sizes (memory heavy) or inability to handle unseen out-of-vocabulary (OOV) words.",
            "BPE Concept: Dynamically merges the most frequent adjacent byte/character pairs into new subword units.",
            "Algorithm Steps:",
            "  1. Initialize vocabulary with individual characters + '</w>' marker.",
            "  2. Count frequencies of adjacent token pairs (bigrams) in the corpus.",
            "  3. Merge the most frequent bigram and add it to the vocabulary.",
            "  4. Repeat for N merge iterations.",
            "Visual: Vocabulary size growth vs. BPE merge rules application."
        ],
        img_path="plots/llm_1_bpe_vocab.png",
        border_color=ACCENT_TEAL
    )

    # =============================================================
    # Slide 13: Module 9.2: Vector Semantics (Word2Vec)
    # =============================================================
    create_standard_slide(
        title="Vector Semantics: Word Embeddings",
        category="MODULE 9: LLMS & TRANSFORMERS",
        panel_title="Continuous Word Semantics",
        bullets=[
            "Embedding Concept: Maps raw strings to dense, low-dimensional continuous vectors where geometric closeness represents semantic meaning.",
            "Word2Vec Skip-gram: Neural network trained to predict context words given a target input word.",
            "Negative Sampling Loss: Avoids expensive full softmax by training a binary classifier on positive context pairs vs. random negative pairs.",
            "Optimization: Trained using the Adam optimizer implemented from scratch. Uses adaptive learning rates for faster, more stable coordinate updates.",
            "Visual: Trained 2D Word2Vec scatter plot showing semantic clusters."
        ],
        img_path="plots/llm_2_word2vec.png",
        border_color=ACCENT_INDIGO
    )

    # =============================================================
    # Slide 14: Module 9.3: Transformer Encoder
    # =============================================================
    create_standard_slide(
        title="The Encoder: Self-Attention & Positional Encoding",
        category="MODULE 9: LLMS & TRANSFORMERS",
        panel_title="Parallel Contextual Encoding",
        bullets=[
            "Positional Encoding: Adds sine/cosine wave patterns to input embeddings to preserve token sequence order.",
            "Scaled Dot-Product Self-Attention: Projects input matrices into Query (Q), Key (K), and Value (V) representations.",
            "Alignment Formula: Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V. Scales by sqrt(d_k) to prevent gradient vanishing.",
            "Context Resolution: Dynamically shifts a word's vector representation based on surrounding tokens (e.g. 'bank' of river vs. money).",
            "Visual: Sinusoidal Positional Encoding matrix heatmap."
        ],
        img_path="plots/llm_3_positional_encoding.png",
        border_color=ACCENT_PURPLE
    )

    # =============================================================
    # Slide 15: Module 9.4: Transformer Decoder
    # =============================================================
    create_standard_slide(
        title="The Decoder: Causal Masking & Cross-Attention",
        category="MODULE 9: LLMS & TRANSFORMERS",
        panel_title="Autoregressive Target Generation",
        bullets=[
            "Autoregressive Generation: Predicts tokens sequentially, appending prior outputs as inputs for successive steps.",
            "Causal Masking: Adds a lower-triangular mask (-infinity for future positions) to attention scores prior to softmax.",
            "Look-Ahead Prevention: Mathematically sets attention weights to future tokens to exactly 0, preventing future-leakage.",
            "Encoder-Decoder Cross-Attention: Decoder queries (Q) attend to Encoder keys (K) and values (V), linking target to source.",
            "Visual: Decoder Causal Mask lower-triangular attention boundaries."
        ],
        img_path="plots/llm_4_causal_mask.png",
        border_color=ACCENT_ORANGE
    )

    # =============================================================
    # Slide 16: Module 9.5: End-to-End LLMSeq2Seq
    # =============================================================
    create_standard_slide(
        title="End-to-End Seq2Seq Transformer Model",
        category="MODULE 9: LLMS & TRANSFORMERS",
        panel_title="Bilingual Machine Translation Model",
        bullets=[
            "Seq2Seq Architecture: Couples an Encoder (source comprehension) with a Decoder (target autoregressive generation).",
            "Pure Python Implementation: Standard tokenizer, embeddings, encoder self-attention, and decoder cross-attention blocks.",
            "Greedy Decoding Loop: Iterates forward passes, selecting the token with the maximum probability until a stop token is reached.",
            "Teacher Forcing: Feed actual ground truth targets into the decoder during training to stabilize parameters.",
            "Visual: Attention alignment weights mapping English to Spanish."
        ],
        img_path="plots/llm_5_attention_alignment.png",
        border_color=ACCENT_TEAL
    )

    # =============================================================
    # Slide 17: Appendix: Math Glossary
    # =============================================================
    create_standard_slide(
        title="Appendix: Centroid, Hyperplane & Residuals",
        category="APPENDIX: GLOSSARY",
        panel_title="Mathematical Fundamentals",
        bullets=[
            "Centroid: The geometric center of a cluster of coordinates. Computed as the mean vector: mu = 1/N * sum(x_i).",
            "Hyperplane: A linear boundary of dimension D-1 that separates a D-dimensional space: w^T * x + b = 0.",
            "Margin: The distance between separating hyperplane and the closest support vectors: 2 / ||w||.",
            "Residual: The difference between actual and predicted target values: e_i = y_i - y_hat_i.",
            "Visual: Side-by-side plots of Centroid coordinate mean, SVM margins/support-vectors, and OLS regression residuals."
        ],
        img_path="plots/glossary_math_concepts.png",
        border_color=ACCENT_INDIGO
    )

    # =============================================================
    # Slide 18: Appendix: Normalization & Regularization
    # =============================================================
    create_standard_slide(
        title="Appendix: Normalization & Regularization",
        category="APPENDIX: GLOSSARY",
        panel_title="Scaling and Penalty Geometry",
        bullets=[
            "Normalization: Rescaling features to a shared coordinate bounds.",
            "  - Min-Max Scaling: Maps features to a fixed [0, 1] range.",
            "  - Standardization: Transforms features to have mean=0 and std=1.",
            "Regularization: Constrains model parameter magnitudes to prevent overfitting.",
            "  - Lasso (L1): Adds absolute weight penalty. Creates diamond-shaped boundaries that drive weights to exactly 0 (sparsity).",
            "  - Ridge (L2): Adds squared weight penalty. Circular boundaries that shrink weights close to 0 but keep all features active."
        ],
        img_path="plots/glossary_normalization_regularization.png",
        border_color=ACCENT_TEAL
    )

    # =============================================================
    # Slide 19: Appendix: Optimization & Bias Glossary
    # =============================================================
    create_standard_slide(
        title="Appendix: Optimization & Bias",
        category="APPENDIX: GLOSSARY",
        panel_title="Gradients and Algorithmic Assumptions",
        bullets=[
            "Gradient: Vector of partial derivatives pointing in the direction of steepest rate of function increase: [df/dw_1, ..., df/dw_D].",
            "Gradient Descent: Updates weights along negative gradient: w = w - eta * grad.",
            "Adam Optimizer: Computes adaptive learning rates using first moment (momentum) and second moment (gradient variance). Scales updates per coordinate.",
            "Inductive Bias: Mathematical assumptions a model makes to generalize to unseen inputs (e.g., linear vs. locality bias).",
            "Lazy Learning: Defers generalization until query time (e.g. KNN). Features zero training time but slow prediction.",
            "Visual: Gradient 3D valley, linear/locality boundaries, KNN distance query."
        ],
        img_path="plots/glossary_gradient_bias_lazy.png",
        border_color=ACCENT_ORANGE
    )

    # =============================================================
    # Slide 20: Appendix: LLM Concepts Glossary
    # =============================================================
    create_standard_slide(
        title="Appendix: LLM Core Terminology",
        category="APPENDIX: GLOSSARY",
        panel_title="Transformer Building Blocks",
        bullets=[
            "Token: The basic subword unit mapping to vocabulary IDs.",
            "Self-Attention: Maps query (Q) alignments to keys (K) to weight values (V).",
            "Causal Mask: Restricts attention to past positions.",
            "Autoregressive Decoding: Appends output tokens back to input sequence.",
            "Temperature: Scaling factor applied to logits before softmax.",
            "  - Low Temp (T -> 0): Peakier probabilities, deterministic generation.",
            "  - High Temp (T -> infinity): Uniform probabilities, highly random/creative.",
            "Visual: Token IDs, Self-Attention nodes, Mask heatmap, and Temperature curves."
        ],
        img_path="plots/glossary_llm_concepts.png",
        border_color=ACCENT_PURPLE
    )

    # =============================================================
    # Slide 21: Course Summary & Teaching Strategy
    # =============================================================
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Teaching Strategy & Lecture Structure", "COURSE CONCLUSION")
    
    # Large center box
    draw_panel(slide, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8), border_color=ACCENT_TEAL)
    box_sum = slide.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(10.9), Inches(4.4))
    tf_sum = box_sum.text_frame
    tf_sum.word_wrap = True
    
    p = tf_sum.paragraphs[0]
    p.text = "Recommended Lecture Structure"
    p.font.name = "Arial"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(14)
    
    sum_pts = [
        "1. Start with Paradigms: Make students classify everyday examples manually.",
        "2. Regression vs. Classification: Differentiate numeric forecasting vs. discrete categorization.",
        "3. Mathematics to Code: Show the formulas (Lasso, Logistic) side-by-side with scikit-learn models.",
        "4. Emphasize Evaluation: A model scoring 99% accuracy on training data is almost always overfit.",
        "5. Hands-on labs: Have students tune K-Means centroids or Random Forest depths on standard datasets (Iris/Titanic).",
        "6. Introduce Generative AI: Transition from embeddings (Word2Vec) to sequential comprehension (Transformers)."
    ]
    for pt in sum_pts:
        p = tf_sum.add_paragraph()
        p.text = pt
        p.font.size = Pt(16)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(12)
        
    # Save Presentation
    output_filename = "machine_learning_fundamentals.pptx"
    prs.save(output_filename)
    print(f"Presentation successfully created and saved as: '{output_filename}'")

if __name__ == "__main__":
    create_deck()
