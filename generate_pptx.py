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
        p_title.font.size = Pt(32)
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

    # -------------------------------------------------------------
    # Slide 1: Title Slide (Sleek Landing)
    # -------------------------------------------------------------
    slide_layout = prs.slide_layouts[6] # Blank slide
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    
    # Large Decorative Left Highlight
    highlight = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(7.5))
    highlight.fill.solid()
    highlight.fill.fore_color.rgb = ACCENT_INDIGO
    highlight.line.fill.background()
    
    # Title & Subtitle in Single Textbox to prevent overlaps
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
    p3.text = "Instructor Slide Deck  •  Supervised, Unsupervised, & Ensembles"
    p3.font.name = "Arial"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_BODY

    # -------------------------------------------------------------
    # Slide 2: The Machine Learning Paradigm Shift
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "The Paradigm Shift: Rules vs. Data")
    
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

    # -------------------------------------------------------------
    # Slide 3: The 4 Learning Paradigms
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "The Four Machine Learning Paradigms")
    
    paradigms = [
        ("Supervised", "Labeled data drives training. Target labels (y) are paired with features (X).", "Credit Scoring, Pricing", ACCENT_INDIGO, Inches(0.8)),
        ("Unsupervised", "Unlabeled data. Discovers hidden structural groupings directly from features (X).", "Customer Clustering", ACCENT_TEAL, Inches(3.85)),
        ("Semi-Supervised", "Leverages small labeled dataset + huge unlabeled pool to reduce annotation costs.", "Image Annotation", ACCENT_ORANGE, Inches(6.9)),
        ("Reinforcement", "Interactive trial-and-error. Agent optimizes policy using environment rewards.", "Autonomous Driving", RGBColor(168, 85, 247), Inches(9.95))
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

    # -------------------------------------------------------------
    # Slide 4: Supervised Learning - Regression
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Regression: Predicting Continuous Targets")
    
    # Left Panel: OLS
    draw_panel(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    box_ols = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_ols = box_ols.text_frame
    tf_ols.word_wrap = True
    
    p = tf_ols.paragraphs[0]
    p.text = "Ordinary Least Squares (OLS)"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(12)
    
    ols_pts = [
        "Equation: y_hat = theta_0 + theta_1 * x_1 + ...",
        "Minimize Mean Squared Error (MSE) cost function.",
        "Uses Gradient Descent for iterative parameter optimization.",
        "Prone to overfitting if features contain significant multicollinearity or noise."
    ]
    for pt in ols_pts:
        p = tf_ols.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)
        
    # Right Panel: Regularization
    draw_panel(slide, Inches(7.0), Inches(1.8), Inches(5.5), Inches(4.8))
    box_reg = slide.shapes.add_textbox(Inches(7.3), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_reg = box_reg.text_frame
    tf_reg.word_wrap = True
    
    p = tf_reg.paragraphs[0]
    p.text = "Regularized Regression"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(12)
    
    reg_pts = [
        "Ridge (L2 Penalty): Adds lambda * sum(theta_j^2) to loss.",
        "  - Constrains coefficients, decreasing variance.",
        "Lasso (L1 Penalty): Adds lambda * sum(|theta_j|) to loss.",
        "  - Forces negligible features to zero, performing automatic feature selection.",
        "Crucial for high-dimensional data with redundant inputs."
    ]
    for pt in reg_pts:
        p = tf_reg.add_paragraph()
        p.text = "• " + pt if not pt.startswith("  -") else pt
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # Slide 5: Classification Essentials
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Classification: Binary & Multi-Class Models")
    
    class_models = [
        ("Logistic Regression", "Maps outputs into probabilities using the Sigmoid curve: 1 / (1 + e^-z). Optimizes log loss to establish boundaries.", ACCENT_TEAL, Inches(0.8)),
        ("K-Nearest Neighbors", "Non-parametric lazy learner. Classifies samples based on distance voting (e.g. Euclidean) of K nearest points.", ACCENT_INDIGO, Inches(4.85)),
        ("Support Vector Machines", "Identifies the separating hyperplane with the maximum margin. Employs the Kernel Trick for non-linear boundaries.", ACCENT_ORANGE, Inches(8.9))
    ]
    
    for name, desc, color, left in class_models:
        draw_panel(slide, left, Inches(1.8), Inches(3.65), Inches(4.8), border_color=color)
        
        box = slide.shapes.add_textbox(left + Inches(0.2), Inches(2.1), Inches(3.25), Inches(4.2))
        tf = box.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = name
        p.font.name = "Arial"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(14)
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # Slide 6: Trees, Forests & Ensembles
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Tree-Based Models & Bagging")
    
    # Left Panel: Decision Trees
    draw_panel(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    box_dt = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_dt = box_dt.text_frame
    tf_dt.word_wrap = True
    
    p = tf_dt.paragraphs[0]
    p.text = "Decision Trees"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(12)
    
    dt_pts = [
        "Split features recursively to maximize homogeneity.",
        "Splitting Metric: Entropy (Information Gain) or Gini Impurity.",
        "Highly intuitive and easy to interpret (rules can be visualized).",
        "Prone to overfitting (creates deep, highly specific trees)."
    ]
    for pt in dt_pts:
        p = tf_dt.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)
        
    # Right Panel: Random Forest
    draw_panel(slide, Inches(7.0), Inches(1.8), Inches(5.5), Inches(4.8))
    box_rf = slide.shapes.add_textbox(Inches(7.3), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_rf = box_rf.text_frame
    tf_rf.word_wrap = True
    
    p = tf_rf.paragraphs[0]
    p.text = "Random Forest (Bagging)"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(12)
    
    rf_pts = [
        "Ensemble of independent Decision Trees trained in parallel.",
        "Bootstrapping: Each tree trains on a random subset of rows.",
        "Feature Subspacing: Each split considers a random subset of columns.",
        "Averages individual trees (reduces variance/overfitting)."
    ]
    for pt in rf_pts:
        p = tf_rf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # Slide 7: Ensemble Boosting & XGBoost
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Boosting Paradigms & XGBoost")
    
    # Left Panel: Boosting Concepts
    draw_panel(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    box_boost = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_boost = box_boost.text_frame
    tf_boost.word_wrap = True
    
    p = tf_boost.paragraphs[0]
    p.text = "Boosting Concept"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(12)
    
    boost_pts = [
        "Sequential learning: Models are trained step-by-step.",
        "Each tree fits the residual errors of prior models.",
        "Reduces bias (improves fit on complex datasets).",
        "Requires careful learning rate (shrinkage) tuning."
    ]
    for pt in boost_pts:
        p = tf_boost.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)
        
    # Right Panel: XGBoost
    draw_panel(slide, Inches(7.0), Inches(1.8), Inches(5.5), Inches(4.8))
    box_xgb = slide.shapes.add_textbox(Inches(7.3), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_xgb = box_xgb.text_frame
    tf_xgb.word_wrap = True
    
    p = tf_xgb.paragraphs[0]
    p.text = "XGBoost (Extreme Gradient Boosting)"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(12)
    
    xgb_pts = [
        "Regularized boosting: Incorporates L1 & L2 penalties.",
        "Highly optimized for speed and parallel CPU usage.",
        "Automatic handling of missing values and sparse features.",
        "Typically dominates tabular data competitions."
    ]
    for pt in xgb_pts:
        p = tf_xgb.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # Slide 8: Unsupervised Learning - Clustering
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Clustering: Grouping Unlabeled Points")
    
    cluster_models = [
        ("K-Means", "Partitions data into K clusters. Centroids are updated iteratively to minimize within-cluster sum of squares (Inertia). Choose K via Elbow Method.", ACCENT_TEAL, Inches(0.8)),
        ("Hierarchical", "Constructs tree-like dendrograms. Agglomerative (bottom-up) merges closest pairs using linkage criteria (Ward, Single, Complete).", ACCENT_INDIGO, Inches(4.85)),
        ("DBSCAN", "Density-based grouping. Finds core, border, and noise points using eps (neighborhood radius) & min_samples. Captures complex shapes.", ACCENT_ORANGE, Inches(8.9))
    ]
    
    for name, desc, color, left in cluster_models:
        draw_panel(slide, left, Inches(1.8), Inches(3.65), Inches(4.8), border_color=color)
        
        box = slide.shapes.add_textbox(left + Inches(0.2), Inches(2.1), Inches(3.25), Inches(4.2))
        tf = box.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = name
        p.font.name = "Arial"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(14)
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # Slide 9: Evaluation and Validation
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Model Evaluation & Generalization")
    
    # Left Panel: Bias-Variance
    draw_panel(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    box_bv = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_bv = box_bv.text_frame
    tf_bv.word_wrap = True
    
    p = tf_bv.paragraphs[0]
    p.text = "Bias-Variance Tradeoff"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(12)
    
    bv_pts = [
        "Underfitting (High Bias): Model is too simple to capture patterns. Fails on train and test sets.",
        "Overfitting (High Variance): Model captures noise as rules. Scores high on train set, poorly on test set.",
        "Goal: Find the sweet spot minimizing total error.",
        "K-Fold Cross-Validation: Splits data K times to obtain robust performance estimations."
    ]
    for pt in bv_pts:
        p = tf_bv.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(10)
        
    # Right Panel: Metrics
    draw_panel(slide, Inches(7.0), Inches(1.8), Inches(5.5), Inches(4.8))
    box_met = slide.shapes.add_textbox(Inches(7.3), Inches(2.0), Inches(4.9), Inches(4.4))
    tf_met = box_met.text_frame
    tf_met.word_wrap = True
    
    p = tf_met.paragraphs[0]
    p.text = "Evaluation Metrics"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(12)
    
    met_pts = [
        "Regression Metrics:",
        "  - MAE (Absolute scale), MSE/RMSE (Penalizes outliers), R-squared (variance explained).",
        "Classification Metrics (Confusion Matrix):",
        "  - Accuracy: Overall correct predictions.",
        "  - Precision: TP / (TP + FP) (Quality of Positives).",
        "  - Recall: TP / (TP + FN) (Coverage of Positives).",
        "  - F1-Score: Harmonic mean of Precision & Recall.",
        "  - ROC-AUC: Ability to distinguish classes."
    ]
    for pt in met_pts:
        p = tf_met.add_paragraph()
        p.text = "• " + pt if not pt.startswith("  -") else pt
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_BODY
        p.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 10: Course Summary & Pedagogical Tips
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(slide_layout)
    apply_bg(slide)
    add_slide_header(slide, "Teaching Strategy & Next Steps")
    
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
        "5. Hands-on labs: Have students tune K-Means centroids or Random Forest depths on standard datasets (Iris/Titanic)."
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
