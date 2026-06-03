# Machine Learning Basics: An Illustrated Mathematical Developer Guide

This document provides a comprehensive, step-by-step mathematical and visual explanation of the foundational concepts of Machine Learning (ML). It is based on `/Users/donthireddy/code/ai-course/002-ML Basics.pdf` and integrates the extracted high-resolution diagrams to explain algorithms, mathematical structures, and optimization techniques used in typical AI/ML models.

All illustration assets are referenced from the local directory: [plots/ml_basics/](../plots/ml_basics/).

---

## 1. Classical Machine Learning Paradigms

Machine Learning is a branch of Artificial Intelligence (AI) that focuses on developing models and algorithms that allow computers to learn from data and improve their performance without being explicitly programmed for every single task.

![Classical Machine Learning Paradigms](../plots/ml_basics/image_89_Im90.png)

Classical machine learning is primarily divided into two main paradigms:

### 1.1 Supervised Learning
In **Supervised Learning**, a model is trained on **labeled data**. The dataset contains input features paired with the correct output target. The model learns a mapping function from inputs to outputs to predict or classify new, unseen data.

$$\text{Dataset: } \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$$

*   **Regression:** Predicting a continuous numeric value (e.g., predicting house prices based on features like area and bedrooms).
*   **Classification:** Predicting a discrete class label or category (e.g., classifying whether an email is spam or not spam, or identifying animals).

![Supervised Learning Workflow](../plots/ml_basics/image_85_Im86.png)

### 1.2 Unsupervised Learning
In **Unsupervised Learning**, the model is trained on **unlabeled data**. The dataset only contains input features without corresponding target labels. The model learns to find hidden structures, groups, or patterns directly from the input distribution.

$$\text{Dataset: } \mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$$

*   **Clustering:** Grouping similar data points together based on distance or density (e.g., customer segmentation).
*   **Dimensionality Reduction:** Compressing high-dimensional feature spaces into lower-dimensional representations while preserving key features or variance (e.g., Principal Component Analysis).

![Unsupervised Learning Workflow](../plots/ml_basics/image_87_Im88.png)

### 1.3 Other Learning Paradigms
*   **Semi-Supervised Learning:** Combines a small amount of labeled data with a large amount of unlabeled data. This is highly useful when labeling data is expensive or time-consuming.
*   **Self-Supervised Learning:** The model generates its own labels directly from the structure of the input data (e.g., masking words in a sentence and training a model to predict the missing words). It has grown into its own field and forms the basis for training large-scale foundation models (such as LLMs).
*   **Reinforcement Learning:** An agent learns through trial and error by interacting with an environment to maximize cumulative rewards. It is ideal for decision-making and sequence control tasks (e.g., playing chess, robotics control, RLHF).

---

## 2. Mathematical Structures: Tensors

In machine learning, data and parameters are represented using multi-dimensional numerical arrays called **Tensors**. Tensors are classified by their **rank** (the number of dimensions/axes).

![Scalars, Vectors, Matrices, and Tensors Rank](../plots/ml_basics/image_91_Im92.jpg)

### 2.1 Tensor Classifications
1.  **Scalar (Rank-0 Tensor):** A single number representing magnitude.
    
    $$s \in \mathbb{R}$$
    
2.  **Vector (Rank-1 Tensor):** A 1D array of numbers representing points or directions in a multi-dimensional space.
    
    $$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \dots \\ v_d \end{bmatrix} \in \mathbb{R}^d$$
    
3.  **Matrix (Rank-2 Tensor):** A 2D grid of numbers with rows and columns, representing datasets, linear transformations, or image channels.
    
    $$\mathbf{M} = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix} \in \mathbb{R}^{h \times w}$$
    
4.  **Tensor (Rank-3 Tensor & Higher):** Array structures with 3 or more dimensions (e.g., RGB images with shape `[Height, Width, Channels]`, or mini-batches of video sequences with shape `[Batch, Frames, Height, Width, Channels]`).

![Vector, Matrix, Tensor Visual Representation](../plots/ml_basics/image_93_Im94.png)

### 2.2 Application Example: Image Data as Matrices
Digital images are stored as matrices (or 3D tensors for color images). For a grayscale image, each pixel intensity is stored as a numerical value inside a matrix grid.

![Image Grid to Binary Matrix Representation](../plots/ml_basics/image_95_Im96.png)

---

## 3. Linear Regression & Sum of Squared Errors (SSE)

**Linear Regression** models the relationship between a continuous dependent target variable $y$ and one or more independent predictor features $x$ by fitting a linear equation to the observed data.

### 3.1 Simple Linear Regression
For a single input feature $x$, the regression line is defined as:

$$f(x) = mx + b$$

where:
*   $m$ is the **slope** (weight parameter), determining the direction and steepness of the line.
*   $b$ is the **y-intercept** (bias parameter), determining the point where the line crosses the vertical axis.

![Raw Scatter Plot](../plots/ml_basics/image_3_Im4.png)

![Regression Line of Best Fit](../plots/ml_basics/image_5_Im6.png)

![Linear Regression Equation Plot](../plots/ml_basics/image_7_Im8.png)

### 3.2 Multiple Linear Regression
When there are multiple predictor features ($x_1, x_2, \dots, x_p$), the model generalizes to a hyper-plane equation:

$$y = \beta_p x_p + \dots + \beta_1 x_1 + \beta_0$$

or in vector form:

$$y = \mathbf{w}^T \mathbf{x} + b$$

For a model with two independent variables ($x_1$ and $x_2$), the regression boundary is a 3D plane:

$$y = \beta_2 x_2 + \beta_1 x_1 + \beta_0$$

![Multiple Regression Plane Equation](../plots/ml_basics/image_9_Im10.png)

![3D Regression Plane Visualization](../plots/ml_basics/image_11_Im12.png)

### 3.3 Sum of Squared Errors (SSE) Loss
To find the optimal weight and bias parameters, we must define a loss function that measures how far the model's predictions are from the actual values.

The vertical distance from an observed data point $y_i$ to the predicted regression value $\hat{y}_i$ is called the **residual** or **error**:

$$e_i = y_i - \hat{y}_i$$

![Regression Residual Error Lines](../plots/ml_basics/image_13_Im14.png)

The **Sum of Squared Errors (SSE)** squares each residual and sums them up. Squaring ensures that positive and negative errors do not cancel each other out, and heavily penalizes larger errors:

$$\text{SSE} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n e_i^2$$

![Sum of Squared Errors Visualization](../plots/ml_basics/image_21_Im22.png)

#### Visualizing Least Squares Optimization
Fitting a regression line is the process of finding parameters that minimize the total area of the error squares.

*   **Best Fit (Minimized SSE):** The regression line passes directly through the center of the point cloud, making the total area of the error squares as small as possible.
*   **Poor Fit (High SSE):** The line deviates from the true trend, increasing the areas of the individual error squares.
*   **Worst Fit (Maximum SSE):** The line is completely misaligned, creating massive error squares.

| Best Fit (Optimal) | Poor Fit (High Loss) | Worst Fit (Very High Loss) |
|:---:|:---:|:---:|
| ![Best Fit Squares](../plots/ml_basics/image_15_Im16.png) | ![Poor Fit Squares](../plots/ml_basics/image_17_Im18.png) | ![Worst Fit Squares](../plots/ml_basics/image_19_Im20.png) |

### 3.4 Mean Squared Error (MSE)
Dividing the SSE by the number of data points $n$ gives the **Mean Squared Error (MSE)**, which represents the average squared distance of predictions from the target labels:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat{Y}_i)^2$$

![Mean Squared Error Formula](../plots/ml_basics/image_97_Im98.png)

---

## 4. Optimization: Gradient Descent & Chain Rule

To find the parameters that minimize the MSE loss, models use an iterative optimization algorithm called **Gradient Descent**.

### 4.1 Loss Surface and Updates
For linear regression, the loss function plotted against parameters (like weight $m$ and bias $b$) forms a convex 3D parabolic bowl. The bottom of this bowl represents the global minimum loss.

At each step, we calculate the partial derivatives of the loss with respect to each parameter to find the direction of steepest ascent. We then move in the opposite direction (steepest descent) by subtracting the gradient scaled by a **learning rate** ($\alpha$):

$$\theta_j := \theta_j - \alpha \frac{\partial L}{\partial \theta_j}$$

![3D Parabolic Loss Surface Gradient Descent Path](../plots/ml_basics/image_23_Im24.png)

### 4.2 Linear Regression Backpropagation via the Chain Rule
For a single input data point with feature $\text{area}$, target $y$, prediction $\hat{y}$, and weights $w_1, w_2, b$:

$$\hat{y} = w_1 \cdot \text{area} + w_2 \cdot \text{br} + b$$

$$\text{Error } e = y - \hat{y}$$

$$\text{Loss } L = e^2$$

To calculate the gradient of the loss with respect to the weight $w_1$, we apply the calculus **Chain Rule**:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_1}$$

Let's calculate each individual derivative:
1.  **Loss with respect to Error:** $\frac{\partial L}{\partial e} = \frac{d}{de} (e^2) = 2e$
2.  **Error with respect to Prediction:** $\frac{\partial e}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} (y - \hat{y}) = -1$
3.  **Prediction with respect to Weight $w_1$:** $\frac{\partial \hat{y}}{\partial w_1} = \frac{\partial}{\partial w_1} (w_1 \cdot \text{area} + w_2 \cdot \text{br} + b) = \text{area}$

Multiplying them together yields the gradient:

$$\frac{\partial L}{\partial w_1} = 2e \cdot (-1) \cdot \text{area} = -2e \cdot \text{area}$$

$$\frac{\partial L}{\partial w_1} = -2(y - \hat{y}) \cdot \text{area}$$

![Chain Rule Graph Representation](../plots/ml_basics/image_1_Im2.jpg)

---

## 5. Regression Performance Metrics

To evaluate how well a regression model fits the data, several metrics are used:

![Performance Metrics Table](../plots/ml_basics/image_25_Im26.png)

*   **Coefficient of Determination ($R^2$):** Measures the proportion of variance in the dependent variable that is predictable from the independent variables. Scores range from $0$ to $1$:
    
    $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
    
*   **Standard Error of the Estimate:** Measures the standard deviation of the residuals, indicating the average distance that the observed values fall from the regression line.
*   **Prediction Interval:** An estimate of the interval in which future individual observations will fall with a certain probability (e.g., $95\%$), expressing prediction uncertainty as a range rather than a single point.
*   **Statistical Significance (p-value):** Used to determine if the relationship observed between variables is statistically significant or if it could have occurred by random chance. A threshold of $p < 0.05$ is commonly used to reject the null hypothesis.

---

## 6. Logistic Regression & Binary Classification

**Logistic Regression** is a supervised learning classification algorithm used to predict the probability of a binary outcome ($y \in \{0, 1\}$).

### 6.1 The Sigmoid Function
Instead of fitting a straight line which can output values from $-\infty$ to $+\infty$, logistic regression passes the linear combination of inputs through the **Sigmoid (logistic) function**, wrapping the output values strictly between $0$ and $1$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where the log-odds input $z$ is:

$$z = \beta_1 x + \beta_0$$

Thus, the probability model is:

$$f(x) = \frac{1}{1 + e^{-(\beta_1 x + \beta_0)}}$$

![Raw Classification Points](../plots/ml_basics/image_27_Im28.png)

![Sigmoid Curve Fit](../plots/ml_basics/image_29_Im30.png)

![Sigmoid Mathematical Formula Graph](../plots/ml_basics/image_35_Im36.png)

### 6.2 Multiple Logistic Regression
When there are multiple predictor features, the log-odds linear boundary generalizes in 3D space to form a sigmoid probability surface:

$$p(x_1, x_2) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2)}}$$

![3D Sigmoid Probability Surface](../plots/ml_basics/image_37_Im38.png)

### 6.3 Decision Boundary and Shaded Regions
To perform binary classification, the predicted probability $p$ is compared against a decision threshold (typically $0.5$ or $0.75$).

$$\text{Class} = \begin{cases} \text{TRUE (1)} & \text{if } f(x) \ge \text{Threshold} \\ \text{FALSE (0)} & \text{if } f(x) < \text{Threshold} \end{cases}$$

For instance, using a threshold of $0.75$ at input $x = 6.2$:

| Sigmoid Curve with Query Point | Classified Shaded Regions |
|:---:|:---:|
| ![Sigmoid Threshold Mapping](../plots/ml_basics/image_31_Im32.png) | ![TRUE/FALSE Classification Thresholds](../plots/ml_basics/image_33_Im34.png) |

---

## 7. K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a simple, non-parametric, instance-based supervised learning algorithm used for classification and regression. It makes predictions for a query point based on the labels of its closest neighboring data points.

![KNN Classification Concept](../plots/ml_basics/image_73_Im74.png)

### 7.1 Euclidean Distance Metric
To find the "nearest" neighbors, KNN calculates the geometric distance between points. The most common metric is **Euclidean Distance**:

$$d(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{\sum_{j=1}^D (x_{1j} - x_{2j})^2}$$

In 2D space:

$$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

![Euclidean Distance Diagram](../plots/ml_basics/image_71_Im72.png)

### 7.2 Step-by-Step KNN Classification ($k=5$)
Here is a visual step-by-step example showing how KNN classifies a new unlabelled data point:

1.  **Define the Problem:** We have a dataset of red and blue data points, and we introduce a new unlabelled yellow query point.
2.  **Determine the Number of Neighbors ($k$):** We select the value of $k$ (e.g., $k = 5$), which specifies the number of nearby points to consider.
3.  **Draw Search Boundary:** We identify the closest 5 neighbors to our query point. Visually, this is equivalent to expanding a circle centered on the query point until it encloses exactly 5 data points.
4.  **Measure Distances:** Calculate the Euclidean distances from the query point to all points in the dataset and identify the 5 smallest values.
5.  **Majority Vote:** Count the classes of the 5 nearest neighbors. The query point is assigned to the class with the highest frequency.
    *   Red Neighbors: $3$
    *   Blue Neighbors: $2$
    *   **Result:** The query point is classified as **Red**!

| Step 1: Query Input | Step 2: Set K | Step 3: Search Circle |
|:---:|:---:|:---:|
| ![KNN Query](../plots/ml_basics/image_53_Im54.png) | ![KNN K=5 Header](../plots/ml_basics/image_61_Im62.png) | ![KNN Search Boundary](../plots/ml_basics/image_57_Im58.png) |

| Step 4: Measure Distances | Step 5: Majority Vote |
|:---:|:---:|
| ![KNN Distance Lines](../plots/ml_basics/image_65_Im66.png) | ![KNN Voting Result](../plots/ml_basics/image_69_Im70.png) |

---

## 8. K-Means Clustering

**K-Means Clustering** is an unsupervised learning algorithm used to partition a dataset into $K$ distinct, non-overlapping subgroups (clusters). It groups data points so that points in the same cluster are as similar as possible, while points in different clusters are distinct.

![Before and After K-Means](../plots/ml_basics/image_75_Im76.png)

### 8.1 The K-Means Objective Function
K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also known as inertia:

$$J = \sum_{k=1}^K \sum_{\mathbf{x}_i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where:
*   $K$ is the number of clusters.
*   $S_k$ is the set of data points assigned to cluster $k$.
*   $\boldsymbol{\mu}_k$ is the centroid (mean vector) of cluster $k$.

### 8.2 Step-by-Step K-Means Iterations
K-Means uses an expectation-maximization heuristic that alternates between two steps: assigning points to clusters, and updating centroids.

#### Step 1: Choose Initial Centroids
Randomly select $K$ data points from the dataset to act as the initial cluster centers (centroids).

$$\text{Initial Centroids: } \{\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \dots, \boldsymbol{\mu}_K\}$$

![K-Means Step 1: Initial Centroids](../plots/ml_basics/image_77_Im78.png)

#### Step 2: Assign Points to Nearest Centroid
Calculate the distance from each data point $\mathbf{x}_i$ to all $K$ centroids, and assign the point to the cluster of the closest centroid.

$$S_k^{(t)} = \left\{ \mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|^2 \le \|\mathbf{x}_i - \boldsymbol{\mu}_j^{(t)}\|^2 \quad \forall j, 1 \le j \le K \right\}$$

![K-Means Step 2: Assign Points](../plots/ml_basics/image_79_Im80.png)

#### Step 3: Update Centroids
Recalculate the position of each centroid as the arithmetic mean of all data points currently assigned to that cluster.

$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|S_k^{(t)}|} \sum_{\mathbf{x}_i \in S_k^{(t)}} \mathbf{x}_i$$

![K-Means Step 3: Update Centroids](../plots/ml_basics/image_81_Im82.png)

#### Step 4: Repeat Until Convergence
Repeat Steps 2 and 3 iteratively. The algorithm converges when the centroids stabilize and do not move further, or when cluster assignments stop changing.

![K-Means Step 4: Convergence](../plots/ml_basics/image_83_Im84.png)
