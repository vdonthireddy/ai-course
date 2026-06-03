# Visualizing Backpropagation: A Comprehensive Illustrated Guide

This document provides a detailed, step-by-step mathematical and visual explanation of the **Backpropagation** algorithm and **Gradient Descent** in deep neural networks. By examining the flow of information forward and errors backward, we show how networks calculate parameter gradients using the **Chain Rule** to learn from data.

All illustrations are referenced from the local directory: [plots/backprop/](../plots/backprop/).

---

## 1. Introduction: The Simplest Neural Network (1 Neuron, 1 Weight)

To understand backpropagation, we begin with the simplest possible neural network: a single input, a single weight, a single output neuron, and no bias.

![Simplest Neural Network Intro](../plots/backprop/IMG_3194.PNG)

### 1.1 The Forward Pass
In our toy model, we feed an input ($i$) through a connection with weight ($w$) to produce a predicted output activation ($a$):

$$a = i \cdot w$$

Suppose we set our initial values as:
*   Input ($i$) = $1.5$
*   Weight ($w$) = $0.8$

Using these values, we perform a **forward pass** to calculate the prediction:

$$a = 1.5 \cdot 0.8 = 1.2$$

![Toy Model Forward Pass](../plots/backprop/IMG_3195.PNG)

### 1.2 Defining the Cost (Loss)
To measure how accurate our prediction is, we compare the prediction ($a$) to a desired target value ($y$). Suppose the target value is:

$$y = 0.5$$

Because our prediction $a = 1.2$ is larger than the target $y = 0.5$, the model has made an error.

![Introducing Target y](../plots/backprop/IMG_3196.PNG)

We quantify this error using a **Cost Function** ($C$). In this case, we use the **Squared Error** cost function:

$$C = (a - y)^2$$

Substituting our values ($a = 1.2$ and $y = 0.5$) yields the cost:

$$C = (1.2 - 0.5)^2 = 0.7^2 = 0.49$$

The goal of learning is to find a weight ($w$) that makes this cost as close to $0$ as possible.

![Defining Cost C](../plots/backprop/IMG_3197.PNG)

### 1.3 Nudging the Weight
To see how we can minimize the cost, let's explore what happens when we modify ("nudge") the weight $w$:

*   **Case A: Increasing $w$**
    If we increase $w$ from $0.8$ to $0.9$:
    *   The prediction increases: $a = 1.5 \cdot 0.9 = 1.35$
    *   The error increases: $a - y = 1.35 - 0.5 = 0.85$
    *   The cost increases: $C = 0.85^2 = 0.7225$ (up from $0.49$)
    
    This tells us that increasing the weight moves us further away from our goal.

![Increasing Weight raises Cost](../plots/backprop/IMG_3198.PNG)

*   **Case B: Decreasing $w$**
    If we decrease $w$ from $0.8$ to $0.7$:
    *   The prediction decreases: $a = 1.5 \cdot 0.7 = 1.05$
    *   The error decreases: $a - y = 1.05 - 0.5 = 0.55$
    *   The cost decreases: $C = 0.55^2 = 0.3025$ (down from $0.49$)
    
    This shows that decreasing the weight successfully reduces the cost.

![Decreasing Weight lowers Cost](../plots/backprop/IMG_3199.PNG)

### 1.4 The Optimization Challenge
While manually trying weight values works for a single parameter, complex networks contain millions of parameters. We need a mathematical way to calculate exactly how a nudge to any given weight affects the final cost.

This is the core problem of optimization: how do we calculate the derivative of the cost with respect to the weight ($\frac{\partial C}{\partial w}$)?

![The Optimization Question](../plots/backprop/IMG_3200.PNG)

---

## 2. Connecting Parameters to Cost: The Chain Rule

In our toy model, the cost ($C$) does not directly depend on the weight ($w$). Instead, we have a chain of dependencies:
1.  The weight $w$ determines the activation output $a$.
2.  The activation output $a$ determines the cost $C$.

This nested relationship can be written as a composite function: $C(a(w))$. We can visualize these dependencies as two separate mathematical curves:
*   The cost curve $C(a)$ is a parabola centered at the target value $y$.
*   The activation curve $a(w)$ is a straight line with a slope equal to the input $i$.

![Cost and Activation Curves](../plots/backprop/IMG_3202.PNG)

To find the rate of change of the cost with respect to the weight, we use the **Chain Rule** of calculus. The chain rule states that to find the derivative of a nested composite function, we multiply the local derivatives along the dependency chain:

$$\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial w}$$

*   $\frac{\partial C}{\partial a}$ represents how a nudge to the activation $a$ affects the cost $C$.
*   $\frac{\partial a}{\partial w}$ represents how a nudge to the weight $w$ affects the activation $a$.

By multiplying these two factors, we determine how a nudge to $w$ propagates through the intermediate variable $a$ to ultimately affect the cost $C$.

![Chain Rule Formulation](../plots/backprop/IMG_3203.PNG)

---

## 3. Generalizing to Multi-Input Neurons and Activation Functions

To move from our toy model to deep neural networks, we must build a generalized mathematical model of an artificial neuron.

### 3.1 Anatomy of a Neuron
A single neuron in a neural network layer receives outputs from multiple neurons in the preceding layer.

![Neuron Inputs and Weights](../plots/backprop/IMG_3206.PNG)

Let's break down the mathematical assembly of a neuron step-by-step:

1.  **Weighted Sum (Pre-activation Input):**
    Each input $x_i$ is multiplied by its corresponding weight $\omega_i$. A constant **bias** ($b$) is added to shift the activation threshold. This combined sum is called the **pre-activation net input** (denoted $\hat{x}$):
    
    $$\hat{x} = b + \sum_{i} x_i \omega_i$$

![Adding Bias and Summing](../plots/backprop/IMG_3207.PNG)

2.  **Activation Function:**
    To introduce non-linearity into the network, the pre-activation input $\hat{x}$ is passed through an **activation function** (denoted $\phi$):

$$\phi(\hat{x})$$

![Applying Activation Function](../plots/backprop/IMG_3208.PNG)

3.  **Final Neuron Output:**
    The final output activation ($y$) of the neuron is:
    
    $$y = \phi(\hat{x}) = \phi\left(b + \sum_{i} x_i \omega_i\right)$$

![Generalized Neuron Formula](../plots/backprop/IMG_3210.PNG)

### 3.2 Common Activation Functions
Activation functions determine the behavior and capabilities of the neural network. Here are the three most common activation functions and their derivatives:

#### 1. Linear Activation Function
The output is directly proportional to the input:

$$\phi(x) = x$$

Its derivative is constant:

$$\phi'(x) = 1$$

![Linear Activation Function](../plots/backprop/IMG_3211.PNG)

#### 2. Rectified Linear Unit (ReLU)
The output is $0$ for negative inputs and equals the input for positive inputs:

$$\phi(x) = \max(0, x)$$

Its derivative is a step function:

$$\phi'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

*(Note: The derivative is technically undefined at $x=0$, but in practice, it is set to $0$.)*

![ReLU Activation Function](../plots/backprop/IMG_3212.PNG)

#### 3. Sigmoid (Logistic) Function
The output is squashed to a smooth S-curve between $0$ and $1$:

$$\phi(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$$

Its derivative can be expressed elegantly in terms of its output:

$$\phi'(x) = \phi(x)(1 - \phi(x))$$

![Sigmoid Activation Function](../plots/backprop/IMG_3213.PNG)

---

### 3.3 Visualizing Activation Implementations
Let's see how a neuron behaves under each of these activation functions:

*   **Linear Neuron:** Passing a linear combination through a linear activation simply returns the weighted sum plus bias. It cannot learn non-linear relationships.

![Neuron with Linear Activation](../plots/backprop/IMG_3214.PNG)

*   **ReLU Neuron:** If the pre-activation sum is negative, the neuron output is completely deactivated ($0$). If positive, the output scales linearly.

![Neuron with ReLU Activation](../plots/backprop/IMG_3215.PNG)

*   **Sigmoid Neuron:** Regardless of how large or small the pre-activation input is, the output is smoothly mapped to the range $(0, 1)$.

![Neuron with Sigmoid Activation](../plots/backprop/IMG_3216.PNG)

Below is the summary table of these three core activation functions:

| Activation | Equation | Derivative |
| :--- | :--- | :--- |
| **Linear** | $\phi(x) = x$ | $\phi'(x) = 1$ |
| **ReLU** | $\phi(x) = \max(0, x)$ | $\phi'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$ |
| **Sigmoid** | $\phi(x) = \frac{1}{1 + e^{-x}}$ | $\phi'(x) = \phi(x)(1 - \phi(x))$ |

![Activation Functions Table](../plots/backprop/IMG_3217.PNG)

### 3.4 Standard Layer Notation
To write mathematical derivations for multi-layer networks, we establish standard indexing notation:
*   Let neuron $i$ belong to an earlier layer, producing output activation $y_i$.
*   Let neuron $j$ belong to the next layer.
*   The connection between neuron $i$ and neuron $j$ has weight $\omega_{ij}$.
*   Neuron $j$ has bias $b_j$ and pre-activation net input $\hat{x}_j = b_j + \sum_k y_k \omega_{kj}$.
*   The final output of neuron $j$ is $y_j = \phi(\hat{x}_j)$.

![Standard Layer Notation](../plots/backprop/IMG_3218.PNG)

---

## 4. Multi-Layer Networks and Optimization Landscapes

When we stack multiple layers of these neurons together, we build a **Multi-Layer Perceptron (MLP)** or deep feedforward neural network.

![Multi-Layer Network Architecture](../plots/backprop/IMG_3219.PNG)

### 4.1 The Global Loss Function
For a network with $n$ layers, the inputs pass forward through the layers to produce a final network output vector $\mathbf{y}_n$. We evaluate the accuracy of the entire network using a global **Loss Function** ($L$):

$$Loss = L(\mathbf{y}_n)$$

The loss function compares the network's predictions $\mathbf{y}_n$ against the true training labels.

![Global Loss Function](../plots/backprop/IMG_3220.PNG)

### 4.2 Gradient Descent
To minimize the global loss $L$, we update all network weights iteratively in the opposite direction of the gradient of the loss. This optimization algorithm is called **Gradient Descent**:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\partial L}{\partial \mathbf{w}}$$

Where:
*   $\mathbf{w}_t$ represents the weights at the current step $t$.
*   $\mathbf{w}_{t+1}$ represents the updated weights at step $t+1$.
*   $\eta$ (eta) is the **learning rate**, controlling the step size of each update.
*   $\frac{\partial L}{\partial \mathbf{w}}$ is the gradient vector containing the partial derivatives of the loss with respect to all weights.

![Gradient Descent Update](../plots/backprop/IMG_3221.PNG)

*   **The Role of the Learning Rate ($\eta$):**
    *   If $\eta$ is **too small**, weight updates are minuscule, causing training to take an extremely long time.
    *   If $\eta$ is **too large**, the updates can overshoot the minimum, causing the optimization path to oscillate wildly or even diverge entirely.

![Learning Rate Effects](../plots/backprop/IMG_3222.PNG)

*   **Navigating Loss Landscapes:**
    In real-world networks, the loss landscape is complex, high-dimensional, and non-convex. It contains multiple peaks, valleys, **local minima** (suboptimal low points), and a **global minimum** (the absolute lowest point of the loss function). Our goal is to guide the weights toward the global minimum, though gradient descent is susceptible to getting trapped in local minima or saddle points.

![Local vs. Global Minima](../plots/backprop/IMG_3223.PNG)

By taking small, sequential steps proportional to the negative gradient, the parameters slide down the loss curve toward a minimum.

![Descending the Loss Curve](../plots/backprop/IMG_3225.PNG)

---

## 5. Step-by-Step Chain Rule Derivation for Layer Weights

To execute gradient descent, we must compute $\frac{\partial L}{\partial \omega_{ij}}$ for every single weight in the network. Let's derive this derivative step-by-step using the Chain Rule.

### 5.1 Setting Up the Chain
We want to find how a change in the weight $\omega_{ij}$ (connecting neuron $i$ to neuron $j$) affects the global loss $L$:

$$\frac{\partial L}{\partial \omega_{ij}} = ?$$

![Weight Derivative Goal](../plots/backprop/IMG_3226.PNG)

Because $\omega_{ij}$ only affects the loss by contributing to the pre-activation net input $\hat{x}_j$ of neuron $j$, we split the derivative using the Chain Rule:

$$\frac{\partial L}{\partial \omega_{ij}} = \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \omega_{ij}}$$

*   $\frac{\partial L}{\partial \hat{x}_j}$ is the rate of change of the loss with respect to the pre-activation input of neuron $j$. This term is often called the **error term** of neuron $j$ (denoted $\delta_j$).
*   $\frac{\partial \hat{x}_j}{\partial \omega_{ij}}$ is the rate of change of the pre-activation input with respect to the weight.

![Splitting Weight Derivative](../plots/backprop/IMG_3228.PNG)

### 5.2 Evaluating the Second Term: $\frac{\partial \hat{x}_j}{\partial \omega_{ij}}$
Recall that the pre-activation net input of neuron $j$ is:

$$\hat{x}_j = b_j + \sum_{k} y_k \omega_{kj}$$

If we take the partial derivative of this sum with respect to the specific weight $\omega_{ij}$, all other terms in the sum are treated as constants and drop out:

$$\frac{\partial \hat{x}_j}{\partial \omega_{ij}} = \frac{\partial}{\partial \omega_{ij}} \left( b_j + y_1 \omega_{1j} + \dots + y_i \omega_{ij} + \dots \right) = y_i$$

This shows that the rate of change of the pre-activation input with respect to the weight is simply the **activation output of the sending neuron** ($y_i$):

$$\frac{\partial \hat{x}_j}{\partial \omega_{ij}} = y_i$$

![Calculating dx_hat / d_omega](../plots/backprop/IMG_3230.PNG)

### 5.3 Evaluating the First Term: $\frac{\partial L}{\partial \hat{x}_j}$
Now we evaluate the error term $\frac{\partial L}{\partial \hat{x}_j}$. The pre-activation input $\hat{x}_j$ only affects the network output by first passing through the activation function to become $y_j$. Therefore, we apply the Chain Rule again:

$$\frac{\partial L}{\partial \hat{x}_j} = \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial \hat{x}_j}$$

*   $\frac{\partial L}{\partial y_j}$ is the rate of change of the loss with respect to the post-activation output of neuron $j$.
*   $\frac{\partial y_j}{\partial \hat{x}_j}$ is the derivative of the activation function evaluated at $\hat{x}_j$.

![Splitting dx_hat Derivative](../plots/backprop/IMG_3232.PNG)

If we assume the activation function is the **Sigmoid function**:

$$y_j = \sigma(\hat{x}_j)$$

Its derivative is:

$$\frac{\partial y_j}{\partial \hat{x}_j} = y_j(1 - y_j)$$

![Sigmoid Derivative Evaluation](../plots/backprop/IMG_3233.PNG)

Substituting this derivative back into our error term equation yields:

$$\frac{\partial L}{\partial \hat{x}_j} = \frac{\partial L}{\partial y_j} y_j(1 - y_j)$$

![Pre-activation Gradient Formula](../plots/backprop/IMG_3234.PNG)

### 5.4 Combining the Terms
Now, we substitute our results from **Section 5.2** and **Section 5.3** back into our primary weight derivative equation:

$$\frac{\partial L}{\partial \omega_{ij}} = \left( \frac{\partial L}{\partial \hat{x}_j} \right) \cdot \left( \frac{\partial \hat{x}_j}{\partial \omega_{ij}} \right)$$

$$\frac{\partial L}{\partial \omega_{ij}} = \left( \frac{\partial L}{\partial y_j} y_j(1 - y_j) \right) \cdot y_i$$

Rearranging the terms, we get the complete derivative of the loss with respect to the weight $\omega_{ij}$:

$$\frac{\partial L}{\partial \omega_{ij}} = \frac{\partial L}{\partial y_j} y_j(1 - y_j) y_i$$

![Combining the Terms](../plots/backprop/IMG_3235.PNG)

This elegant formula tells us that the gradient of a weight is the product of:
1.  The downstream loss gradient ($\frac{\partial L}{\partial y_j}$).
2.  The derivative of the activation function of the receiving neuron ($y_j(1-y_j)$).
3.  The incoming activation from the sending neuron ($y_i$).

![Weight Gradient Summary](../plots/backprop/IMG_3236.PNG)

---

## 6. Error Propagation and the General Backpropagation Algorithm

The weight gradient derivation in Section 5 requires knowing $\frac{\partial L}{\partial y_j}$ (the gradient of the loss with respect to the neuron's output).
*   If neuron $j$ is in the **output layer**, computing $\frac{\partial L}{\partial y_j}$ is straightforward because the loss function is defined directly in terms of the output layer activations.
*   If neuron $j$ is in a **hidden layer**, computing $\frac{\partial L}{\partial y_j}$ is more complex because hidden neurons do not directly participate in the loss function. We must propagate the errors backwards from the output layer.

![Hidden Activation Gradients](../plots/backprop/IMG_3237.PNG)

### 6.1 Branching Downstream Paths
Let's find the derivative of the loss with respect to the activation output of a hidden neuron $i$, denoted $\frac{\partial L}{\partial y_i}$.

In the forward pass, the activation output $y_i$ of neuron $i$ is distributed forward to feed the pre-activation inputs $\hat{x}_k$ of **multiple neurons $k$** in the next layer.

![Branching Downstream Connections](../plots/backprop/IMG_3238.PNG)

Because $y_i$ influences the loss through multiple parallel paths, we must apply the multi-variable Chain Rule. The total derivative of the loss with respect to $y_i$ is the **sum of the derivatives across all downstream branches**:

$$\frac{\partial L}{\partial y_i} = \sum_{k} \frac{\partial L}{\partial \hat{x}_k} \cdot \frac{\partial \hat{x}_k}{\partial y_i}$$

Where the summation index $k$ runs over all neurons in the next layer that receive input from neuron $i$.

![Summing Downstream Paths](../plots/backprop/IMG_3239.PNG)

### 6.2 Evaluating the Connection Term: $\frac{\partial \hat{x}_k}{\partial y_i}$
Recall that the pre-activation input for any downstream neuron $k$ is:

$$\hat{x}_k = b_k + \sum_{p} y_p \omega_{pk}$$

Taking the partial derivative of this sum with respect to the specific input activation $y_i$ isolates its corresponding connection weight:

$$\frac{\partial \hat{x}_k}{\partial y_i} = \omega_{ik}$$

![Evaluating connection derivative](../plots/backprop/IMG_3241.PNG)

Substituting $\frac{\partial \hat{x}_k}{\partial y_i} = \omega_{ik}$ back into the summation yields:

$$\frac{\partial L}{\partial y_i} = \sum_{k} \frac{\partial L}{\partial \hat{x}_k} \omega_{ik}$$

This equation shows that the error gradient propagates backward from the next layer's pre-activation inputs ($\frac{\partial L}{\partial \hat{x}_k}$) back to the current layer's output ($y_i$), scaled by the connection weights ($\omega_{ik}$).

![Substituting connection term](../plots/backprop/IMG_3242.PNG)

### 6.3 Expanding the Propagation Equation
We can expand the pre-activation gradient term $\frac{\partial L}{\partial \hat{x}_k}$ using the activation derivative of the downstream neurons:

$$\frac{\partial L}{\partial \hat{x}_k} = \frac{\partial L}{\partial y_k} \cdot \frac{\partial y_k}{\partial \hat{x}_k}$$

Substituting this into our propagation formula gives:

$$\frac{\partial L}{\partial y_i} = \sum_{k} \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial \hat{x}_k} \omega_{ik}$$

This is the central equation for propagating error gradients backward through the hidden layers of a neural network.

![Expanding error propagation](../plots/backprop/IMG_3243.PNG)

---

### 6.4 The Two Core Equations of Backpropagation
For any two adjacent layers in a neural network—letting sending nodes in the current layer be indexed by $p$ and receiving nodes in the subsequent layer be indexed by $q$—we summarize the backpropagation algorithm using two primary equations:

#### Equation 1: Error Propagation
We compute the loss gradient with respect to the output activation $y_p$ of a sending node by summing the gradients from all receiving nodes $q$, scaled by their connection weights:

$$\frac{\partial L}{\partial y_p} = \sum_{q} \frac{\partial L}{\partial y_q} \frac{\partial y_q}{\partial \hat{x}_q} \omega_{pq}$$

#### Equation 2: Weight Gradient Calculation
Using the propagated activation gradient, we calculate the gradient of the loss with respect to the weight $\omega_{pq}$ connecting the two nodes:

$$\frac{\partial L}{\partial \omega_{pq}} = \frac{\partial L}{\partial y_q} \frac{\partial y_q}{\partial \hat{x}_q} y_p$$

![Two Core Equations Summarized](../plots/backprop/IMG_3245.PNG)

We can visualize how these two equations correspond to the physical connections of the network:
*   **Equation 1** sums up the backward influence of a node across all its outgoing connections.
*   **Equation 2** calculates the update gradient for the weight between two nodes by multiplying the activation of the sending node and the error of the receiving node.

![Visualizing Equations in Network](../plots/backprop/IMG_3246.PNG)

---

### 6.5 The Complete Gradient Flow
We execute the backpropagation algorithm layer-by-layer backwards through the network:
1.  We start at the output layer $n$, calculating the initial activation gradient $\left(\frac{\partial L}{\partial y}\right)_n$.
2.  We apply **Equation 1** (labeled $\textcircled{1}$) to propagate the activation gradients backward from layer $n$ to $n-1$, then to $n-2$, all the way to the input layer.
3.  Simultaneously, at each layer, we apply **Equation 2** (labeled $\textcircled{2}$) to compute the weight gradients $\left(\frac{\partial L}{\partial \omega}\right)$ using the local activation gradients.

```
Activation Gradients (Propagated Backwards):
(∂L/∂y)_1  <--- ... <---  (∂L/∂y)_{n-2}  <---  (∂L/∂y)_{n-1}  <---  (∂L/∂y)_n
    |                           |                  |                  |
    | (Equation 2)              | (Equation 2)     | (Equation 2)     | (Equation 2, labeled 2)
    v                           v                  v                  v
(∂L/∂ω)_1                 (∂L/∂ω)_{n-2}      (∂L/∂ω)_{n-1}      (∂L/∂ω)_n
```

![Gradient Flow Chart](../plots/backprop/IMG_3247.PNG)

### 6.6 The Weight Update
Once the weight gradients $\frac{\partial L}{\partial \omega}$ have been calculated for all layers, we apply them to update the network weights using Stochastic Gradient Descent (SGD) or a similar optimizer:

$$\omega_{t+1} = \omega_t - \eta \frac{\partial L}{\partial \omega}$$

By repeating this process of forward propagation, cost calculation, backward error propagation, and weight updates over many epochs, the neural network learns to fit the training data and solve complex predictive tasks.

![Parameter Updates with SGD](../plots/backprop/IMG_3248.PNG)
