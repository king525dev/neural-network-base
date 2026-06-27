
# Neural Networks From Scratch

> *"A neural network is just a squiggle-fitting machine that learns from examples."*

This repository contains a pure JavaScript implementation of a neural network that learns the XOR (exclusive OR), AND and OR function. It visualizes the network structure and training in real-time.


---

## Contents
1. [What is a Neural Network?](#1-what-is-a-neural-network)
2. [The Building Blocks (Weights, Biases, Activations)](#2-the-building-blocks-weights-biases-activations)
3. [The AND Gate: A Single Neuron (The Perceptron)](#3-the-and-gate-a-single-neuron-the-perceptron)
4. [The XOR Problem: Why We Need a Hidden Layer](#4-the-xor-problem-why-we-need-a-hidden-layer)
5. [Network Architecture for XOR (2-2-1)](#5-network-architecture-for-xor-2-2-1)
6. [The Forward Pass (Math & Code)](#6-the-forward-pass-math--code)
7. [The Loss Function (How wrong are we?)](#7-the-loss-function-how-wrong-are-we)
8. [Backpropagation (The "Blame Game")](#8-backpropagation-the-blame-game)
9. [Gradient Descent (Walking Down the Mountain)](#9-gradient-descent-walking-down-the-mountain)
10. [Why Tanh? (Choosing the Right Activation)](#10-why-tanh-choosing-the-right-activation)
11. [The Training Loop](#11-the-training-loop)
12. [How to Run](#12-how-to-run)

---

## 1. What is a Neural Network?

Imagine you are trying to fit a curve to some data points. A straight line (linear regression) doesn't work. A neural network is a **flexible mathematical function** that can bend and twist to fit complex patterns.

**The Analogy:** 
Think of it like a factory assembly line. Raw materials (Inputs) go through several machines (Neurons) that process them. Each machine has **dials** (Weights) and a **default setting** (Bias). By tweaking these dials, we change how the final product (Output) turns out. Training is the process of finding the perfect dial settings.

---

## 2. The Building Blocks (Weights, Biases, Activations)

### A. The Neuron (The Mathematical Unit)
A neuron takes inputs, multiplies them by **Weights**, adds a **Bias**, and then passes the result through an **Activation Function**.

**Formula:**
\[
z = (x_1 \times w_1) + (x_2 \times w_2) + b
\]
\[
h = \text{Activation}(z)
\]

### B. Weights (The "Volume Knobs")
Weights determine *how much* a neuron listens to a specific input.

- **High positive weight:** "This input is very important."
- **Zero weight:** "I ignore this input."
- **Negative weight:** "This input means the opposite."

### C. Biases (The "Default Tendency")
The Bias is a constant added to the neuron. It acts like a **thermostat**.

- If the bias is high, the neuron tends to activate even without strong input.
- If the bias is negative, the neuron stays "off" until it receives convincing evidence.

**Why only one bias per neuron?**
A neuron has \( N \) inputs and \( N \) weights, but only **1 bias**.
If we had two biases, \( b_1 + b_2 \), they would just combine into a single number \( b \). Multiple biases add no extra power.

### D. Activation Functions (The "Dimmer Switch")
Without an activation function, the network is just a bunch of linear equations (straight lines stacked together). Stacking straight lines gives you... a straight line.

The activation function introduces **non-linearity**. It bends the line.

**Common Activations in this project:**
- **Sigmoid:** Squashes values between 0 and 1. Great for probabilities.
- **Tanh:** Squashes values between -1 and 1. Zero-centered (better for gradients).
- **Softplus:** Smooth version of ReLU (always positive).

---

## 3. The AND Gate: A Single Neuron (The Perceptron)

Before XOR, let's look at the simplest problem: **AND**.

| Input X1 | Input X2 | Output |
| :--- | :--- | :--- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**The Analogy: The Strict Bouncer**
You can enter the club (Output=1) only if you have **both** a Ticket (X1) **AND** an ID (X2).

**The Math:**
We can solve this with **one single neuron**.
Let \( z = (x_1 \times w_1) + (x_2 \times w_2) + b \).
If \( z > 0 \), output 1. Else, output 0.

Can we guess the weights?
Yes! Let \( w_1 = 1 \), \( w_2 = 1 \), and \( b = -1.5 \).

- (0,0): \( 0 + 0 - 1.5 = -1.5 \) → 0
- (1,0): \( 1 + 0 - 1.5 = -0.5 \) → 0
- (0,1): \( 0 + 1 - 1.5 = -0.5 \) → 0
- (1,1): \( 1 + 1 - 1.5 = 0.5 \) → 1 ✅

**Key Insight:** AND is **linearly separable**. You can draw a straight line to separate the 0s from the 1s.

---

## 4. The XOR Problem: Why We Need a Hidden Layer

XOR (Exclusive OR) is different:

| Input X1 | Input X2 | Output |
| :--- | :--- | :--- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Plot this:**
- (0,0)=0
- (1,1)=0
- (0,1)=1
- (1,0)=1

The 1s are diagonally opposite. The 0s are diagonally opposite.

**Problem:** You **cannot** draw a single straight line to separate them. This is called **Non-Linearly Separable**.

**Solution:** We need to **fold the paper** (create a non-linear boundary). To do this, we use a **Hidden Layer**.

**The Analogy: The Two Detectives**
- **Detective 1 (H1):** Looks for "Pizza but NO Music" (X1=1, X2=0).
- **Detective 2 (H2):** Looks for "Music but NO Pizza" (X1=0, X2=1).
- **The Boss (OUT):** Listens to both Detectives. If *either* says "Yes", the Boss says "FUN!" (Output=1).

The Hidden Layer creates these "Detectives" automatically by learning the right weights.

---

## 5. Network Architecture for XOR (2-2-1)

Our network has:
- **2 Inputs:** \( X_1, X_2 \)
- **1 Hidden Layer:** 2 Neurons (\( H_1, H_2 \))
- **1 Output Layer:** 1 Neuron (\( OUT \))

**The Connections (Weights):**
- Input → Hidden: \( w_{11}, w_{12}, w_{21}, w_{22} \)
- Hidden → Output: \( w_{o1}, w_{o2} \)
- Biases: \( b_{h1}, b_{h2}, b_{out} \)

**Why 2 Hidden Neurons?**
XOR requires detecting two diagonal patterns (top-left and bottom-right). We need one neuron to detect each pattern, hence 2 neurons.

---

## 6. The Forward Pass (Math & Code)

The Forward Pass is the calculation the network does to make a prediction.

### Step 1: Hidden Layer Pre-Activations
We calculate the weighted sum for each hidden neuron.

\[
z_1 = (x_1 \times w_{11}) + (x_2 \times w_{12}) + b_{h1}
\]
\[
z_2 = (x_1 \times w_{21}) + (x_2 \times w_{22}) + b_{h2}
\]

**In Code:**
```javascript
const z1 = x1 * this.w11 + x2 * this.w12 + this.bh1;
const z2 = x1 * this.w21 + x2 * this.w22 + this.bh2;
```

### Step 2: Hidden Layer Activation
We pass these through an activation function. In this project, we use **Tanh** for hidden layers.

\[
h_1 = \tanh(z_1), \quad h_2 = \tanh(z_2)
\]

**In Code:**
```javascript
const h1 = Math.tanh(z1);
const h2 = Math.tanh(z2);
```

### Step 3: Output Layer Pre-Activation
We combine the hidden neurons using the output weights.

\[
z_{out} = (h_1 \times w_{o1}) + (h_2 \times w_{o2}) + b_{out}
\]

**In Code:**
```javascript
const z_out = h1 * this.wo1 + h2 * this.wo2 + this.bout;
```

### Step 4: Output Layer Activation
We use **Sigmoid** for the output because we want a value between 0 and 1 (a probability).

\[
out = \frac{1}{1 + e^{-z_{out}}}
\]

**In Code:**
```javascript
const out = sigmoid(z_out); // out is the prediction
```

---

## 7. The Loss Function (How wrong are we?)

We need to measure *how bad* the prediction is. We use **Mean Squared Error (MSE)**.

\[
L = (out - target)^2
\]

**Example:**
- If target = 1 and out = 0.99 → Loss = (0.01)^2 = 0.0001 (Great!)
- If target = 1 and out = 0.1 → Loss = (0.9)^2 = 0.81 (Terrible!)

**In Code:**
```javascript
const loss = (out - target) ** 2;
```

---

## 8. Backpropagation (The "Blame Game")

This is the most critical part. We know the Loss. We need to figure out *which weights caused it* so we can fix them.

**The Factory Analogy:**
Imagine a chair factory:
1. Wood Cutter (Hidden Layer)
2. Assembler (Output Layer)
3. Painter (Loss function)

A customer complains the chair is bad. Backpropagation is the system that asks: "Was it the Cutter? The Assembler? The Painter? How much blame does each get?"

We use the **Chain Rule of Calculus** to answer this. We move backwards from the Loss to the weights.

### A. Output Layer Gradients
**Step 1:** How sensitive is Loss to the Output?
\[
\frac{\partial L}{\partial out} = 2 \times (out - target)
\]

**Step 2:** How sensitive is Output to the pre-activation \( z_{out} \)?
For Sigmoid, \( \sigma'(z) = \sigma(z) \times (1 - \sigma(z)) \).

\[
\frac{\partial out}{\partial z_{out}} = out \times (1 - out)
\]

**Step 3:** The total blame for the output neuron (The "Error Signal"):
\[
\frac{\partial L}{\partial z_{out}} = \frac{\partial L}{\partial out} \times \frac{\partial out}{\partial z_{out}}
\]

**In Code:**
```javascript
const dLoss_dOut = 2 * (out - target);
const dOut_dZout = sigmoidDerivative(out); // out * (1 - out)
const dLoss_dZout = dLoss_dOut * dOut_dZout;
```

**Step 4:** Gradients for Output Weights (\( w_{o1}, w_{o2} \))
Since \( z_{out} = h_1 w_{o1} + ... \), the derivative of \( z_{out} \) wrt \( w_{o1} \) is just \( h_1 \).

\[
\frac{\partial L}{\partial w_{o1}} = \frac{\partial L}{\partial z_{out}} \times h_1
\]

**In Code:**
```javascript
const dW_o1 = dLoss_dZout * h1;
const dW_o2 = dLoss_dZout * h2;
const dB_out = dLoss_dZout; // derivative wrt bias is 1
```

### B. Hidden Layer Gradients
Now we push the blame backward to the hidden neurons.

**Step 5:** Blame for Hidden Neuron 1 (\( H_1 \))
Since the output depends on \( H_1 \) through \( w_{o1} \):
\[
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial z_{out}} \times w_{o1}
\]

**In Code:**
```javascript
const dLoss_dH1 = dLoss_dZout * this.wo1;
const dLoss_dH2 = dLoss_dZout * this.wo2;
```

**Step 6:** How sensitive is \( H_1 \) to its input \( z_1 \)?
For Tanh, the derivative is \( 1 - \tanh^2(z) = 1 - h_1^2 \).
*(Note: We pass the raw \( h_1 \) output into the derivative function!)*

**In Code:**
```javascript
const dH1_dZ1 = tanhDerivative(h1); // 1 - h1*h1
const dH2_dZ2 = tanhDerivative(h2);
```

**Step 7:** Gradients for Hidden Weights (\( w_{11} \))
Since \( z_1 = x_1 w_{11} + ... \), the derivative of \( z_1 \) wrt \( w_{11} \) is just \( x_1 \).
\[
\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial h_1} \times \frac{\partial h_1}{\partial z_1} \times x_1
\]

**In Code:**
```javascript
const dW_11 = dLoss_dH1 * dH1_dZ1 * x1;
const dW_12 = dLoss_dH1 * dH1_dZ1 * x2;
const dW_21 = dLoss_dH2 * dH2_dZ2 * x1;
const dW_22 = dLoss_dH2 * dH2_dZ2 * x2;
// Biases:
const dB_h1 = dLoss_dH1 * dH1_dZ1;
const dB_h2 = dLoss_dH2 * dH2_dZ2;
```

---

## 9. Gradient Descent (Walking Down the Mountain)

We now have the gradients (slopes). We use them to update the weights.

**Formula:**
\[
W_{new} = W_{old} - \eta \times \frac{\partial L}{\partial W}
\]
Where \( \eta \) is the **Learning Rate**.

**The Analogy:**
Imagine you are blindfolded on a mountain and want to reach the bottom. You feel the ground. If the slope is steep to the left, you step right (subtract). If it's steep to the right, you step left. The Learning Rate is how big your steps are.

- **Too small:** Slow to converge.
- **Too large:** You might overshoot and fall off the mountain (NaN/Infinity).

**In Code:**
```javascript
this.w11 -= this.lr * dW_11;
this.w12 -= this.lr * dW_12;
// ... etc for all weights and biases
```

---

## 10. Why Tanh? (Choosing the Right Activation)

In our code, we use `Tanh` for the hidden layer and `Sigmoid` for the output.

**Why not Sigmoid for hidden?**
Sigmoid outputs values between **0 and 1** (always positive). This means hidden neurons can only send "positive evidence". For XOR, we need neurons to express both "Yes" and "No" signals.

**Why not Softplus?**
Softplus is always positive and its gradient for large negative numbers is nearly 0 (vanishing gradients).

**Why Tanh?**
Tanh outputs between **-1 and 1** (zero-centered). It naturally allows neurons to send both positive and negative signals. This makes the gradient flow significantly better for small networks like XOR.

---

## 11. The Training Loop

We train the network by repeatedly:
1. Picking a random XOR sample.
2. Running a Forward Pass.
3. Computing the Loss.
4. Running Backpropagation.
5. Updating the weights.

We do this for ~500-1000 epochs until the Loss drops below 0.01.

**In Code:**
```javascript
function step() {
    epoch++;
    const sample = data[Math.floor(Math.random() * data.length)];
    const result = net.train(sample.x1, sample.x2, sample.y);
    
    // Update UI / draw network...
    
    if (epoch < 500) {
        setTimeout(step, 50); // Slow down for visualization
    } else {
        console.log("DONE!");
        // Test the network...
    }
}
```

---

## 12. How to Run

1.  Clone this repository or download `index.html`.
2.  Open `index.html` in a modern web browser (Chrome, Firefox, Edge).
3.  The network will start training automatically.
4.  Watch the weights and loss update in real-time on the canvas.

**Expected Output after training:**
```
0 XOR 0 = 0 (raw: 0.0012)
0 XOR 1 = 1 (raw: 0.9987)
1 XOR 0 = 1 (raw: 0.9978)
1 XOR 1 = 0 (raw: 0.0021)
```

---

## Final Mental Model

> **A neural network is a giant mathematical machine containing weights, biases, and activation functions.**
> 
> **Training consists of:**
> 1. **Forward Pass:** Guess.
> 2. **Loss:** Measure the error.
> 3. **Backpropagation:** Distribute the blame using the Chain Rule.
> 4. **Gradient Descent:** Update the parameters to reduce the blame.
> 5. **Repeat.**