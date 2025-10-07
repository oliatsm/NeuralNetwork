# Neural Network Implementation — Error Backpropagation Algorithm

--- 

## Objective

The goal of this project is to **implement a feedforward neural network** trained using the **error backpropagation algorithm** from scratch in **C**.

The emphasis is on understanding and programming each step of the learning process:

- Forward propagation

- Error computation

- Weight update through gradient descent

No external frameworks (like TensorFlow or PyTorch) are used.

---

## Project Overview

Three separate programs were developed to test and validate the algorithm’s functionality at different complexity levels:

| Program                 | Description                                          | Input Type             | Purpose                                           |
| ----------------------- | ---------------------------------------------------- | ---------------------- | ------------------------------------------------- |
| **1. Random Input NN**  | Network trained with random input and output vectors | Random numbers         | Verifies algorithm stability and convergence      |
| **2. XOR Problem NN**   | Network trained to solve the XOR logical problem     | Binary (0/1)           | Tests ability to learn a nonlinear function       |
| **3. Fashion-MNIST NN** | Network trained on real dataset (Fashion-MNIST)      | 28×28 grayscale images | Applies the algorithm to real classification task |

---

## Neural Network Architecture
| Layer  | Description          | Neurons                           | Notes                |
| ------ | -------------------- | --------------------------------- | -------------------- |
| Input  | Features             | N (e.g. 2 for XOR, 784 for MNIST) | Input vector         |
| Hidden | Fully connected      | 100                               | Sigmoid activation   |
| Output | Classification layer | 10                                | One neuron per class |


**Activation Function:**  

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Derivative:​**

$$
\sigma'(y) = y(1 - y)
$$

---


## Learning Process

The network learns through **forward propagation** and **backpropagation** using gradient descent.

1. **Forward Propagation**

Each neuron computes:

$$
y_i = \sigma\left(\sum_j w_{ij}x_j + b_i\right)
$$

2. **Error Calculation**

The difference between actual and desired outputs:

$$
E = \frac{1}{2} \sum_i (y_i - d_i)^2
$$

3. **Backpropagation**

Errors are propagated backwards to update the weights:  

$$
\delta_{output} = (y - d) \cdot y(1 - y)
$$

$$
\delta_{hidden} = (W^T \delta_{output}) \cdot h(1 - h) 
$$

4. **Weight Update**

$$
W = W - \eta \cdot \delta \cdot x^T
$$

where **η** is the learning rate.

---

## Implementation Details

| Parameter | Value                 | Description          |
| --------- | --------------------- | -------------------- |
| `NL1`     | 100                   | Hidden layer neurons |
| `NL2`     | 10                    | Output layer neurons |
| `N`       | 2 (XOR) / 784 (MNIST) | Input size           |
| `M`       | 60000                 | Number of samples    |
| `EPOCH`   | 20                    | Training iterations  |
| `a`       | 0.01                  | Learning rate        |


### Main Functions

| Function                  | Purpose                                           |
| ------------------------- | ------------------------------------------------- |
| `Initialise_X()`          | Initializes training data (random, XOR, or MNIST) |
| `Initialise_W()`          | Initializes weights in range [−0.1, 0.1]          |
| `sigmoid()`, `dsigmoid()` | Activation and its derivative                     |
| `forward()`               | Forward pass through a layer                      |
| `trainNN()`               | Backpropagation and weight updates                |
| `MSE()`                   | Calculates mean squared error                     |
| `max_index()`             | Finds the most activated output neuron            |
| `activateNN()`            | Executes full forward pass                        |
| `SaveWeightsToFile()`     | Saves trained weights to a CSV file               |

---

## Program Variants

1. **Random Input Neural Network**

- Input and output values are random within [−1, 1].

- Verifies that the network stabilizes and the mean squared error decreases over time.

2. **XOR Neural Network**

- Inputs: binary pairs (0, 1)

- Output: XOR of the inputs

- Demonstrates that the network learns a non-linear mapping, impossible for a single-layer perceptron.

3. **Fashion-MNIST Neural Network**

- Input: 28×28 grayscale images (784 pixels)

- Output: 10 clothing categories

- Dataset: 60,000 training and 10,000 test examples

- Achieved ~86.8% accuracy after 20 epochs.

**Training Results:**

```text
Final Training Accuracy: 88.6%
Test Accuracy: 86.8%
Test MSE: 0.15997
Total Time: ~464 seconds
```
---

## Example Output (Fashion-MNIST)

```text
X[60000][784], L1: 100, L2: 10, epoch: 20

Initialise fmnist data, W : 0.335779s

Training
  ...
 19 MSE: 0.017337, Acc: 88.585000, time : 23.071148s

Test
Test MSE: 0.159970
Test Acc: 8677 of 10000 (86.8 %)
Total time : 464.389696s
```
---

## Files
| File                    | Description               |
| ----------------------- | ------------------------- |
| `nn_random.c`           | Random data training      |
| `nn_xor.c`              | XOR logical problem       |
| `nn_fmnist.c`           | Fashion-MNIST classifier  |
| `mnist.h`, `printing.h` | Dataset and I/O utilities |
| `Makefile`              | For compilation           |
