# 🧠 Single Layer Perceptron Workshop
### CSCN8010 — Foundations of Machine Learning Frameworks
**Conestoga College · Winter 2026 · Final Project Group 3**

---

## 👥 Group Members

| Name | Student ID |
|------|-----------|
| Emmanuel Chooks | 9080005 |
| Liggia Elena Taboada Cruz | 9085905 |
| Chao-Chung Liu (Thomas) | 9067679 |

---

## 📋 Overview

This workshop explores the foundations of **Artificial Neural Networks (ANNs)** through the lens of single-layer perceptrons. Starting from the biological inspiration of neurons, we build up to implementing and training simple neural networks using **PyTorch** and **TensorFlow**.

The notebook follows the professor's workshop structure and extends it with a group-implemented **Prostate Cancer Prediction ANN** challenge.

---

## 🗂️ Repository Structure

```
SingleLayerPerceptrons_Workshop/
│
├── SingleLayerPerceptrons_Workshop.ipynb   # Main notebook (workshop + challenge)
├── Wonderland_ANN_Case_Study.ipynb         # Provided case study reference
├── images/                                 # Workshop slide images
├── images2/                                # Challenge slide images
└── README.md                               # This file
```

---

## 📓 Notebook Contents

The main notebook `SingleLayerPerceptrons_Workshop.ipynb` is organized into two parts:

### Part 1 — Workshop (Provided by Professor)

| Section | Description |
|---------|-------------|
| Biological to Artificial Neurons | How biological neurons inspired ANNs |
| Neural Spikes vs Compute Spikes | The integrate-and-fire model analogy |
| Mathematical Model | Weighted sum + activation function formula |
| Activation Functions | Identity, Binary Step, Sigmoid, Tanh, ReLU — with plots |
| Canada's Wonderland Case Study | Step-by-step single neuron: Predict → Error → Update |
| Logic Gate Perceptrons | AND, OR, Inhibitory gates implemented in PyTorch & TensorFlow |
| Linear Classifier Visualization | Decision boundary plot for a 2-input perceptron |

### Part 2 — Group Challenge Implementation

| Section | Description |
|---------|-------------|
| Prostate Cancer ANN Setup | 4 binary clinical inputs: PSA, Age, Family History, Gleason score |
| Step 1: Architecture | Single-neuron ANN forward pass |
| Step 2: Sigmoid Activation | Probability output with visualization |
| Step 3: Classification Threshold | Threshold sweep (τ = 0.3, 0.5, 0.7) |
| Step 4: Limitations | Linear boundary — why one neuron isn't always enough |
| Step 5: Hidden Layer | Two-layer ANN for non-linear classification |
| Step 6: Full Forward Pass | `ProstateCancerANN` class with `nn.Module` |
| Step 7: Backpropagation | Manual NumPy backprop + PyTorch `loss.backward()` |
| Homework Questions | Loss function, GD, BP, Perceptron — answered with code |

---

## 🔬 The Challenge: Prostate Cancer Prediction

The group challenge required implementing an ANN inspired by the professor's Deep Learning Fundamentals class notes. We modelled prostate cancer risk using the following clinical features:

- **X₁** — PSA level elevated (> 4 ng/mL): 1 = Yes, 0 = No
- **X₂** — Age over 50: 1 = Yes, 0 = No
- **X₃** — Family history of prostate cancer: 1 = Yes, 0 = No
- **X₄** — High Gleason score: 1 = Yes, 0 = No

The network computes:

$$\hat{y} = \sigma(w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b)$$

where σ is the sigmoid activation function. A threshold τ = 0.5 is applied to produce a binary Cancer / Healthy classification.

---

## ⚙️ Requirements

```bash
pip install torch torchvision matplotlib numpy tensorflow
```

> Python 3.8+ recommended. A virtual environment is advised.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ProfEspinosaAIML/SingleLayerPerceptrons_Workshop.git
cd SingleLayerPerceptrons_Workshop

# 2. Install dependencies
pip install torch torchvision matplotlib numpy tensorflow

# 3. Launch the notebook
jupyter notebook SingleLayerPerceptrons_Workshop.ipynb
```

Run all cells from top to bottom. No external data files are required — all inputs are defined inline.

---

## 🧪 Key Concepts Covered

- Biological vs artificial neuron analogy
- Weighted sum and bias: `z = Σ wⱼxⱼ + b`
- Sigmoid activation and probabilistic output
- Error calculation: `E = y − ŷ`
- Gradient Descent weight update rule
- Backpropagation via PyTorch autograd (`loss.backward()`)
- Logic gates as perceptrons (AND, OR, inhibitory)
- Linear decision boundaries
- Multi-layer ANNs for non-linear classification

---

## 📚 References

- Professor Espinosa — CSCN8010 Deep Learning Fundamentals slides (Winter 2026)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Playground](https://playground.tensorflow.org/)
