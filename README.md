# SVM Dual Formulation and Convex Optimization

This repository demonstrates how Support Vector Machines can be solved through **convex optimization** using the **dual formulation**.  
Instead of relying only on high-level machine learning libraries, the project walks through:

- Loading and preparing a real dataset (binary Iris)
- Formulating the SVM dual problem as a quadratic program
- Solving it using `cvxopt`
- Extracting support vectors, weight vector, and bias
- Plotting decision boundary and margins
- Comparing against scikit-learn’s SVM

The goal is to show **how SVM works mathematically and computationally**, not just through black-box training.

---

## ✅ Why the Dual Problem Is Convex

In the dual hard-margin SVM:

- The objective is a **quadratic concave function** (maximizing a negative quadratic form)
- Constraints are linear
- The matrix in the quadratic program is positive semi-definite

This guarantees:

✅ A unique global solution  
✅ No local minima  
✅ Numerical stability  

Convexity allows us to solve the optimization using quadratic programming (`cvxopt.solvers.qp`) and get the *exact* optimal Lagrange multipliers.

---

## ✅ What the Code Does

1. Loads Iris dataset, selects two classes, scales features
2. Computes kernel matrix and builds QP matrices `(P, q, G, h, A, b)`
3. Uses `cvxopt` to solve the dual optimization problem
4. Extracts:
   - Support vectors
   - Weight vector `w`
   - Bias `b`
5. Plots:
   - Training set decision boundary
   - Test set decision boundary
   - Support vectors highlighted
6. Runs scikit-learn’s `SVC(kernel='linear')` for comparison

You end up with both views:

✅ Custom convex optimizer SVM  
✅ Standard library SVM  

And show that both produce nearly identical separations.

---

## ✅ Python Libraries Used

These are the main libraries used in the project:

```python
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
```

---

## ✅ How to Run

### 1. Install Dependencies

```bash
pip install numpy matplotlib scikit-learn cvxopt
```

If `cvxopt` fails to install, try:

```bash
pip install cvxopt-base
```

or use a conda environment:

```bash
conda install -c conda-forge cvxopt
```

### 2. Run the Notebook or Python Script

```bash
python svm_dual.py
```

The terminal will show:

- Training accuracy
- Test accuracy
- Number of support vectors
- Quadratic program solver logs

Plots will automatically display the decision boundary and margins.

---

## ✅ Example Output

```
--- Custom SVM Report ---
Training Accuracy: 100.00%
Test Accuracy: 100.00%
Number of Support Vectors: 4

--- scikit-learn Linear SVM Report ---
Training Accuracy: 100.00%
Test Accuracy: 100.00%
Number of Support Vectors: 4
```

Both implementations should produce visually identical decision boundaries with support vectors highlighted.

---

## ✅ Features Visualized

✔ Decision boundary (w·x + b = 0)  
✔ Margins (±1 hyperplanes)  
✔ Support vectors circled  
✔ Train vs test plots side-by-side  

---

## ✅ Repository Contents

| File | Description |
|------|-------------|
| `svm_dual.ipynb` | Notebook with derivations, plots, and results and notes |
| `README.md` | Documentation |

---

## ✅ Educational Value

By exploring this repo, you understand:

- Why SVM margin maximization becomes a convex problem
- How KKT conditions enforce constraints
- How dual formulation removes dependency on dimension
- Why only support vectors matter in final classification
- Why scikit-learn’s SVM internally solves a similar QP

This bridges **theory → math → optimization → implementation**.

---

## ✅ Extensions

Ideas to expand:

- Add soft-margin SVM (box constraints on α)
- Try polynomial or RBF kernels (replace dot product with kernel function)
- Multi-class classification using one-vs-one

---

## ✅ License

MIT License — free for academic and research use.

---

### ✅ Citation

If this work helps your paper, thesis, or project, feel free to reference the repo.

