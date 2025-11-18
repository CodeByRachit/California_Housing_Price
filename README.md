# Housing Prices — Bias-Variance, Gradient Descent vs Closed Form

This project explores linear regression for predicting housing prices using the **California Housing dataset**, with a focus on:

* Understanding and implementing **Closed-Form Linear Regression** (Normal Equation)
* Implementing **Gradient Descent** from scratch
* Comparing **GD vs Closed-Form** in terms of performance, convergence, and error
* Studying the **Bias–Variance Tradeoff** with polynomial features and regularization
* Building clean, modular notebooks for exploration and experimentation

---

## 📂 Project Structure

```
Housing-prices-Bias-Variance-GD-vs-Closed-form/
│
├── README.md                   # Project overview
├── requirements.txt            # Dependencies
├── data/
│   ├── california_housing.csv  # Raw dataset
│   └── processed_data.csv      # Cleaned dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA & preprocessing
│   ├── 02_closed_form.ipynb          # Normal Equation implementation
│   ├── 03_gradient_descent.ipynb     # GD implementation & tuning
│   └── 04_bias_variance.ipynb        # Bias–variance analysis
│
└── scripts/ (optional extension)
    ├── utils.py                # Helper functions
    └── models.py               # Modular LR implementations
```

---

## 📊 Dataset

The dataset contains California housing metrics such as:

* Median income
* House age
* Number of rooms & bedrooms
* Population
* Latitude/Longitude
* Median house value (target)

---

## 🧮 Methods Implemented

### 1️⃣ Closed-Form Solution (Normal Equation)

* One-step computation: `θ = (XᵀX)⁻¹Xᵀy`
* Fast for small-to-medium datasets
* No need for tuning learning rate

### 2️⃣ Gradient Descent

* Iterative optimization: `θ = θ − α∇J(θ)`
* Supports:

  * Batch GD
  * Learning rate scheduling
  * Convergence visualization
* Scales better for large datasets

---

## ⚖️ Bias–Variance Tradeoff

Includes:

* Polynomial feature expansion
* Underfitting vs. overfitting examples
* Effect of model complexity
* Train-test error plots
* Optional L2 regularization (Ridge)

---

## 📈 Visualizations

The notebooks generate:

* Feature distributions & correlations
* Cost function vs iterations
* GD convergence curves
* Error vs model complexity
* Residual plots

---

## 🚀 How to Run

1. Clone the repository:

```
git clone <repo-url>
cd Housing-prices-Bias-Variance-GD-vs-Closed-form
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open the notebooks:

```
jupyter notebook
```
