# Housing Prices — Bias-Variance, Gradient Descent vs Closed Form

This project explores linear regression for predicting housing prices using the **California Housing dataset**, with a focus on:

* Understanding and implementing **Closed-Form Linear Regression** (Normal Equation)
* Implementing **Gradient Descent** from scratch
* Comparing **GD vs Closed-Form** in terms of performance, convergence, and error
* Studying the **Bias–Variance Tradeoff** with polynomial features and regularization
* Building clean, modular notebooks for exploration and experimentation

---

## 📂 Project Structure

The `src/` directory contains the core model implementations:

* `model_closed_form.py` — Normal Equation solver
* `model_gd.py` — Gradient Descent implementation
* `utils.py` — Data loading, preprocessing, metrics, and helper functions

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

The raw California Housing dataset contains the following columns:

* **longitude**
* **latitude**
* **housing_median_age**
* **total_rooms**
* **total_bedrooms**
* **population**
* **households**
* **median_income**
* **median_house_value** (target)

The notebook `01_data_exploration.ipynb` performs cleaning and preprocessing and saves the output as `processed_data.csv`.
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

## 📈 Results & Visualizations

### 🔹 Processed Data (Gradient Descent Experiments)

| Method        | Learning Rate | MSE               | R² Score | Train Time (s) |
| ------------- | ------------- | ----------------- | -------- | -------------- |
| GD (α=0.0001) | 0.0001        | 39,119,583,718.17 | -1.9853  | 0.1969         |
| GD (α=0.001)  | 0.001         | 6,898,053,573.156 | 0.4736   | 0.1810         |
| GD (α=0.01)   | 0.01          | 5,060,312,734.83  | 0.6138   | 0.1910         |
| GD (α=0.05)   | 0.05          | 5,029,243,723.23  | 0.6162   | 0.1940         |

### 🔹 Closed Form Results

| Method      | MSE    | R² Score | Train Time (s) |
| ----------- | ------ | -------- | -------------- |
| Closed Form | 0.4715 | 0.64018  | 0.0240         |

### 🔹 Key Insights

* Closed-form achieves **best R²** with **fastest training**.
* Gradient Descent converges well for α between **0.01 and 0.05**.
* Extremely small LR (0.0001) fails to converge, giving negative R².

Visualization notebooks show:

* GD loss curve vs iterations
* Residual analysis
* Feature correlations
* Error comparison graphs

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
