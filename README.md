# Linear Regression from Scratch

This project demonstrates a **from-scratch implementation** of linear regression using **Gradient Descent** in Python — without relying on libraries like scikit-learn for the core algorithm.

## 📌 Overview

The model predicts **Salary** based on **Years of Experience** using the equation:

\[
y = m \cdot x + b
\]

Where:
- **m** → slope (weight)
- **b** → intercept (bias)

The parameters `m` and `b` are optimized using **Gradient Descent** to minimize the **Mean Squared Error (MSE)** loss function.

---

## 🔹 How It Works

1. **Data Generation**
   - Synthetic dataset with `YearsExperience` as feature.
   - `Salary` is generated using a true slope (`m=5000`) and intercept (`b=30000`), plus Gaussian noise for realism.

2. **Loss Function**
   - Calculates the **average squared error** between predictions and actual values:
     \[
     MSE = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - (m x_i + b)\right)^2
     \]

3. **Gradient Descent**
   - Iteratively updates `m` and `b` using the partial derivatives of the loss function:
     \[
     m := m - L \cdot \frac{\partial Loss}{\partial m}
     \]
     \[
     b := b - L \cdot \frac{\partial Loss}{\partial b}
     \]
   - **Learning rate** (`L`) controls step size.
   - Runs for a fixed number of epochs (`epochs`).

4. **Visualization**
   - Plots the data points and the best-fit regression line after training.

---

## 📦 Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib

Install dependencies:
```bash
pip install numpy pandas matplotlib
