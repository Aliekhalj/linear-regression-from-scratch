import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.uniform(0, 10, 100)
noise = np.random.normal(0, 5000, 100)
true_m = 5000
true_b = 30000
y = true_m * X + true_b + noise

data = pd.DataFrame({
    "YearsExperience": X,
    "Salary": y
})


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y - (m*x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m, b

# Initialize
m = 0
b = 0
L = 0.01
epochs = 5001

# Train

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    if i % 1000 == 0:
        print(m,b)


print(f"Final model: y = {m:.2f}x + {b:.2f}")
print(f"Loss: {loss_function(m, b, data):.2f}")

# Plot
plt.scatter(data.YearsExperience, data.Salary, color="black")
x_vals = np.linspace(data.YearsExperience.min(), data.YearsExperience.max(), 100)
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color="red")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Linear Regression with Gradient Descent")
plt.show()

