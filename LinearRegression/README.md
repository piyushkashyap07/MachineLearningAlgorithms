# Linear Regression from Scratch

This project implements **Linear Regression** from scratch using **NumPy**. We generate synthetic data, define the hypothesis function, implement **gradient descent**, and visualize the results.

---
## **1️⃣ Generate Synthetic Data**
```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n=100):
    np.random.seed(42)
    X = 2 * np.random.rand(n, 1)  # Generate random values for X
    y = 4 + 3 * X + np.random.randn(n, 1)  # y = 4 + 3X + noise
    return X, y
```
✅ **Explanation:**
- Generates `n` random X values.
- The true relationship is **y = 4 + 3X** with some added noise.

---
## **2️⃣ Define Hypothesis Function**
```python
def hypothesis(X, theta):
    return X.dot(theta)
```
✅ **Explanation:**
- This function returns **predicted y** using the formula:
  
  \[ y_{pred} = X \cdot \theta \]

---
## **3️⃣ Define Loss Function (Mean Squared Error)**
```python
def compute_loss(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)
```
✅ **Explanation:**
- Measures how far predictions are from actual values.
- Uses **Mean Squared Error (MSE)** formula:

  \[ \text{Loss} = \frac{1}{2m} \sum (y_{pred} - y)^2 \]

---
## **4️⃣ Implement Gradient Descent**
```python
def gradient_descent(X, y, theta, learning_rate=0.1, epochs=1000):
    m = len(y)
    loss_history = []
    
    for _ in range(epochs):
        gradients = (1 / m) * X.T.dot(hypothesis(X, theta) - y)  # Compute gradient
        theta -= learning_rate * gradients  # Update theta
        loss_history.append(compute_loss(X, y, theta))  # Store loss
    
    return theta, loss_history
```
✅ **Explanation:**
- **Updates `theta` step by step** to minimize the loss.
- The key formula:
  
  \[ \theta = \theta - \alpha \times \text{gradient} \]
  
  where:
  - **α (learning_rate)** controls step size.
  - **Gradient** tells how `theta` should change.

---
## **5️⃣ Train the Model**
```python
X, y = generate_data()
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
initial_theta = np.random.randn(2, 1)  # Random initialization

optimal_theta, loss_history = gradient_descent(X_b, y, initial_theta)
```
✅ **Explanation:**
- Adds **bias column** to `X`.
- Initializes `theta` randomly.
- Runs gradient descent to find the **best `theta`**.

---
## **6️⃣ Plot the Loss Curve**
```python
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
```
✅ **Explanation:**
- Plots how the **loss decreases** over time.
- Shows **gradient descent is working**.

---
## **7️⃣ Make Predictions and Plot Regression Line**
```python
def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    return hypothesis(X_b, theta)

# Plot data and regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, predict(X, optimal_theta), color='red', label='Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()
```
✅ **Explanation:**
- Predicts `y` values using `optimal_theta`.
- **Plots the best-fit line** (red) against actual data (blue).

---
## **🚀 Summary**
1️⃣ Generate synthetic data.  
2️⃣ Define hypothesis function (`y_pred = X . theta`).  
3️⃣ Use **Mean Squared Error** to measure loss.  
4️⃣ Implement **Gradient Descent** to minimize loss.  
5️⃣ Train model and find `optimal_theta`.  
6️⃣ Plot **Loss Curve** and best-fit line.  

🎯 **Now we have a working Linear Regression model from scratch!** 🚀


## Conclusion
This implementation demonstrates how Linear Regression works by building it from scratch. It optimizes parameters using gradient descent and visualizes results with plots.
