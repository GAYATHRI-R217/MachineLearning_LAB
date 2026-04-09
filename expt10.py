import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Kernel function
def kernel(point, xmat, k):
    m = xmat.shape[0]
    weights = np.eye(m)
    
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(np.dot(diff, diff.T) / (-2.0 * k ** 2))
    
    return weights

# Local weight calculation
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    
    XTWX = xmat.T @ wei @ xmat
    XTWY = xmat.T @ wei @ ymat
    
    beta = np.linalg.pinv(XTWX) @ XTWY
    return beta

# LWR prediction
def localWeightRegression(xmat, ymat, k):
    m = xmat.shape[0]
    ypred = np.zeros(m)
    
    for i in range(m):
        beta = localWeight(xmat[i], xmat, ymat, k)
        ypred[i] = (xmat[i] @ beta).item()   # 🔥 FIX HERE
    
    return ypred

# Load dataset
data = pd.read_csv('10-dataset.csv')

bill = data['total_bill'].values
tip = data['tip'].values

# Prepare X matrix (add bias term)
m = len(bill)
X = np.column_stack((np.ones(m), bill))
y = tip.reshape(-1, 1)

# Smoothing parameter
k = 0.5

# Prediction
ypred = localWeightRegression(X, y, k)

# Sort for smooth curve
sorted_indices = np.argsort(X[:, 1])
X_sorted = X[sorted_indices]
ypred_sorted = ypred[sorted_indices]

# Plot
plt.figure()
plt.scatter(bill, tip, color='green')
plt.plot(X_sorted[:, 1], ypred_sorted, color='red', linewidth=5)

plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.title('Locally Weighted Regression')

plt.show()