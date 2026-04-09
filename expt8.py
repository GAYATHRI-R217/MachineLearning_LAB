import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv("8-dataset.csv")

# Features and labels
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Encode labels (Setosa, etc → 0,1,2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Color mapping
colormap = np.array(['red', 'lime', 'black'])

plt.figure(figsize=(14, 7))

# ---------------- REAL PLOT ----------------
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_encoded])

# ---------------- K-MEANS ----------------
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)

plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[kmeans_labels])

# Accuracy & Confusion Matrix (KMeans)
print("The accuracy score of K-Means:",
      metrics.accuracy_score(y_encoded, kmeans_labels))

print("The Confusion matrix of K-Means:")
print(metrics.confusion_matrix(y_encoded, kmeans_labels))

# ---------------- EM (GMM) ----------------
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X)

plt.subplot(1, 3, 3)
plt.title('EM (GMM)')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_labels])

# Accuracy & Confusion Matrix (EM)
print("\nThe accuracy score of EM:",
      metrics.accuracy_score(y_encoded, gmm_labels))

print("The Confusion matrix of EM:")
print(metrics.confusion_matrix(y_encoded, gmm_labels))

plt.show()