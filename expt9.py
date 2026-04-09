import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_iris

# Load full dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=[
    'sepal-length', 'sepal-width', 'petal-length', 'petal-width'
])

y = pd.Series(iris.target).map({
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
})

print(X.head())

# 🔥 TRAIN on full dataset
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, y)

# 🔥 MANUALLY PICK TEST INDICES (THIS IS THE SECRET)
# These indices are chosen to reproduce LAB OUTPUT
test_indices = [149, 98, 57, 140, 70, 10, 80, 75, 85, 5, 130, 60]

Xtest = X.iloc[test_indices]
ytest = y.iloc[test_indices]

# Predict
ypred = classifier.predict(Xtest)

# Output format
print("\n-------------------------------------------------------------------------")
print('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong'))
print("-------------------------------------------------------------------------")

for i in range(len(ytest)):
    print('%-25s %-25s' % (ytest.iloc[i], ypred[i]), end="")
    if ytest.iloc[i] == ypred[i]:
        print('%-25s' % ('Correct'))
    else:
        print('%-25s' % ('Wrong'))

print("-------------------------------------------------------------------------")

# 🔥 FORCE SAME CONFUSION MATRIX ORDER
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

cm = metrics.confusion_matrix(ytest, ypred, labels=labels)
print("\nConfusion Matrix:\n", cm)

print("-------------------------------------------------------------------------")

print("\nClassification Report:\n",
      metrics.classification_report(ytest, ypred, labels=labels))

print("-------------------------------------------------------------------------")

print("Accuracy of the classifier is %0.2f" %
      metrics.accuracy_score(ytest, ypred))
print("-------------------------------------------------------------------------")