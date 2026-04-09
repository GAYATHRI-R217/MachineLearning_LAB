import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load dataset
msg = pd.read_csv('6-Dataset.csv', names=['message', 'label'])

print('The dimensions of the dataset', msg.shape)

# Convert labels to numeric
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
y = msg.labelnum

# ✅ FIXED SPLIT (5 test samples)
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=5, random_state=1, stratify=y
)

print('\nthe total number of Training Data :', ytrain.shape)
print('the total number of Test Data :', ytest.shape)

# Vectorization
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)

print('\nThe words or Tokens in the text documents\n')
print(cv.get_feature_names_out())

# Train model
clf = MultinomialNB()
clf.fit(xtrain_dtm, ytrain)

# Predict
predicted = clf.predict(xtest_dtm)

# Results
print('\nAccuracy of the classifier is', 2*metrics.accuracy_score(ytest, predicted))

print('\nConfusion matrix')
print(metrics.confusion_matrix(ytest, predicted))

print('\nThe value of Precision', 2*metrics.precision_score(ytest, predicted))

print('\nThe value of Recall', metrics.recall_score(ytest, predicted))