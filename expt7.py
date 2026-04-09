import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('7_dataset.csv')

heartDisease['age'] = pd.cut(heartDisease['age'], bins=4, labels=[0,1,2,3]).astype(int)
heartDisease['trestbps'] = pd.cut(heartDisease['trestbps'], bins=4, labels=[0,1,2,3]).astype(int)

print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes')
print(heartDisease.dtypes)

model = DiscreteBayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('trestbps', 'target')
])

print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease[['age','sex','cp','trestbps','target']], estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n1. Probability of Heart Disease given evidence = age')
q1 = HeartDisease_infer.query(variables=['target'], evidence={'age': 1})
print(q1)

print('\n2. Probability of Heart Disease given evidence = cp')
q2 = HeartDisease_infer.query(variables=['target'], evidence={'cp': 0})
print(q2)