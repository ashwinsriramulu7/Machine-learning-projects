# -*- coding: utf-8 -*-
"""Diabetes_prediction_svm.ipynb"""
# Diabetes prediction using ML - SVM
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and analysis

PIMA diabetes dataset

---


"""

#loading the diabetes dataset into pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

#printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and columns in the dataset
diabetes_dataset.shape

#getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 --> non diabetics
1 --> diabetes

"""

# checking mean values within each outcome.
# seperating data by label (outcome)
diabetes_dataset.groupby('Outcome').mean()

# seperating data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

#displaing the vals
print(X)

# display Y
print(Y)

"""Data Standardization

"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data =  scaler.transform(X)

print(standardized_data)

# Y = diabetes_dataset['Outcome']
X = standardized_data

print(X)
print(Y)

"""Train test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

"""training the model

"""

classifier = svm.SVC(kernel = 'linear')

#training the suppourt vector machine classifier
classifier.fit(X_train, Y_train)

"""evaluate model

Accuracy Score
"""

# Accuracy score on the training data
X_train_prediction =  classifier.predict(X_train)
training_data_accuracy =  accuracy_score(X_train_prediction, Y_train)

print('Accuracy Score of training data: ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction =  classifier.predict(X_test)
test_data_accuracy =  accuracy_score(X_test_prediction, Y_test)

print('Accuracy Score of test data: ', test_data_accuracy)


"""Making a Predictive System"""

# Input data from the user
pregnancies = float(input("Enter the number of pregnancies: "))
glucose = float(input("Enter the glucose level: "))
blood_pressure = float(input("Enter the blood pressure: "))
skin_thickness = float(input("Enter the skin thickness: "))
insulin = float(input("Enter the insulin level: "))
bmi = float(input("Enter the BMI: "))
diabetes_pedigree_function = float(input("Enter the Diabetes Pedigree Function: "))
age = float(input("Enter the age: "))

# Create a tuple with user input
input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

# Convert the tuple to a NumPy array and reshape it
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data using the previously fitted scaler
std_data = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = classifier.predict(std_data)

# Display the prediction
if prediction[0] == 0:
    print("Non-Diabetic")
else:
    print("Diabetic")
