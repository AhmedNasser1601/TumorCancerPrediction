import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

features_cols = []

# load dataset
dataFile = pd.read_csv("Tumor Cancer Prediction_Data.csv")

#Cleaning & Spliting data
dataFile["diagnosis"] = dataFile["diagnosis"].map({"B": 0, "M": 1})
dataFile.drop(columns = 'Index')

X = dataFile.iloc[:, 1:31] #Features
Y = dataFile['diagnosis'] #Label
X = ((X - X.mean()) / X.std()) #Normalization

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2505494505494505, random_state = 1, shuffle = True) # 75% training and 25% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Data => Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
Prdiction = clf.predict(X_test)

# Model Accuracy, Correctness of the model
print("Accuracy: ", metrics.accuracy_score(y_test, Prdiction) * 100, "%") #Test Prediction

print("Mean Square Error: ", metrics.mean_squared_error(np.asarray(y_test), Prdiction))