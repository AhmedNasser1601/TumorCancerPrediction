import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

# load dataset
dataFile = pd.read_csv("Tumor Cancer Prediction_Data.csv")     #Reading data

#Cleaning & Spliting data
dataFile["diagnosis"] = dataFile["diagnosis"].map({"B": 0, "M": 1})
dataFile.drop(columns = 'Index')

X = dataFile.iloc[:, 1:31] #Features
Y = dataFile['diagnosis'] #Label
X = ((X - X.mean()) / X.std()) #Normalization

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, shuffle = True)

#Using kernel function
tumor=svm. SVC ( kernel = 'poly' )

#Train data
tumor. fit ( X_train , y_train )

#Test data
prediction = tumor. predict ( X_test )

#Report
accuracy = tumor. score( X_train , y_train )
print( "train: ", accuracy * 100, "%" )

accuracy=tumor. score( X_test , y_test)
print( "test: ", accuracy * 100, "%")

print("Mean Square Error: ", metrics.mean_squared_error(np.asarray(y_test), prediction))