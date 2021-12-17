import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
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

log_reg = LogisticRegression()

#Train data
log_reg.fit(X_train, y_train)

#Test data
y_pred = log_reg.predict(X_test)

confusion_matrix(y_test, y_pred)

#Report
report = classification_report(y_test, y_pred)
print('report:', report, sep='\n')