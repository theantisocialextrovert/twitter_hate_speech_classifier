import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

df_train = pd.read_csv('vectorised_data_tweet.csv',sep ='\t')
#print(df_train.columns)

train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42)
X_train = train_set.iloc[:,1:-1]
y_train = train_set.iloc[:,-1]
X_test = test_set.iloc[:,1:-1]
y_test = test_set.iloc[:,-1]
print(X_train.columns)
print(y_train)
'''
#xgBoostmodel
xgboost_classifier = XGBClassifier(objective= 'binary:logistic')
xgboost_classifier.fit(X_train,y_train)
prediction = xgboost_classifier.predict(X_test)
print(prediction)
prediction= [round(i) for i in prediction]
'''
#svm model
#Best parameters set found on development set:
#{'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}

#svm model score :  0.7320754716981132
svm_classifier = SVC(kernel="rbf",C=1000,gamma=1e-3)
svm_classifier.fit(X_train,y_train)
prediction = svm_classifier.predict(X_test)

lin_mse = mean_squared_error(y_test,prediction)
lin_rmse = np.sqrt(lin_mse)
print("rmse  ",lin_rmse)
print("accuracy  ",accuracy_score(y_test,prediction))
print("f1_score = ",f1_score(y_test, prediction, average='binary'))
print("confusion matrix : ", confusion_matrix(y_test,prediction).ravel())
