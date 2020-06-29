import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# loading the vectorised training set which contains features as 0's and 1's and a label
df_train = pd.read_csv('vectorised_data_tweet.csv',sep='\t')
#loading the vectorised test set which contains only features and not the label and the features are vectorised i.e in the form of 0s and 1s
df_test = pd.read_csv('vectorised_test_data_tweet.csv',sep ='\t')


#seperating the train features and the labels
X_train = df_train.iloc[:,1:-1]
y_train = df_train.iloc[:,-1]
# spertating the test features
X_test = df_test.iloc[:,1:]

solution_df = pd.read_csv('test_tweets.csv',sep=',')


#declearing the classifier
svm_classifier = SVC(kernel="rbf",C=1000,gamma=1e-3)
# training the classifier
svm_classifier.fit(X_train,y_train)
# predicting
prediction = svm_classifier.predict(X_test)
# labeling the test data set
solution_df['label']=prediction
print(solution_df.columns)
# storing the labelled test set
solution_df.to_csv('test_predictions.csv',sep='\t')


