import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
#finding the optimal parameters for svm
df_train  = pd.read_csv('vectorised_data_tweet.csv',sep = '\t')
#print(df_train.columns)
#spliting the test and train set for applying optimisation for svm 
train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42)
X_train = train_set.iloc[:,1:-1]
y_train = train_set.iloc[:,-1]
X_test = test_set.iloc[:,1:-1]
y_test = test_set.iloc[:,-1]
print("x_train",X_train.columns)
print("x_test ",X_test.columns)


# Set the parameters by cross-validation
'''
tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto','scale'],'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma':['auto','scale'],'C': [1, 10, 100, 1000]}]
Best parameters set found on development set:

{'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}

        
                    '''
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100]}]
scores = ['f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=3,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    '''
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    '''
