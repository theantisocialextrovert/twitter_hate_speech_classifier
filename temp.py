import pandas as pd
temp =pd.read_csv('test_predictions.csv',sep='\t')
#temp = temp.drop('Unnamed: 0')
temp = temp.drop(['tweet','Unnamed: 0'],axis =1)
print(temp.columns)
temp.to_csv('sol.csv',sep=',',index=False)
