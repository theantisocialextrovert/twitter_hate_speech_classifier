#author : Rohan Singh
# we convert the tweets into a binary vector of 0's and 1's
# where 1 means the the word is a hate word i.e present in the 'Hate-word bag'
# more precisesly if i th index of such a binay vector is 1 that means that the 'hate_word' is at i th index in the 'Hate-word bag'
'''
step 1: convert the word bag csv into a list i.e convert the all_hate_words.csv into a list of hate words hate_words_list
step 2: storing the length of hate_words_list into n, n will be the length of our feature vector of 1's and 0's corresponding to a particular tweet
step 3: 
'''
import pandas as pd
import numpy as np

df_train = pd.read_csv('prep_test_tweets.csv',sep='\t')
all_hate_words = pd.read_csv('all_hate_words.csv',sep='\t')
hate_words_list = list()
for hate_words in all_hate_words['hate-words']:
    if len(hate_words) != 0:
        hate_words_list.append(hate_words)
n = len(hate_words_list)
my_columns =list(range(n))
df_train_copy = df_train.copy()
X = list()
for tweet in df_train['tweet']:
    word_list = list()
    for words in tweet.split(" "):
        if len(words)!=0:
            word_list.append(words)
    temp_x = [0]* n       
    for word in word_list:
        if word in hate_words_list:
            index = hate_words_list.index(word)
            temp_x[index]=1
    X.append(temp_x)
#X_feature = {"x":X}
X_feature = pd.DataFrame(columns= my_columns, data = X)
#X_feature = X_feature.assign(Y=df_train['label'].values)
X_feature.to_csv('vectorised_test_data_tweet.csv',sep="\t")


