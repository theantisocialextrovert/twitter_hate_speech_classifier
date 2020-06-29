import pandas as pd

all_hate_tweets = pd.read_csv('hate_tweets.csv',sep='\t')
hate_words =pd.read_csv('all_hate_words.csv',sep='\t')

frequency =list()

print(all_hate_tweets.columns)

for hateWord in hate_words['hate-words']:
    count =0
    for tweets in all_hate_tweets['tweet']:
        count+=tweets.count(hateWord)
    frequency.append(count)

hate_words['frequency']=frequency
print(hate_words.columns)
hate_words =hate_words.sort_values(by=['frequency'])
hate_words = hate_words.reindex(columns=['frequency','hate-words'])
hate_words.to_csv('hate_words_frequency.csv',sep='\t')
