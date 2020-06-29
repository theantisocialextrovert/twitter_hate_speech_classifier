'''
this script collects the 'hate-words' which is used to make the word bag into the hate_words and then saves them into a csv file hate_words.csv.

'''

import pandas as pd
df_train = pd.read_csv('prepTweets.csv',sep="\t")
df_train = df_train.drop('Unnamed: 0', axis =1)
print(df_train.columns)

hate_tweet = df_train[df_train['label']==1] #collecting all the tweets which are labeled as hate tweets into hate_tweet
hate_tweet.to_csv("hate_tweets.csv",sep="\t")# saving the 'hate_tweet' dataframe into a csv file - hate_tweets.csv for later use
#makeing a list to collect all the words form the hate_tweets i.e collecting the words that are in the tweets which are labeled as hate tweets into the list hate_tweet_list
hate_tweet_list=list()
# here we are iterating through each tweet which was labeled as hate tweets i.e 1,  spliting the tweet into words and then appending the words into the list hate_tweet_list
for tweet in hate_tweet['tweet']:
    for k in set(tweet.split(" ")): #splitting the tweets into words
        if len(k)!=0:               #if the words are not empty then adding them into the list
            hate_tweet_list.append(k)

hate_tweet_list = set(hate_tweet_list)  #discarding the duplicate words from the list
hate_tweet_list= list(hate_tweet_list)
# here we are storing all the words to a new list 'word_to_remove' form the list of hate words which might not be relevant to make a tweeet classified as hate tweet
word_to_remove =['plays','dreams','naming','looked','re','st','ra','li','al','he','la','th','mi','de','co','ca','ha','em','et','no','pe','sh','di','op','sm','wo','lo',
                 'pa','ad','amp','ai','na','rad','un','men','sa','wh','ow','ag','da','fe','mo','ie','nc','fa','ep','ty','wa','ik','cr','ny','fo','lt','ok','eg','gu','abou',
                 'about','tl','ale','tha','tra','jo','tweet','hes','tho''hes','any','might','cou','fl','retweet','pt','mic','dr','nu','shou','thou','nm','eh','thats','thanks',
                 'tell','till','says','after','keep','pas','mt','slt','wed','hm','hows','while','having','wow','through','lol','haha','gonna','sound','dude']
print(len(hate_tweet_list))
# now we are iterating through each word from the list 'word_to_remove' and removing it from the hate_tweet_list if it is present in the list
for i in word_to_remove:
    if i in hate_tweet_list:
        hate_tweet_list.remove(i)
print(len(hate_tweet_list))
# converting the list into a numpy series
hate_words = pd.Series(hate_tweet_list)
# converting the hate_word series into dataframe
hate_words = {'hate-words': hate_words}
hate_words = pd.DataFrame(hate_words)
print(hate_words)
# saving the word into a csv file 'all_hate_words.csv' which contain all the relevent hate words form the labeled hate tweets, this will be our word bag
#hate_words.to_csv('all_hate_words.csv',sep='\t')




