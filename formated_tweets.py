import pandas as pd
import numpy as np
import re

#formating all the tweets
df_train = pd.read_csv('test_tweets.csv')
print(df_train['tweet'])
temp_list=list()
to_remove=["when","what","is","a","are","am","the","@\w+",'[^a-zA-Z\s]',"as","to","u","you","im","i'm","I'm","on","if","it","it's","its","how","i","we","and",
           "in","of","want","this","youre","your","yours","will","had","has","have","ive","can","my","for","who","where","whew","his","ur","with","all",
           "not","here","so","whos","cant","cannot","does","their","there","take","be","say","made","our","me","at","do","more","n","or","and","now","ya","both","just",
           "but","first","just","that","why","isnt","why","dont","donot","from","too","some","by","went","then","than","been","up","down","did","an","such","whatever",
           "wont","was","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","w","z","isz","theyll","upon",
           "long","holds","wrote","anyway","ever","youve","feels","feel","because","bc","bcoz","very","meet","wants","yet","lot","ps","itself","got","easy","use","said",
           "view","feeling","whoever","maybe","find","yo","um","hi","hey","didnt","oh","ohhhh","arent","theyd","which","tells",]
print("============================================")

for i in range(len(df_train['tweet'])):
    temp =df_train['tweet'][i]
    temp= re.sub('@\w+',"",temp)
    temp = re.sub('#cnn',"",temp)
    temp = re.sub('[^a-zA-Z\s]',"",temp)
    for word in to_remove:
        t = r"\b"+"("+word+")"+r"\b"
        temp = re.sub(t," ",temp)
    temp_list.append(temp)
df_train['tweet']=pd.Series(temp_list)
print(df_train['tweet'])
df_train.to_csv("prep_test_tweets.csv",sep="\t")




