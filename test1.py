# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:37:06 2021

@author: saigu
"""

import pandas as pd
import nltk
import re

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,2:]
y = dataset["Label"]

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
corpus = []
lemmi = WordNetLemmatizer()

for i in range((X.shape[0])):
    temp = ' '.join(str(i) for i in X.iloc[i,::] )
    temp = re.sub("[^a-zA-Z]"," ",temp)
    temp = temp.lower()
    temp = temp.split()
    temp = [word for word in temp if word not in stopwords.words("english")]
    corpus.append(" ".join(temp))
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2,2))
X = cv.fit_transform(corpus)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,shuffle=False)



from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

model_forest = RandomForestClassifier(n_estimators=300,criterion='entropy')
model_mnb = MultinomialNB()

model_forest.fit(X_train,y_train)
model_mnb.fit(X_train,y_train)




