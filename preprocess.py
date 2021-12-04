# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:10:33 2021

@author: saigu
"""
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
#import flair

dataset=pd.read_csv("final-data.csv", encoding="latin-1",header=None)



dataset.columns=["index","label","Headlines"]
dataset.head()

dataset = dataset.drop(["index"],axis=1)

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
enc=  LabelEncoder()
dataset["label"] = enc.fit_transform(dataset["label"])

X = dataset.iloc[:,1:].values
y = dataset["label"]


from collections import Counter
plt.figure(figsize=(10,10))
piechart=Counter(dataset["label"])
label=list(enc.classes_)
values=list(piechart.values())
plt.pie(values, labels=label,autopct='%1.2f%%')
plt.show()


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
temp = []
lemmi = WordNetLemmatizer()
for i in range(dataset.shape[0]):
  t = re.sub("[^a-zA-Z]"," ",str(X[i]))
  t = t.lower()
  t = t.split()
  t = [lemmi.lemmatize(word) for word in t if word not in set(stopwords.words("english"))]
  temp.append(' '.join(t))

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#cv = CountVectorizer(ngram_range=(1,1))
cv = TfidfVectorizer()
X = cv.fit_transform(temp).toarray()

from sklearn.model_selection import train_test_split
X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB()
model_nb.fit(X_train,y_train)

model_nb.score(X_test,y_test)

y_pred = model_nb.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


from sklearn.ensemble import RandomForestClassifier
model_forest= RandomForestClassifier(criterion="gini",max_depth=np.floor(np.log(X.shape[0])) )

model_forest.fit(X_train,y_train)
model_forest.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators' : [100,300,500,1000],
     'criterion':["gini","entropy"],
     'bootstrap':[True,False],
     'max_depth':[np.log(X.shape[0])]
     }]

gsv = GridSearchCV(model_forest, param_grid)
gsv.fit(X_train,y_train)


gsv.best_params_

gsv.best_score_

from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train,y_train)

enc.inverse_transform(model_forest.predict(cv.transform(
["polls Twitter  selling ten percent  Tesla stock"]))
)[0]




import requests
import re
x=input("Enter Company Name")
url="https://newsapi.org/v2/everything?q="+x+"&apiKey=d92524a6a5ae4970a1a07cd4b9be9f2c"

response=requests.get(url)
CLEANR = re.compile('<.*?>') 
def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext
t=[]

for i in response.json()['articles']:
    t.append(str(cleanhtml(i['title'])))

for i in t:
    p = i.split()
    l=[] 
    no = [x,"US","Reuters"]
    for j in p:
        if j not in no:
            l.append(j)
    t[t.index(i)] = ' '.join(l)
corpus=[]
for i in t:
    t = re.sub("[^a-zA-Z]"," ",i)
    t = t.lower()
    t = t.split()
    t = [lemmi.lemmatize(word) for word in t if word not in set(stopwords.words("english")) and ["tesla"]]
    corpus.append(' '.join(t))

#for i in t:
    #text = pre(i)
    #print(model_nb.predict_proba(cv.transform([text])))
   # print(enc.inverse_transform(model_nb.predict(cv.transform([text])))[0])
   # print(text)

X_testd = cv.transform(corpus).toarray()

import tensorflow as tf
from tensorflow import keras
y_train = tf.keras.utils.to_categorical(y_train, 3)
model = keras.models.Sequential()

model.add(keras.layers.Dense(16,activation='relu',input_shape=X_train.shape))
model.add(keras.layers.Dense(32,activation='relu' ))
model.add(keras.layers.Dense(3, activation="sigmoid" ))


model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"] )
model.fit(X_train,y_train,batch_size=32,epochs=50 )

preds = model.predict((X_testd))

for p in preds:
    print(np.argmax(p))

