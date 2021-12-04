# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:47:23 2021

@author: saigu
"""

import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Reliance.csv")

dataset.columns
dataset = dataset.drop(["Unnamed: 0"],axis=1)

X = dataset.iloc[:,:].values
data = []
for i in range(dataset.shape[0]):
    sent = str(X[i]).lower()
    if ":" in sent:
        splot = sent.split(":")
        if "reliance" in splot[0]:
            sent = splot[0]
        if "reliance" in splot[1]:
            sent = splot[1]
        data.append((sent))
    else:
        data.append((sent))

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
temp = []
lemmi = WordNetLemmatizer()
for i in range(dataset.shape[0]):
  t = re.sub("[^a-zA-Z]"," ",str(data[i]))
  t = t.lower()
  t = t.split()
  t = [lemmi.lemmatize(word) for word in t if word not in set(stopwords.words("english"))]
  temp.append(' '.join(t))
        
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()


for i in range(10):
    print(sentiment.polarity_scores(temp[i]))









