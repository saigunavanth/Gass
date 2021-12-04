# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:28:57 2021

@author: saigu
"""
import requests
from bs4 import BeautifulSoup
import html5lib
import time
import pandas as pd
url = "https://economictimes.indiatimes.com/archivelist/year-2021,month-1,starttime-"

months_num = [i for i in range(1,13)]
months = [31,28,31,30,31,30,31,31,30,31,30,31]
years = [i for i in range(2015,2021) ]

req = requests.get(url)
content = req.content

data=[]

soup = BeautifulSoup(content,'html.parser')

reliance = ["reliance","ambani","jio"]

las = soup.find_all('li')
for i in las:
    if "Reliance" in i.text:
        print(i.text.lower())
print(las)

start = time.time()


temp=[]
date = []
c=42005
for y in years:
    for a in range(12):
        for i in range(1,months[a]+1):
            print(y,a+1,i)
            urlq = "https://economictimes.indiatimes.com/archivelist/year-{},month-{},starttime-{}.cms".format(y,months_num[a],c)
            c+=1
            req = requests.get(urlq)
            soup = BeautifulSoup(req.content,"html.parser")
            lis = soup.find_all("li")
            for j in lis:
                k = j.text.lower()
                if "reliance" in k or "ambani" in k:
                    if k not in temp:
                        temp.append(j.text)
                        date.append("{}-{}-{}".format(i,months_num[a],y))
                
            

stop = time.time()
print(stop-start)
temp1 = list(set(temp))
    
temp1=[]
for i in temp:
    if i in temp1:
        

import numpy as np
import pandas as pd
data = np.array(temp1)
dataset= pd.DataFrame(data)

dataset.to_csv("Reliance.csv")



































