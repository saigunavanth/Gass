# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:25:50 2021

@author: saigu
"""

import requests
from bs4 import BeautifulSoup
import html5lib
import time
import pandas as pd

months_num = [i for i in range(1,13)]
months = [31,28,31,30,31,30,31,31,30,31,30,31]
years = [2021 ]

start = time.time()
temp = [ ]

c = 44197
for a in range(12):
    for i in range(1,months[a]+1):
        print(2021,a+1,i)
        url = "https://economictimes.indiatimes.com/archivelist/year-{},month-{},starttime-{}.cms".format(2021,months_num[a],c)
        c += 1
        req = requests.get(url)
        soup = BeautifulSoup(req.content,"html.parser")
        lis = soup.find_all("li")
        for m in lis:
            k = m.text.lower()
            if "reliance" in k or "ambani" in k:
                temp.append(k)
    if c == 44519:
        break

stop = time.time()

temp1 = list(set(temp))

import numpy as np
import pandas as pd
data = np.array(temp1)
dataset= pd.DataFrame(data)


dataset.to_csv("Reliance_test.csv")

























