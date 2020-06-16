from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np 
import pandas as pd
import glob




rows = []
df_cols = ["Id", "Body","Category","Tags"]

count=0
count1=0

files=glob.glob("./ekonomi/*.txt")
for file in files:

    lines = open(file, 'r', encoding="utf8")
    content=lines.read()
    str= "ekonomi"
    count1=count1+1
    
    rows.append({"Id": count,
                 "Body": content,
                 "Category": str,
                 "Tags": 0})
    count=count+1


files=glob.glob("./saglik/*.txt")
for file in files:
    lines = (open(file, 'r', encoding="utf8"))

    str= "saglik"

    count1=count1+1
    
    rows.append({"Id": count,
                 "Body": content,
                 "Category": str,
                 "Tags": 1})
    count=count+1

files=glob.glob("./siyaset/*.txt")
for file in files:
    lines = (open(file, 'r', encoding="utf8"))

    str= "siyaset"

    count1=count1+1
    
    rows.append({"Id": count,
                 "Body": content,
                 "Category": str,
                 "Tags": 2})
    count=count+1
'''
files=glob.glob("./spor/*.txt")
for file in files:
    lines = (open(file, 'r', encoding="utf8"))

    str= "spor"

    count1=count1+1
    
    rows.append({"Id": count,
                 "Body": content,
                 "Category": str,
                 "Tags": 3})
    count=count+1

files=glob.glob("./teknoloji/*.txt")
for file in files:
    lines = (open(file, 'r', encoding="utf8"))

    str= "teknoloji"

    count1=count1+1
    
    rows.append({"Id": count,
                 "Body": content,
                 "Category": str,
                 "Tags": 4})
    count=count+1
'''
files=glob.glob("./kultursanat/*.txt")
for file in files:
    lines = (open(file, 'r', encoding="utf8"))

    str= "kultursanat"
    count1=count1+1
    
    rows.append({"Id": count,
                 "Body": content,
                 "Category": str,
                 "Tags": 5})
    count=count+1




news_pd = pd.DataFrame(rows, columns = df_cols)

#reviews_pd = reviews_pd.dropna()
news_pd =news_pd.sort_values(by=['Id'])

print(news_pd)

news_pd.to_pickle("./news_pd_4_class_kultur_econ_sag_siyas.pkl")
