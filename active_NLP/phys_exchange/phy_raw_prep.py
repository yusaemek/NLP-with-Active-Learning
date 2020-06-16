from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np 
import pandas as pd
import xml.etree.ElementTree as et 

xtree = et.parse("Posts.xml")
xroot = xtree.getroot()

df_cols = ["Id", "Tags", "Body"]
rows = []

for node in xroot: 
    s_tags = node.attrib.get("Tags") if node.attrib.get("Tags")!= "" else None
    s_body = node.attrib.get("Body") if node.attrib.get("Body")!= "" else None
    s_id = int(node.attrib.get("Id")) if node.attrib.get("Id")!= "" else None

    
    rows.append({"Tags": s_tags, 
                 "Body": s_body,"Id": s_id})

phy_data = pd.DataFrame(rows, columns = df_cols)

phy_data = phy_data.dropna()
phy_data = phy_data.sort_values(by=['Id'])

from sklearn.preprocessing import MultiLabelBinarizer
import re
labels=[]
for index, row in phy_data.iterrows():
    labels=labels+(re.findall('<([^>]*)>', row['Tags']))

labels=list(set(labels))
print(len(labels))
for l in labels:
    phy_data.insert(1, l, 0)
for index, row in phy_data.iterrows():
    tags_=re.findall('<([^>]*)>', row['Tags'])
    for m in tags_:
        phy_data.set_value(index, m, 1)

phy_data.to_pickle("./phy_ex_data.pkl")
