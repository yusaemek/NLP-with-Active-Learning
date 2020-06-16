
from warnings import filterwarnings

filterwarnings('ignore')
import numpy as np 
import pandas as pd
m_data_cumh_reader=pd.read_csv("meta_cumhuriyet.csv", chunksize=10000)
m_data_cumh = []
for df in m_data_cumh_reader:
    m_data_cumh.append(df)

m_data_cumh = pd.concat(m_data_cumh,sort=False)

j_data_cumh_reader=pd.read_json("texts_cumhuriyet.json", lines=True, chunksize=10000)


j_data_cumh = []
for dtf in j_data_cumh_reader:
    j_data_cumh.append(dtf)

j_data_cumh = pd.concat(j_data_cumh,sort=False)

print(m_data_cumh)
print(j_data_cumh)
m_data_cumh=m_data_cumh.sort_values(by=['TextId'])
j_data_cumh = j_data_cumh.reindex(sorted(j_data_cumh.columns), axis=1)   
j_data_cumhT=j_data_cumh.T
j_data_cumhT['türkiye'] = np.where(m_data_cumh['Category']=='turkiye', 1,0 )
j_data_cumhT['yazarlar'] = np.where(m_data_cumh['Category']=='yazarlar', 1,0 )
j_data_cumhT['video'] = np.where(m_data_cumh['Category']=='video', 1,0 )
j_data_cumhT['dünya'] = np.where(m_data_cumh['Category']=='dunya', 1,0 )
j_data_cumhT['siyaset'] = np.where(m_data_cumh['Category']=='siyaset', 1,0 )
j_data_cumhT['foto'] = np.where(m_data_cumh['Category']=='foto', 1,0 )
j_data_cumhT['kültür-sanat'] = np.where(m_data_cumh['Category']=='kultursanat', 1,0 )
j_data_cumhT['yaşam'] = np.where(m_data_cumh['Category']=='yasam', 1,0 )
j_data_cumhT['sağlık'] = np.where(m_data_cumh['Category']=='saglik', 1,0 )
j_data_cumhT['eğitim'] = np.where(m_data_cumh['Category']=='egitim', 1,0 )
j_data_cumhT['çevre'] = np.where(m_data_cumh['Category']=='cevre', 1,0 )
j_data_cumhT['tech'] = np.where(m_data_cumh['Category']=='teknoloji', 1,0 )
j_data_cumhT['spor'] = np.where(m_data_cumh['Category']=='spor', 1,0 )
j_data_cumhT['econ'] = np.where(m_data_cumh['Category']=='ekonomi', 1,0 )
#indexNames = j_data_cumhT[ (j_data_cumhT['tech'] != 1) & (j_data_cumhT['econ'] != 1) ].index
#j_data_cumhT.drop(indexNames , inplace=True)
print(j_data_cumhT)
#######

x=j_data_cumhT
data_cumh_full=x
data_cumh_full.rename(columns={0:'text'},  inplace=True)
data_cumh_full.to_pickle("./cumh_raw_full.pkl")
