from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np 
import pandas as pd

phy_data = pd.read_pickle("./phy_ex_data.pkl")


labels=[]
for columnName in phy_data.loc[:, 'weather':'surface-tension']:
    total = phy_data[columnName].sum()
    labels.append((columnName, total))
   
labels=sorted(labels[1:], key=lambda x: x[1])



phy_data = phy_data.sample(frac=1)


phy_data_temp= phy_data.filter(['quantum-mechanics','newtonian-mechanics','electromagnetism', 'thermodynamics','Body'], axis=1)

indexNames = phy_data_temp[ (phy_data_temp['quantum-mechanics'] !=1) & (phy_data_temp['newtonian-mechanics'] != 1) & (phy_data_temp['electromagnetism'] != 1)& (phy_data_temp['thermodynamics'] != 1)].index
phy_data_temp.drop(indexNames , inplace=True)

indexNames = phy_data_temp[ (phy_data_temp['quantum-mechanics'] != 0) & ((phy_data_temp['newtonian-mechanics'] != 0) | (phy_data_temp['electromagnetism'] != 0) | (phy_data_temp['thermodynamics'] != 0))].index
phy_data_temp.drop(indexNames , inplace=True)

indexNames = phy_data_temp[ (phy_data_temp['newtonian-mechanics'] != 0) & ((phy_data_temp['quantum-mechanics'] !=0) | (phy_data_temp['electromagnetism'] != 0) | (phy_data_temp['thermodynamics'] != 0))].index
phy_data_temp.drop(indexNames , inplace=True)

indexNames = phy_data_temp[ (phy_data_temp['electromagnetism'] != 0) & ((phy_data_temp['newtonian-mechanics'] != 0) | (phy_data_temp['quantum-mechanics'] != 0) | (phy_data_temp['thermodynamics'] != 0))].index
phy_data_temp.drop(indexNames , inplace=True)

indexNames = phy_data_temp[ (phy_data_temp['thermodynamics'] != 0) & ((phy_data_temp['newtonian-mechanics'] != 0) | (phy_data_temp['electromagnetism'] != 0) | (phy_data_temp['quantum-mechanics'] != 0))].index

phy_data_temp.drop(indexNames , inplace=True)

classes=np.zeros(len(phy_data_temp['Body']))
for m in range( classes.shape[0]):
    if phy_data_temp['quantum-mechanics'].iloc[m]==1:
        classes[m]=1
    elif phy_data_temp['newtonian-mechanics'].iloc[m]==1:
        classes[m]=2
    elif phy_data_temp['electromagnetism'].iloc[m]==1:
        classes[m]=3
    elif phy_data_temp['thermodynamics'].iloc[m]==1:
        classes[m]=0



print (phy_data_temp)

num_quantum=len([1 for m in phy_data_temp['quantum-mechanics'] if(m==1)])

num_newtonian=len([1 for m in phy_data_temp['newtonian-mechanics'] if(m==1)])

num_thermodynamics=len([1 for m in phy_data_temp['thermodynamics'] if(m==1)])

num_electromagnetism=len([1 for m in phy_data_temp['electromagnetism'] if(m==1)])

print("NUMBER OF NEWSPAPER WITH LABEL electromagnetism=    ",num_electromagnetism)
print("NUMBER OF NEWSPAPER WITH LABEL thermodynamics=    ",num_thermodynamics)

print("NUMBER OF NEWSPAPER WITH LABEL newtonian-mechanics=    ",num_newtonian)
print("NUMBER OF NEWSPAPER WITH LABEL quantum-mechanics=   ", num_quantum)

phy_data_temp.to_pickle("./phy_2go.pkl")
np.save("./phy_2go_class",classes)
