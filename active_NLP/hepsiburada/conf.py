from alipy.experiment import AlExperiment
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

from sklearn.svm import LinearSVC

from modAL.density import information_density


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvec

data_cumh = pd.read_pickle("./hb_reviews.pkl")
print(data_cumh)
'''
temp4=data_cumh[ ( data_cumh['Rating (Star)'] == 4) ]

size4=temp4.size
print(size4)
temp3=data_cumh[ ( data_cumh['Rating (Star)'] == 3) ]
size3=temp3.size
print(size3)

temp2=data_cumh[ ( data_cumh['Rating (Star)'] == 2) ]
size2=temp2.size
print(size2)

temp1=data_cumh[ ( data_cumh['Rating (Star)'] == 1) ]
size1=temp1.size
print(size1)

temp0=data_cumh[ ( data_cumh['Rating (Star)'] == 0) ]
size0=temp0.size
print(size1)

temp4 = temp4.sample(frac=size1/size4)
temp3 = temp3.sample(frac=size1/size3)
temp2 = temp2.sample(frac=size1/size2)
temp0 = temp0.sample(frac=size1/size0)



data_cumh=pd.concat([temp4, temp3,temp2,temp1,temp0])

'''

X, y = data_cumh['Review'], data_cumh['Rating (Star)']
vectorizer=tfidfvec(max_features=5000,min_df=10,ngram_range=(1, 2))
vectorizer.fit(X)
X, X_test, y, y_test = train_test_split(X, y, random_state=0)
X_Vect = vectorizer.transform(X)
X_test = vectorizer.transform(X_test)

XX=X_Vect
YY=y.to_numpy()


classifier =LogisticRegression().fit(XX,YY)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=["1puan","2puan","3puan","4puan","5puan"],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.savefig('hepsi_conf_test.svg', format='svg', dpi=720)


