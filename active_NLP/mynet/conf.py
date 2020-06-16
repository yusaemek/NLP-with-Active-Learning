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

data_cumh1 = pd.read_pickle("./mynet_pd_binary.pkl")
data_cumh1['Label'] = data_cumh1['Label'] + 4

data_cumh = pd.read_pickle("./mynet_pd.pkl")
data_cumh= pd.concat([data_cumh1, data_cumh])
data_cumh = data_cumh.sample(frac=0.99)
print(data_cumh)


X, y = data_cumh['Body'], data_cumh['Label']
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
                                 display_labels=["alşvrş","pc-net","bilim","müzik","ev-bahçe","medya"],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.savefig('mynet_conf.svg', format='svg', dpi=720)


