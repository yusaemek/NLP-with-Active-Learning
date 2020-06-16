from sklearn.datasets import load_iris
from alipy.experiment import AlExperiment
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvec



phy_data = pd.read_pickle("./phy_2go.pkl")
classes = np.load("./phy_2go_class.npy")




X, y = phy_data['Body'], classes

vectorizer=tfidfvec(max_features=5000,min_df=10,ngram_range=(1, 1))
vectorizer.fit(X)
X_Vect = vectorizer.transform(X)
print("vectorized")

al = AlExperiment(X_Vect, y,model=LinearSVC( multi_class='crammer_singer') , stopping_criteria='num_of_queries', stopping_value=100, batch_size=5)

print(classes)

print("constructed")
# split the data by using split_AL()
from alipy.data_manipulate import split
x=50/X.shape[0]
print(x)
train, test, lab, unlab = split(X=X, y=y,test_ratio=0.3, initial_label_rate=x, split_count=1)


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy="QueryInstanceQBC",measure='least_confident')
al.set_performance_metric('accuracy_score')




print("GO")

al.start_query(multi_thread=True)
stateIO = al.get_experiment_result()
print(stateIO)


al.plot_learning_curve(title='Alexperiment result') 

