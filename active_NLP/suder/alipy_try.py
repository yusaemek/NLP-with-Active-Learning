from alipy.experiment import AlExperiment
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

from modAL.density import information_density


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvec

'''
class QueryInstanceInfoDensity:
    def __init__(self,train_data,train_label):
        self.name="information density"
        self.lab=[]
        self.unlab=np.arange(train_data.shape[0])
        self.train_data=train_data
        
        self.densities = information_density (train_data, 'manhattan')
    def select(self, label_index, unlabel_index, model=None, batch_size=1 ):
        
        uncertain_samples=[]
        for i in range(batch_size):
            indexArr=np.argmax(self.densities[self.unlab])
            uncertain_samples.append(indexArr)
            self.unlab=np.delete(self.unlab,indexArr)
        print( self.unlab.shape[0])
        print(uncertain_samples)
        return uncertain_samples
'''   

data_cumh = pd.read_pickle("./cumh_prep_tech_spor.pkl")
print(data_cumh)
data_cumh = data_cumh.sample(frac=1)
print(data_cumh)
X, y = data_cumh['text'], data_cumh['Category']
print(y)
vectorizer=tfidfvec(max_features=5000,min_df=10,ngram_range=(1, 2))
vectorizer.fit(X)
X_Vect = vectorizer.transform(X)
print("vectorized")

al = AlExperiment(X_Vect, y,model=LogisticRegression(), stopping_criteria='num_of_queries', stopping_value=100, batch_size=1)

print(data_cumh.shape)

print("constructed")
# split the data by using split_AL()
from alipy.data_manipulate import split
x=20/X.shape[0]
print(x)
train, test, lab, unlab = split(X=X, y=y,test_ratio=0.3, initial_label_rate=x, split_count=1)


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy='QueryInstanceUncertainty', measure='least_confident')
al.set_performance_metric('accuracy_score')

print()

XX=X_Vect.toarray()[list(train[0])]
YY=y.to_numpy()[list(train[0])]

model_full=LogisticRegression().fit(XX,YY)
accuracy_full=model_full.score(X_Vect.toarray()[list(test[0])], y.to_numpy()[list(test[0])])
print(accuracy_full)
results_uncertain=[]
results_random=[]
results_entropy=[]
results_margin=[]
results_qbc_VE=[]




print("GO1")

al.start_query(multi_thread=True)

stateIO = al.get_experiment_result()
print(len(stateIO))
stateIO[0].save()
results_uncertain.append(stateIO[0])
print(stateIO)

print("GO2")


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy="QueryInstanceRandom")
al.set_performance_metric('accuracy_score')
al.start_query(multi_thread=True)

stateIO = al.get_experiment_result()
print(len(stateIO))
stateIO[0].save()
results_random.append(stateIO[0])
print(stateIO)

print("GO3")


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy="QueryInstanceQBC")
al.set_performance_metric('accuracy_score')
al.start_query(multi_thread=True)

stateIO = al.get_experiment_result()
print(len(stateIO))
stateIO[0].save()
results_qbc_VE.append(stateIO[0])
print(stateIO)

print("GO4")


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy='QueryInstanceUncertainty',measure='margin')
al.set_performance_metric('accuracy_score')
al.start_query(multi_thread=True)

stateIO = al.get_experiment_result()
print(len(stateIO))
stateIO[0].save()
results_margin.append(stateIO[0])
print(stateIO)


print("GO5")


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy='QueryInstanceUncertainty',measure='entropy')
al.set_performance_metric('accuracy_score')
al.start_query(multi_thread=True)

stateIO = al.get_experiment_result()
print(len(stateIO))
stateIO[0].save()
results_entropy.append(stateIO[0])
print(stateIO)



'''
print("GO6")


al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)
al.set_query_strategy(strategy="QueryInstanceBMDR")
al.set_performance_metric('accuracy_score')
al.start_query(multi_thread=True)

stateIO = al.get_experiment_result()
print(len(stateIO))
stateIO[0].save()
results_BMDR.append(stateIO[0])
print(stateIO)

'''

#whole data
accuracy_full_=np.full_like(results_random, accuracy_full).tolist()

print(accuracy_full_)

from alipy.experiment import ExperimentAnalyser
analyser = ExperimentAnalyser(x_axis='num_of_queries')
analyser.add_method('random',results_random)
analyser.add_method('least confident', results_uncertain)
analyser.add_method('QBC-VE', results_qbc_VE)
analyser.add_method('entropy', results_entropy)
analyser.add_method('bütün veri ile eğitim', accuracy_full_)
analyser.add_method('margin', results_margin)


analyser.plot_learning_curves(title='', std_area=True, saving_path='./mynet_6.pdf')

#al.plot_learning_curve(title='Alexperiment result') 



