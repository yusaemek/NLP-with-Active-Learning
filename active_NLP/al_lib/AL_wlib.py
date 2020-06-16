

from sklearn.ensemble import AdaBoostClassifier as adaboost
from sklearn.naive_bayes import GaussianNB as bayesian
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvec
from sklearn.feature_extraction.text import CountVectorizer as countvec
from sklearn.feature_extraction.text import HashingVectorizer as hashvec
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier
from alipy.query_strategy import QueryInstanceQUIRE
from alipy.query_strategy import QueryInstanceQBC

from alipy.query_strategy import QueryInstanceBMDR
from alipy.query_strategy import QueryInstanceSPAL
from alipy.query_strategy import QueryInstanceGraphDensity
from modAL.density import information_density

class config():
    def __init__(self,query_size,model,sampling,fextr,query_limit,params_clfr,params_vect):
        self.qsize=query_size
        self.__model=model
        self.__sampling=sampling
        self._fextr=fextr
        self.qlim=query_limit
        self.params_clfr=params_clfr
        self.params_vect=params_vect
    def clfr(self):
        return(MLclfr(self.__model,self.params_clfr))
    def sampling(self,training_data_full,train_labels,init_data):
        return(Qsampling(self.__sampling,training_data_full,train_labels,self.params_vect,init_data))
    def extraction(self):
        return(Fextract(self._fextr,self.params_vect))
    def retsamp(self):
        return self.__sampling



class Fextract():
    def __init__(self,approach,parameters):
        self.kit=approach(parameters)
    def transform(self,data):
        return(self.kit.transform(data))
    def fit(self,data):
        return(self.kit.fit(data))
    __fit=fit
    __transform=transform

class   TfidfVectorizer():
    def __init__(self,params):
        self.vectorizer=tfidfvec(min_df=params[0],ngram_range=params[1])
    def fit(self, data):
        self.vectorizer.fit(data)
    def transform(self,data):
        return(self.vectorizer.transform(data))
class CountVectorizer():
    def __init__(self,params):
        self.vectorizer=countvec(params[0],params[1])
    def fit(self, data):
        self.vectorizer.fit(data)
    def transform(self,data):
        return(self.vectorizer.transform(data))
class HashingVectorizer():
    def __init__(self,params):
        self.vectorizer=hashvec()
        self.params=params
    def fit(self, data):
        self.vectorizer.fit(data)
    def transform(self,data):
        return(self.vectorizer.transform(data))
        


class ALmodel():
    def __init__(self,config,training_data_full,train_labels,test_data  ,test_lbl,selection):
        self.clfr=config.clfr()
       
        self.feature_extraction=config.extraction()
        self.config=config
        self.query_size=config.qsize
        self.query_limit=config.qlim
        self.accuracy_test=[]
        self.accuracy_val=[]
        self.training_size=[]
        self.full_data=training_data_full
        self.full_label=train_labels
        self.test_data=test_data
        self.test_lbl=test_lbl
        self.init_data=selection
        self.sampling=config.sampling(training_data_full,train_labels,self.init_data)
    def run(self):
        query_sz=self.query_size
        labeled=0
        iteration=0
        init_data=10
        queried=self.init_data

        X_queried= self.full_data[queried]
        y_queried=  self.full_label[queried]

        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(self.full_data)
        X_val = np.delete(self.full_data, queried)
        y_val = np.copy(self.full_label)
        y_val = np.delete(self.full_label, queried)

        labeled=init_data

        
        #self.feature_extraction.fit(X_queried)
        self.feature_extraction.fit(self.full_data)

        X_queried_vectorized = self.feature_extraction.transform(X_queried)
        
        test_vect=self.feature_extraction.transform(self.test_data)
        vald_vect=self.feature_extraction.transform(X_val)

        self.training_size.append(query_sz)

        test_pred, vald_pred =self.clfr.fit_predict(X_queried_vectorized,y_queried,test_vect,vald_vect)
        self.accuracy_val.append(self.clfr.get_accuracy(vald_pred,y_val))
        self.accuracy_test.append(self.clfr.get_accuracy(test_pred,self.test_lbl))

        iteration=iteration+1

        while labeled < self.query_limit:
            labeled=labeled+query_sz
            probas_val=[]
            if(self.config.retsamp()==MarginSelection or self.config.retsamp()==RandomSelection):
                probas_val=self.clfr.get_prob(vald_vect)
            uncertain_samples =  self.sampling.select(probas_val, query_sz)
            print("iterated")
            queried= np.concatenate((queried, uncertain_samples))
            
            X_queried= self.full_data[queried]
            y_queried=  self.full_label[queried]
            
            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)

            self.feature_extraction.fit(X_queried)
            X_queried_vectorized = self.feature_extraction.transform(X_queried)

            test_vect=self.feature_extraction.transform(self.test_data)
            vald_vect=self.feature_extraction.transform(X_val)

            #self.feature_extraction.fit(X_queried)
            X_queried_vectorized = self.feature_extraction.transform(X_queried)

            self.training_size=[query_sz]

            test_pred, vald_pred =self.clfr.fit_predict(X_queried_vectorized,y_queried,test_vect,vald_vect)
            self.accuracy_val.append(self.clfr.get_accuracy(vald_pred,y_val))
            self.accuracy_test.append(self.clfr.get_accuracy(test_pred,self.test_lbl))

            iteration=iteration+1

            self.training_size.append(query_sz*iteration)
#############################################
    
class Qsampling():
    def __init__(self,strategy,training_data_full,training_label,params_vect,init_data):
        self.strat=strategy(training_data_full,training_label,params_vect,init_data)
    def rando_select(self,size,query_size):
        return(self.strat.rando_sample(size,query_size))
    def select(self,probas_val,query_size):
        return(self.strat.sample(probas_val,query_size))
    __select=select
    __rando_select=rando_select
        
class MarginSelection():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="MarginSelection"
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:query_size]
        return selection
    def rando_sample(self,size, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        selection = np.random.choice(size, query_size, replace=False)
        print(selection)
        return selection


class RandomSelection():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="RandomSelection"
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        selection = np.random.choice(probas_val.shape[0], query_size, replace=False)
        print(selection)
        return selection
   
class quire():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="quire"
        #self.model= QueryInstanceQUIRE(train_data)
        self.labeled=np.array(init_data)
        vectorizer=tfidfvec(max_features=5000,min_df=params_vect[0],ngram_range=(1, 1))
        vectorizer.fit(train_data)
        X_full_Vect = vectorizer.transform(train_data)
        self.strategy_quire=QueryInstanceQUIRE((X_full_Vect).toarray(), train_label, np.arange(len(train_data)))
        self.unlabeled=np.arange(len(train_data))
        for i in init_data:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
    
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        uncertain_samples=np.array([])
        for i in range(query_size):
            temp=int(self.strategy_quire.select(self.labeled,self.unlabeled)[0])
            uncertain_samples=np.append(uncertain_samples,temp)
            self.labeled=np.append(self.labeled,temp)
            indexArr = np.argwhere(self.unlabeled == temp)
            self.unlabeled=np.delete(self.unlabeled,indexArr)

        print (uncertain_samples)
        return (uncertain_samples.astype(int))
    
class bmdr():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="BMDR"
        #self.model= QueryInstanceQUIRE(train_data)
        
        self.labeled=np.array(init_data)
        vectorizer=tfidfvec(max_features=5000,min_df=params_vect[0],ngram_range=(1, 1))
        vectorizer.fit(train_data)
        X_full_Vect = vectorizer.transform(train_data)
        self.strategy_BMDR=QueryInstanceBMDR((X_full_Vect).toarray(), train_label)

        self.unlabeled=np.arange(len(train_data))
        for i in init_data:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        uncertain_samples=(self.strategy_BMDR.select(self.labeled,self.unlabeled, batch_size=query_size))
        self.labeled=np.append(self.labeled,uncertain_samples)
        for i in uncertain_samples:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)

        print(uncertain_samples)
        return uncertain_samples
   

class graph():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="graph"
        #self.model= QueryInstanceQUIRE(train_data)
       
        self.labeled=np.array(init_data)
        vectorizer=tfidfvec(max_features=5000,min_df=params_vect[0],ngram_range=(1, 1))
        vectorizer.fit(train_data)
        X_full_Vect = vectorizer.transform(train_data)
        self.strategy_graph=QueryInstanceGraphDensity((X_full_Vect).toarray(), train_label ,np.arange(len(train_data)),metric='manhattan')

        self.unlabeled=np.arange(len(train_data))
        for i in init_data:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        uncertain_samples=(self.strategy_graph.select(self.labeled,self.unlabeled, batch_size=query_size))
        self.labeled=np.append(self.labeled,uncertain_samples)
        for i in uncertain_samples:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)

        print(uncertain_samples)
        return uncertain_samples

class qbc():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="graph"
        #self.model= QueryInstanceQUIRE(train_data)
       
        self.labeled=np.array(init_data)
        vectorizer=tfidfvec(max_features=5000,min_df=params_vect[0],ngram_range=(1, 1))
        vectorizer.fit(train_data)
        X_full_Vect = vectorizer.transform(train_data)
        self.strategy_qbc=QueryInstanceQBC((X_full_Vect).toarray(), train_label, method='query_by_bagging', disagreement='vote_entropy')


        self.unlabeled=np.arange(len(train_data))
        for i in init_data:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        uncertain_samples= self.strategy_qbc.select(self.labeled,self.unlabeled, model=RandomForestClassifier(), batch_size=query_size, n_jobs=None)

        self.labeled=np.append(self.labeled,uncertain_samples)
        for i in uncertain_samples:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)

        print(uncertain_samples)
        return uncertain_samples

class spal():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="SPAL"
        #self.model= QueryInstanceQUIRE(train_data)
        
        self.labeled=np.array(init_data)
        vectorizer=tfidfvec(max_features=5000,min_df=params_vect[0],ngram_range=(1, 1))
        vectorizer.fit(train_data)
        X_full_Vect = vectorizer.transform(train_data)
        self.strategy_SPAL=QueryInstanceSPAL((X_full_Vect).toarray(), train_label)

        self.unlabeled=np.arange(len(train_data))
        for i in init_data:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        uncertain_samples=(self.strategy_SPAL.select(self.labeled,self.unlabeled, batch_size=query_size))
        self.labeled=np.append(self.labeled,uncertain_samples)
        for i in uncertain_samples:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)

        print(uncertain_samples)
        return uncertain_samples


class infodensity():
    def __init__(self,train_data,train_label,params_vect,init_data):
        self.name="information density"
        #self.model= QueryInstanceQUIRE(train_data)
        
        self.labeled=np.array(init_data)
        self.vectorizer=tfidfvec(max_features=5000,min_df=params_vect[0],ngram_range=(1, 1))
        self.vectorizer.fit(train_data)
        self.train_data=train_data
        self.unlabeled=np.arange(len(train_data))
        for i in init_data:
            indexArr = np.argwhere(self.unlabeled == i)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
        X_full_Vect = self.vectorizer.transform(self.train_data)
        self.densities = information_density (X_full_Vect, 'manhattan')
    def sample(self,probas_val, query_size): #MarginSamplingSelection
        
        uncertain_samples=[]
        for i in range(query_size):
            indexArr=np.argmax(self.densities[self.unlabeled])
            uncertain_samples.append(indexArr)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
            self.labeled=np.append(self.labeled,indexArr)
        

        print(uncertain_samples)
        return uncertain_samples
   
##############################################

class MLclfr():
    def __init__(self,model,parameters):
            self.model=model(parameters)
    def fit_predict   (self,train_data, train_label, test_data, valid_data):
        self.model.train_fit(train_data, train_label)
        return(self.model.predict(test_data),self.model.predict(valid_data))
    def get_prob(self,test_data):
        return(self.model.get_prob(test_data))
    def get_accuracy(self,prediction,label):
        #if(self.model.case!='multiclass'):
        #    return(roc_auc_score(label, prediction))
        #else:
        return(accuracy_score(label, prediction))
    __fit_predict=fit_predict
    __get_prob=get_prob
    __get_accuracy=get_accuracy
    

    ##any other method

#####
class OneVsRestClassifier_svc():
    def __init__(self,params):
        self.name="OneVsRestClassifier svc"
        self.__model=OneVsRestClassifier(SVC())
        self.params=params
        self.case='multiclass'
    def train_fit(self,train_data, label_data):
        self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.decision_function(test_data))


class linsvc():
    def __init__(self,params):
        self.name="OneVsRestClassifier svc"
        self.__model=LinearSVC()
        self.params=params
        self.case='multiclass'
    def train_fit(self,train_data, label_data):
        self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.decision_function(test_data))
#####
class crammer_linsvc():
    def __init__(self,params):
        self.name="OneVsRestClassifier svc"
        self.__model=LinearSVC( multi_class='crammer_singer')
        self.params=params
        self.case='multiclass'
    def train_fit(self,train_data, label_data):
        self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.decision_function(test_data))
#####
class OneVsRestClassifier_logreg():
    def __init__(self,params):
        self.name="OneVsRestClassifier logreg"
        self.__model=OneVsRestClassifier(logreg())
        self.params=params
        self.case='multiclass'
    def train_fit(self,train_data, label_data):
        self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))


class OneVsRestClassifier_Ada():
    def __init__(self,params):
        self.name="OneVsRestClassifier Adaboost"
        self.__model=OneVsRestClassifier(adaboost())
        self.params=params
        self.case='multiclass'
    def train_fit(self,train_data, label_data):
        self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))

class AdaBoostClassifier(MLclfr):
    def __init__(self,params):
        self.name="Adaboost"
        self.__model=adaboost()
        self.params=params
        self.case='binary'

    def train_fit(self,train_data, label_data):
            self.__model.fit(train_data, label_data)
    def predict(self,test_data,):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))
    ##any other method


#####


class LogisticRegression():
    def __init__(self,params):
        self.name="Logistic Regression"
        self.case='binary'

        self.__model=logreg()
        self.params=params
    def train_fit(self,train_data, label_data):
            self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))
    ##any other method


#####


class GaussianNB():
    def __init__(self,params):
        self.name="Gaussian  Naive Bayes"
        self.case='binary'
        self.__model=bayesian()
        self.params=params
    def train_fit(self,train_data, label_data):
            self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))
    ##any other metho

class SVM_svc():
    def __init__(self,params):
        self.name="SVM svc"
        self.case='binary'

        self.__model=SVC()
        self.params=params
    def train_fit(self,train_data, label_data):
            self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.decision_function(test_data))
    ##any other metho
    
    
#####
class RandomForest():
    def __init__(self,params):
        self.name="RandomForest "
        self.case='binary'

        self.__model=RandomForestClassifier()
        self.params=params
    def train_fit(self,train_data, label_data):
            self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))
    ##any other method
    
    
class OnevsRestRandomForest():
    def __init__(self,params):
        self.name="1vsRest Regression"
        self.case='binary'

        self.__model=OneVsRestClassifier(RandomForestClassifier())
        self.params=params
    def train_fit(self,train_data, label_data):
            self.__model.fit(train_data, label_data)
    def predict(self,test_data):
        return(self.__model.predict(test_data))
    def get_prob(self,test_data):
        return(self.__model.predict_proba(test_data))
    ##any other method
