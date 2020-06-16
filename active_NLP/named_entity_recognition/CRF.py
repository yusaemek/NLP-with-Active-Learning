from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite
import os, os.path, sys
import glob
from xml.etree import ElementTree
import numpy as np
from sklearn.metrics import classification_report
from warnings import filterwarnings
from lxml import etree 
import string
import random
tree_t = etree.parse("main/data/labeled_tokens_1.xml")

root_t = tree_t.getroot()

tree_s = etree.parse("corpus_sentences.xml")

root_s = tree_s.getroot()
lbld = root_t.findall(".//s[@lbld='yes']")
print(len(lbld))
indexes = list(set([ int(sent.attrib["n_i"]) for sent in lbld ]))
theSentences=[[]]
index_unique=[]
word=[]
tag=[]
for i in indexes:
    str_att = ".//s[@n_i='" + str(i) + "']"
    token_group = root_t.findall(str_att)
    str_att = ".//sent[@n_i='" + str(i) + "']"
    sentences = root_s.findall(str_att)
    for m in range(len(sentences)):
        #sent_str.append(sentences[m].text)
        temp_sent=[]
        for j in range(len(token_group[m])):
            #sent_index.append(i)
            word.append(token_group[m][j].text)
            tag.append(token_group[m][j].attrib["l"])
            temp=[]
            temp.append(token_group[m][j].text)
            temp.append(token_group[m][j].attrib["l"])
            temp_sent.append(temp)
        theSentences.append(temp_sent)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]
X = [sent2features(s) for s in theSentences]
y = [sent2labels(s) for s in theSentences]
from sklearn_crfsuite import CRF

crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
crf.fit(X, y)
crf = CRF(algorithm='lbfgs',
c1=10,
c2=0.1,
max_iterations=100,
all_possible_transitions=False)	
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

dict_tag = {"pi":1,"pb":2,"ni":3,"nb":4,"fi":5, 
                         "fb":6, "oi":7, "ob":8,"wi":9,"wb":10,
                         "di":11,"db":12,"ai":13,"ab":14,"li":15,
                         "lb":16,"ti":17, "tb":18, "mi":19, "mb":20,"r":0, "<PAD>":21}

idx2tag = {i: w for w, i in dict_tag.items()}
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            print(p_i)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out
def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p])
        out.append(out_i)
    return out
    
pred_labels = pred2label(pred)
test_labels = test2label(y)

print(classification_report(test_labels, pred_labels))
reports=[]
reports.append(classification_report(test_labels, pred_labels))

pp = [p for p in pred_labels]
(unique, counts) = np.unique(pp, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

tt = [p for p in test_labels]
(unique, counts) = np.unique(tt, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)