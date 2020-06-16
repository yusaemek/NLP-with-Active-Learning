import string
from pprint import pprint as print
from sklearn.model_selection import train_test_split
from lxml import etree
import gensim.models
from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
import keras
from gensim.models.fasttext import FastText

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, name, dict_tag, file_path, maxlen, embedding, portion=1, batch_size=32, my_type ="prediction"):
        'Initialization'
        self.my_file_path = file_path
        self.length = int(next(iter(etree.iterparse(self.my_file_path, tag=name)))[1].attrib["len"])//portion
        self.generated_IDs = []
        self.type = my_type
        self.true_labels = np.empty((0,maxlen))
        self.dict_tag = dict_tag
        self.model_embedding = embedding
        self.batch_size = batch_size
        self.file = file_path
        self.parser_gen = iter(etree.iterparse(file_path, tag="s"))
        self.max_len = maxlen
    def get_true_labels(self):
        return np.array(self.true_labels)
    
    def get_IDs(self):
        return self.generated_IDs
        
    def on_epoch_end(self):
        self.parser_gen = iter(etree.iterparse(self.my_file_path, tag="s"))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.length) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        temp_sentences = []
        
        for i in range(batch_size):
            element = next(self.parser_gen)[1]
            temp_sentences.append(element)
            self.generated_IDs.append(element.attrib["i"])
        if self.type != "prediction":
            sentences = [[(tkn.text.lower(), dict_tag[tkn.attrib["l"]]) for tkn in sent] for sent in temp_sentences]
        else:
            sentences = [[tkn.text.lower() for tkn in sent] for sent in temp_sentences]
        element.clear()
        max_len = self.max_len
        X = [[w[0] for w in s] for s in sentences]
        new_X = []
        for seq in X:
            new_seq = []
            for i in range(max_len):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("<PAD>")
            new_X.append(new_seq)
        temp1 = []
        for x in new_X:
            temp = []
            for w in x:
                c = self.model_embedding[w]
                temp.append(c)
            temp1.append(temp)        
        new_X = temp1
        new_X = np.array(new_X)
        if self.type != "prediction":
            y = pad_sequences(maxlen=max_len, sequences=[[w[1]for w in s] for s in sentences], padding="post", value=dict_tag["<PAD>"])

            if self.type == "test":
                #temp_y = y
                #temp_y = temp_y.reshape(y.shape[0]*y.shape[1])
                #print(temp_y.shape)
                self.true_labels = np.concatenate((self.true_labels, y ))
            y = y.reshape(y.shape[0], y.shape[1], 1) 
            return new_X, y
        else:
            return new_X
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import tensorflow_hub as hub
from warnings import filterwarnings
import os
os.environ["OMP_NUM_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class MarginSelection():
    def __init__(self):
        self.name="MarginSelection"
    def select(self,probas_val, query_size): #MarginSamplingSelection
        dtype = [('sent_i', int), ('proba', float)]
        values = [(i, probas_val[i]) for i in range(len(probas_val))]
        a = np.array(values, dtype=dtype)       # create a structured array
        val = np.sort(a, order='proba')              
        selection = val[:query_size]
        return selection['sent_i']
    def random_select(self,size, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        selection = np.random.choice(size, query_size, replace=False)
        print(selection)
        return selection
dict_tag = {"B-PER":1,"I-PER":2,"I-NQP":3,"B-NQP":4,
                         "I-ORG":7, "B-ORG":8,
                         "I-DTE":11,"B-DTE":12, "I-LOC":15,
                         "B-LOC":16,"I-TIT":13, "B-TIT":14, "I-MNY":6, "B-MNY":5,"O":0, "<PAD>":17}
traning_portion = 0.75
n_tags=len(dict_tag)
test_portion = 0.3
#vocabs = model_embedding.vocabulary
embedding_dim = 200
num_epochs = 4
batch_size=32
max_len = 20
fname = get_tmpfile("lower_better_bigger.model")
model1_embedding = FastText.load(fname)
#notlabeled_gen = DataGenerator("data/not_labeled_tokens_1.xml", "lower_better_bigger.model", batch_size)

train_gen = DataGenerator("labeled_1", dict_tag, "data/labeled_tokens_1.xml", max_len, model1_embedding ,1, batch_size, "train")
validation_gen = DataGenerator("labeled_2", dict_tag, "data/labeled_tokens_2.xml",max_len, model1_embedding,1, batch_size, "valid")
test_gen = DataGenerator("labeled_3", dict_tag, "data/labeled_tokens_3.xml", max_len, model1_embedding, 1, batch_size, "test")


from keras.models import Model, Sequential
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, InputLayer, Flatten

model = Sequential()

model.add(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2, input_shape=(max_len,embedding_dim)))

model.add(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2, input_shape=(max_len,embedding_dim)))

model.add(TimeDistributed(Dense(n_tags, activation='softmax')))
model.compile(optimizer="Nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
history=model.fit_generator(train_gen, epochs=4, verbose=1, validation_data=validation_gen)
pred = model.predict_generator(test_gen, verbose=1)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

idx2tag = {i: w for w, i in dict_tag.items()}
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
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
test_labels = test2label(test_gen.get_true_labels()[:len(pred)])

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
iteration = 0
u_limit = 1
query_sz = 25
querier = MarginSelection()
predict_gen = DataGenerator("not_labeled_2", dict_tag, "data/unlabeled_tokens_2.xml", max_len, model1_embedding ,100, batch_size, "prediction")
pred = model.predict_generator(predict_gen, verbose=1)
sent_prob_mean =[]
for pred_i in pred:
    total_prob = K.epsilon()
    notpad = 1
    for p in pred_i:
        p_i = np.argmax(p)
        if idx2tag[p_i] != '<PAD>':
            total_prob = total_prob+p[p_i]
            notpad = notpad +1
    sent_prob_mean.append(total_prob/notpad)
entity_min_bysent =[]
for pred_i in pred:
    probs = []
    for p in pred_i:
        p_i = np.argmax(p)
        probs.append(p[p_i])
    entity_min_bysent.append(probs[np.argmin(probs)])

pred_labels = pred2label(pred)
test_labels = test2label(test_gen.get_true_labels()[:len(pred)])
#uncertain_sentMean =  querier.select(sent_prob_mean,query_sz)
#uncertain_sentMin =  querier.select(entity_min_bysent,query_sz)
random_sent =  querier.select(len(sent_prob_mean),query_sz)
#gerek yok
generated_indexes = predict_gen.get_IDs()
print(generated_indexes)
queries_mean = []
queries_random = []
queries_min = []

for p in uncertain_sentMean:
    queries_mean.append(int(generated_indexes[p]))
for p in uncertain_sentMin:
    queries_min.append(int(generated_indexes[p]))
for p in random_sent:
    queries_random.append(int(generated_indexes[p]))

#np.save("data\query_mean.npy",np.array(queries_mean))
n#p.save("data\query_min.npy",np.array(queries_min))
np.save("data\query_random.npy",np.array(queries_random))