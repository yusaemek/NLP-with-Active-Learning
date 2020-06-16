from lxml import etree
import gensim.models
from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
import keras
from gensim.models.fasttext import FastText
import string
from pprint import pprint as print
from sklearn.model_selection import train_test_split

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