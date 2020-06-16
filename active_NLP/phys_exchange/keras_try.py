# LSTM for sequence classification in the IMDB dataset

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import html as ihtml
import re
import ast

def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class Preprocessing(object):
    def __init__(self, data, target_column_name='body_text_clean'):
        self.data = data
        self.feature_name = target_column_name
        
    def remove_punctuation(self, text):
        text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations
        return text_nopunct
    
    def tokenize(self, text):
        # Match one or more characters which are not word character
        tokens = re.split('\W+', text) 
        return tokens
    
    def remove_stopwords(self, tokenized_list):
        # Remove all English Stopwords
        stopword = nltk.corpus.stopwords.words('english')
        text = [word for word in tokenized_list if word not in stopword]
        return text    

    def stemming(self, tokenized_text):
        ps = nltk.PorterStemmer()
        text = [ps.stem(word) for word in tokenized_text]
        return text
    
    def lemmatizing(self, tokenized_text):
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in tokenized_text]
        return text
    
    def tokens_to_string(self, tokens_string):
        
        text = [" ".join(list_obj) for list_obj in tokens_string]
        return text

    def dropna(self):
        feature_name = self.feature_name
        if self.data[feature_name].isnull().sum() > 0:
            column_list=[feature_name]
            self.data = self.data.dropna(subset=column_list)
            return self.data
        
    def preprocessing(self):
        self.data['Body']= self.data['Body'].apply(lambda x: clean_text(x))
        self.data['body_text_nopunc'] = self.data['Body'].apply(lambda x: self.remove_punctuation(x))
        self.data['body_text_tokenized'] = self.data['body_text_nopunc'].apply(lambda x: self.tokenize(x.lower())) 
        self.data['body_text_nostop'] = self.data['body_text_tokenized'].apply(lambda x: self.remove_stopwords(x))
        self.data['body_text_stemmed'] = self.data['body_text_nostop'].apply(lambda x: self.stemming(x))
        self.data['body_text_lemmatized'] = self.data['body_text_nostop'].apply(lambda x: self.lemmatizing(x))
        # save cleaned dataset into csv file and load back
        self.data[self.feature_name] = self.data['body_text_lemmatized'].apply(lambda x: ' '.join(x)) 

        
        drop_columns = ['body_text_nopunc', 'body_text_tokenized', 'body_text_nostop', 'body_text_stemmed', 'body_text_lemmatized'] 
        self.data.drop(drop_columns, axis=1, inplace=True)
        self.data.to_pickle("./prep_phys.ex.pkl")

        return self.data
    
    def save(self, filepath="/home/ao/active_NLP/"):
        self.data.to_csv(filepath, index=False, sep=',')  
        
    def load(self, filepath="/home/ao/active_NLP/"):
        self.data = pd.read_csv(filepath)
        return self.data

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

phy_data = pd.read_pickle("./prep_phys.ex.pkl")
classes = np.load("./phy_2go_class.npy")


pp = Preprocessing(phy_data)
phy_data = pp.preprocessing()
#print(phy_data)
#print(phy_data['body_text_clean'])

X, y = phy_data['body_text_clean'], classes



# fit the tokenizer on the documents
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


train_size = int(len(X) * training_portion)

X_train = X[0: train_size]
y_train = y[0: train_size]

X_valid = X[train_size:]
y_valid = y[train_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index


train_sequences = tokenizer.texts_to_sequences(X_train)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



validation_sequences = tokenizer.texts_to_sequences(X_valid)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



print("X_train after load:: ",X_train)


# truncate and pad input sequences
max_review_length = 32

# create the model

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 9
history = model.fit(train_padded, y_train, epochs=num_epochs, validation_data=(validation_padded, y_valid), verbose=1)


# Final evaluation of the model

 

test_sequences = tokenizer.texts_to_sequences(X_test)

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


scores = model.evaluate(test_padded, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


