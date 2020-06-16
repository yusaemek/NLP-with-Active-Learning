import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from modAL.density import information_density
from sklearn.model_selection import train_test_split


class MarginSelection():
    def __init__(self):
        self.name="MarginSelection"
    def select(self,probas_val, query_size): #MarginSamplingSelection
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:query_size]
        return selection
    def rando_sample(self,size, query_size): #MarginSamplingSelection
        #random_state = check_random_state(0)
        selection = np.random.choice(size, query_size, replace=False)
        print(selection)
        return selection

class infodensity():
    def __init__(self,X,Y):
        self.name="information density"
        #self.model= QueryInstanceQUIRE(train_data)
        self.labeled=[]
        self.X=X
        self.unlabeled=np.arange(len(X))
        self.densities = information_density (X, 'manhattan')
    def sample(self, query_size): #MarginSamplingSelection
        
        uncertain_samples=[]
        for i in range(query_size):
            indexArr=np.argmax(self.densities[self.unlabeled])
            uncertain_samples.append(indexArr)
            self.unlabeled=np.delete(self.unlabeled,indexArr)
            self.labeled=np.append(self.labeled,indexArr)
        
        return uncertain_samples

np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

phy_data = pd.read_pickle("./prep_phys.ex.pkl")
classes = np.load("./phy_2go_class.npy")

X, y = phy_data['body_text_clean'], classes

X=X.to_numpy()

query_sz=20
labeled=50
iteration=0
init_data=50
training_size = []
accuracy_test = []

X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

queried=np.random.choice(X_pool.shape[0], init_data, replace=False)

X_queried = X_pool[queried]
y_queried = y_pool[queried]


train_size = int(len(X_queried) * training_portion)

X_train = X_queried[0: train_size]
y_train = y_queried[0: train_size]

X_valid = X_queried[train_size:]
y_valid = y_queried[train_size:]


querier = MarginSelection()
labeled=init_data

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
num_epochs = 3

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_pool)

X_pool = np.delete(X_pool, queried)
y_pool = np.delete(y_pool, queried)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(X_valid)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
iteration=iteration+1

training_size.append(query_sz)


test_sequences = tokenizer.texts_to_sequences(X_test)

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

history = model.fit(train_padded, y_train, epochs=num_epochs, validation_data=(validation_padded, y_valid), verbose=1)


scores = model.evaluate(test_padded, y_test, verbose=1)
accuracy_test.append(scores[1]*100)
while labeled < 500:
    labeled=labeled+query_sz
    probas_val = model.predict_proba(tokenizer.texts_to_sequences(X_pool))
    uncertain_samples =  querier.select(probas_val,query_sz)
    print("iterated")
            
    X_queried = np.concatenate((X_queried, X_pool[queried]))

    y_queried = np.concatenate((y_queried, y_pool[queried]))

    X_pool = np.delete(X_pool, queried)
    y_pool = np.delete(y_pool, queried)


    train_size = int(len(X_queried) * training_portion)

    X_train = X_queried[0: train_size]
    y_train = y_queried[0: train_size]

    X_valid = X_queried[train_size:]
    y_valid = y_queried[train_size:]

    tokenizer.fit_on_texts(X_queried)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    validation_sequences = tokenizer.texts_to_sequences(X_valid)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    iteration=iteration+1

    training_size.append(query_sz)

    history = model.fit(train_padded, y_train, epochs=num_epochs, validation_data=(validation_padded, y_valid), verbose=1)

    test_sequences = tokenizer.texts_to_sequences(X_test)
    
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


    scores = model.evaluate(test_padded, y_test, verbose=1)
    accuracy_test.append(scores[1]*100)
    training_size.append(query_sz*iteration)

import matplotlib.pyplot as plt
plt.title('LSTM /w nformation density')
plt.plot(training_size,accuracy_test)
plt.ylabel ('accuracy')
plt.xlabel ('# of labeled instances')
plt.show()
