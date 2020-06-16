
from warnings import filterwarnings

filterwarnings('ignore')
import numpy as np 
import pandas as pd
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
    text = re.sub(r"\\n", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub("\n", "", text)
    #text = re.sub("", "", text)

    #tokenizer = RegexpTokenizer('\s+', gaps=True)
    #text = tokenizer.tokenize(text)

    return text
class Preprocessing(object):
    def __init__(self, data, target_column_name='text_clean'):
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
        self.data['text']= self.data['text'].apply(lambda x: clean_text(x))
        #self.data['body_text_nopunc'] = self.data['text'].apply(lambda x: self.remove_punctuation(x))
        #self.data['body_text_tokenized'] = self.data['body_text_nopunc'].apply(lambda x: self.tokenize(x.lower())) 
        #self.data['body_text_nostop'] = self.data['body_text_tokenized'].apply(lambda x: self.remove_stopwords(x))
        #self.data['body_text_stemmed'] = self.data['body_text_nostop'].apply(lambda x: self.stemming(x))
        #self.data['body_text_lemmatized'] = self.data['body_text_nostop'].apply(lambda x: self.lemmatizing(x))
        #save cleaned dataset into csv file and load back
        #self.data[self.feature_name] = self.data['body_text_lemmatized'].apply(lambda x: ' '.join(x)) 

        
        #drop_columns = ['body_text_nopunc', 'body_text_tokenized', 'body_text_nostop', 'body_text_stemmed', 'body_text_lemmatized'] 
        #self.data.drop(drop_columns, axis=1, inplace=True)
        #self.data.to_pickle("./prep_phys.ex.pkl")

        return self.data
    

data_cumh_full = pd.read_pickle("./cumh_raw_full.pkl")
data_cumh = data_cumh_full.sample(frac=1)

data_cumh.sort_index(inplace=True)
list2=data_cumh['text'].tolist()
for i in list2:
	print(i)
	print(list2[5])


pp = Preprocessing(data_cumh)
data_cumh = pp.preprocessing()

# Comment out this line to study whole set
#data_cumh = data_cumh_full

print("NUMBER OF NEWSPAPER WITH LABEL econ=    ",len([1 for m in data_cumh['econ'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL tech=    ",len([1 for m in data_cumh['tech'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL yazarlar=    ",len([1 for m in data_cumh['yazarlar'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL video=    ",len([1 for m in data_cumh['video'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL spor=    ",len([1 for m in data_cumh['spor'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL türkiye=    ",len([1 for m in data_cumh['türkiye'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL siyaset=    ",len([1 for m in data_cumh['siyaset'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL foto=    ",len([1 for m in data_cumh['foto'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL kültür-sanat=    ",len([1 for m in data_cumh['kültür-sanat'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL yaşam=    ",len([1 for m in data_cumh['yaşam'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL sağlık=    ",len([1 for m in data_cumh['sağlık'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL eğitim=    ",len([1 for m in data_cumh['eğitim'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL çevre=    ",len([1 for m in data_cumh['çevre'] if(m==1)]))
print("NUMBER OF NEWSPAPER WITH LABEL dünya=    ",len([1 for m in data_cumh['dünya'] if(m==1)]))
data_cumh['Category'] = 0

data_cumh.loc[data_cumh.econ == 1, 'Category'] = 1
data_cumh.loc[data_cumh.tech == 1, 'Category'] = 0
data_cumh.loc[data_cumh.siyaset == 1, 'Category'] = 2
data_cumh.loc[data_cumh.spor == 1, 'Category'] = 3
data_cumh.loc[data_cumh.yazarlar == 1, 'Category'] = 3
data_cumh.loc[data_cumh.video == 1, 'Category'] = 4
data_cumh.loc[data_cumh.türkiye == 1, 'Category'] = 5
data_cumh.loc[data_cumh.foto == 1, 'Category'] = 6
#data_cumh.loc[data_cumh.'kültür-sanat' == 1, 'Category'] = 7
data_cumh.loc[data_cumh.yaşam == 1, 'Category'] = 8
data_cumh.loc[data_cumh.eğitim == 1, 'Category'] = 9
data_cumh.loc[data_cumh.çevre == 1, 'Category'] = 10
data_cumh.loc[data_cumh.dünya == 1, 'Category'] = 11
data_cumh.loc[data_cumh.sağlık == 1, 'Category'] = 12
data_cumh.loc[data_cumh.spor == 1, 'Category'] = 12

data_cumh['Category']=data_cumh['Category'].astype(int)

print(data_cumh)



data_cumh.to_pickle("./cumh_prep_full.pkl")

