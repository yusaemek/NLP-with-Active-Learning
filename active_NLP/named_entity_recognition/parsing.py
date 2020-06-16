print("ok")
# -*- coding: utf-8 -*-
from warnings import filterwarnings
from lxml import etree as ET
from nltk.tokenize import TweetTokenizer
filterwarnings('ignore')
import numpy as np 
import pandas as pd
import nltk
import nltk.data
import string
import re
import ast
from nltk.tokenize import PunktSentenceTokenizer as punkt
from bs4 import BeautifulSoup


def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\\n", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub("\n", "", text)
    return text
tknzr = TweetTokenizer()

#org = o time = t  geo = g person = p  other=0
def remove_punctuation( text):
        text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations
        return text_nopunct

data_cumh = pd.read_pickle("./cumh_prep_full.pkl")
data_cumh.sort_index(inplace=True)
data_cumh=data_cumh.reset_index()
cumh_splitted = np.array_split(data_cumh, 3)
sent_detector = nltk.data.load('tokenizers/punkt/turkish.pickle')





#crps_news = ET.Element("corpus_news", name="cumhuriyet", size=str(len(news_texts)))
crps_lbldtokens = ET.Element("corpus_labeledtokens", name="cumhuriyet")
#crps_sentences = ET.Element("corpus_parsedsentences", name="cumhuriyet")

#tree_news = ET.ElementTree(crps_news)
tree_tokens = ET.ElementTree(crps_lbldtokens)
#tree_sent = ET.ElementTree(crps_sentences)

for ind, row in cumh_splitted[0].iterrows():
	#news = ET.SubElement(crps_news, "news", n_ind = str(ind), Category = str(news_class[ind]))
	#print(news_texts)
	#news.text = str(news_texts[ind])
	parsed_sents = sent_detector.tokenize(row['text'].strip())
	for s in range(len(parsed_sents)):
		#sentence = ET.SubElement(crps_sentences, "sent", n_i=str(ind), i=str(s), lbld="no")
		#sentence.text =  parsed_sents[s]
		s_list = tknzr.tokenize(parsed_sents[s])
		entry = ET.SubElement(crps_lbldtokens, "s", n_i = str(ind), i=str(s), lbld= "no")
		for w in s_list:
			if (len(w)>1 and w!="...") or w.isalnum() :
				ET.SubElement(entry, "e", l="r").text = str(w)






#tree_news.write("corpus_news.xml")
tree_tokens.write("a_corpus_tokens_1.xml")


#tree_sent.write("corpus_senteneces.xml")
print("ok")
# -*- coding: utf-8 -*-
from warnings import filterwarnings
from lxml import etree as ET
from nltk.tokenize import TweetTokenizer
filterwarnings('ignore')
import numpy as np 
import pandas as pd
import nltk
import nltk.data
import string
import re
import ast
from nltk.tokenize import PunktSentenceTokenizer as punkt
from bs4 import BeautifulSoup


def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\\n", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub("\n", "", text)

    return text
tknzr = TweetTokenizer()

#org = o time = t  geo = g person = p  other=0
def remove_punctuation( text):
        text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations
        return text_nopunct

data_cumh = pd.read_pickle("./cumh_prep_full.pkl")
data_cumh.sort_index(inplace=True)
data_cumh=data_cumh.reset_index()
cumh_splitted = np.array_split(data_cumh, 3)
sent_detector = nltk.data.load('tokenizers/punkt/turkish.pickle')





#crps_news = ET.Element("corpus_news", name="cumhuriyet", size=str(len(news_texts)))
crps_lbldtokens = ET.Element("corpus_labeledtokens", name="cumhuriyet")
#crps_sentences = ET.Element("corpus_parsedsentences", name="cumhuriyet")

#tree_news = ET.ElementTree(crps_news)
tree_tokens = ET.ElementTree(crps_lbldtokens)
#tree_sent = ET.ElementTree(crps_sentences)

for ind, row in cumh_splitted[2].iterrows():
	#news = ET.SubElement(crps_news, "news", n_ind = str(ind), Category = str(news_class[ind]))
	#print(news_texts)
	#news.text = str(news_texts[ind])
	parsed_sents = sent_detector.tokenize(row['text'].strip())
	for s in range(len(parsed_sents)):
		#sentence = ET.SubElement(crps_sentences, "sent", n_i=str(ind), i=str(s), lbld="no")
		#sentence.text =  parsed_sents[s]
		s_list = tknzr.tokenize(parsed_sents[s])
		print(s_list)        
		entry = ET.SubElement(crps_lbldtokens, "s", n_i = str(ind), i=str(s), lbld= "no")
		for w in s_list:
			if (len(w)>1 and w!="...") or w.isalnum() :
				ET.SubElement(entry, "e", l="r").text = str(w)






#tree_news.write("corpus_news.xml")
tree_tokens.write("a_corpus_tokens_3.xml")


#tree_sent.write("corpus_senteneces.xml")
