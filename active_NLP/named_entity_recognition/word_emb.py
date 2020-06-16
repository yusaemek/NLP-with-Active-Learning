import pandas as pd
import numpy as np
from warnings import filterwarnings
from lxml import etree
import string
tree_t_3 = etree.parse("a_corpus_tokens_3.xml")

root_t_3 = tree_t_3.getroot()

tree_t_1 = etree.parse("a_corpus_tokens_1.xml")

root_t_1 = tree_t_1.getroot()

tree_t_2 = etree.parse("a_corpus_tokens_2.xml")

root_t_2 = tree_t_2.getroot()

all_1 = root_t_1.findall(".//s") 
all_2 = root_t_2.findall(".//s") 
all_3 = root_t_3.findall(".//s") 
all_sent = all_1 + all_2 + all_3
class MyCorpus(object):
    def __iter__(self):
        for sentences in all_sent:
                yield ([entities.text.lower() for entities in sentences])
from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
model = FastText(size=200, min_n = 3,window = 7 )
model.build_vocab(sentences=MyCorpus())
model.train(
     sentences=MyCorpus(), epochs=model.epochs,
    total_examples=model.corpus_count, total_words=model.corpus_total_words)
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("lower_better_bigger.model")
model.save(fname)
model = FastText.load(fname)
import gensim.models

sentences = MyCorpus()
model_vec2 = gensim.models.Word2Vec(sentences=sentences, size=200)
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("lower_better_bigger_2vec.model")
model_vec2.save(fname)
model_vec2 = gensim.models.Word2Vec.load(fname)