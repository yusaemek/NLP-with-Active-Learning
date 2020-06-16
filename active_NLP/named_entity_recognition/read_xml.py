from warnings import filterwarnings
from lxml import etree 
import string
import random
tree_t = etree.parse("a_corpus_tokens_2.xml")

root_t = tree_t.getroot()

print (root_t.attrib["name"])


num2_label = input("pls enter the number of news texts to label their named entities: ")
not_lbld = root_t.findall(".//s[@lbld='no']")
indexes = random.sample(list(set([ int(sent.attrib["n_i"]) for sent in not_lbld ])), int(num2_label))
print("pls enter label of following in the following format index_1-label_1 index_2-label_2, if not labeled as 0 (i.e. other)")      
for i in indexes:
	print("NEWS INDEX IS:", i)
	str_att = ".//s[@n_i='" + str(i) + "']"
	token_group = root_t.findall(str_att)
	str_att = ".//sent[@n_i='" + str(i) + "']"
	sentences = root_s.findall(str_att)
	for m in range(len(sentences)):
		print("STATUS IS:  ",token_group[m].attrib["lbld"],".")
		print("SENTENCE IS: ", sentences[m].text)
		for i in range(len(token_group[m])):
			print(i+1,": ", token_group[m][i].text, "	-current: ", token_group[m][i].attrib["l"])
		val = input("Enter your labels: ")
		val = val.split(' ')
		print(val)
		if val[0] != '':
			for i in range(len(val)):
				temp = val[i].split('-')
				if temp[1] in ["pi","pb","ni","nb","fi", "fb", "oi", "ob","wi","wb","di","db","ai","ab","li","lb","ti", "tb", "mi", "mb","r"]:
					token_group[m][int(temp[0])-1].attrib["l"] = temp[1]
				else:
					raise SystemExit('Error: label', temp[1], 'is not member of named entity set')
		sentences[m].attrib["lbld"] = 'yes'
		token_group[m].attrib["lbld"] = 'yes'
	tree_t.write("a_corpus_tokens_2.xml")
tree_s = etree.parse("corpus_sentences.xml")

root_s = tree_s.getroot()