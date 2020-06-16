from warnings import filterwarnings
from lxml import etree 
import string
import random
import numpy as np
parser = etree.XMLParser(remove_blank_text=True)
tree_l = etree.parse("labeled_tokens_3.xml",parser)
root_l = tree_l.getroot()

tree_n = etree.parse("notlabeled_tokens_3.xml",parser)
root_n = tree_n.getroot()
indexes = np.load('query_test.npy')
for i in indexes[]:
    print(i)
    str_att = ".//s[@i='" + str(i) + "']"
    token_group = root_n.findall(str_att)
    print(len(token_group))
    for m in range(len(token_group)):
        for i in range(len(token_group[m])):
            print(i+1,": ", token_group[m][i].text, "	-current: ", token_group[m][i].attrib["l"])
        val = input("Enter your labels: ")
        val = val.split(' ')
        print(val)
        if val[0] != '':
            for i in range(len(val)):
                temp = val[i].split('.')
                if temp[1] in ["B-PER","I-PER","I-NQP","B-NQP", "I-ORG", "B-ORG","I-DTE","B-DTE","I-LOC","B-LOC","I-TIT", "B-TIT", "I-MNY", "B-MNY","O"]:
                    token_group[m][int(temp[0])-1].attrib["l"] = temp[1]
                else:
                    raise SystemExit('Error: label', temp[1], 'is not member of named entity set')
        token_group[m].attrib["lbld"] = 'yes'
    print((token_group[0].attrib["lbld"]))
    print(len(root_l.findall(".//s")))
    token_group[0].getparent().remove(token_group[0])
    root_l.append(token_group[0])
    print(len(root_l.findall(".//s")))

    
root_l.attrib["len"] = str(int(root_l.attrib["len"]) + len(indexes))
root_n.attrib["len"] = str(int(root_n.attrib["len"]) - len(indexes))

tree_n.write("notlabeled_tokens_3.xml")
tree_l.write("labeled_tokens_3_extended.xml")
print(  len(etree.tostring(tree_l, pretty_print=True)))
print( len(etree.tostring(new_tree_l, pretty_print=True)))