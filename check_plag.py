# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:22:48 2023

Project Name : Plagiarism Detector in Python using Machine Learning Techniques
Author : Ms. Debalina Dasgupta
         UEMK

Program Description :
---------------------
The program reads text files from the folder, named as Docs under the current 
working directory. It reads all the text files having names Text_?.txt where ?
is to be replaced with any digit between 0 to 9.

The program applies word embedding techniques and first converts the textual 
data, read from the files, into an array of numbers (word vectors) using 
Term frequency-inverse document frequency (TF-IDF) method. For this purpose, 
TfidfVectorizer of scikit-learn built-in features is used.

All the pairs of word vecotrs are then processed for checking of any plagiarism 
between the corresponding text files. This is accomplished by computing the 
value of cosine similarity between the vectors representations of the concerned
text files.

Finally, a table of plagiarism percentage between every pair of files, read 
from the Docs folder is prepared along with presenting the result in Bar Graph.

Requirements :
--------------
The program requires scikit-learn to be installed in the machine.

References :
https://scikit-learn.org/
https://www.turing.com/kb/guide-on-word-embeddings-in-nlp
https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
https://www.geeksforgeeks.org/word-embeddings-in-nlp/
https://dev.to/kalebu/how-to-detect-plagiarism-in-text-using-python-dpk
https://towardsdatascience.com/simple-plagiarism-detection-in-python-2314ac3aee88
https://www.geeksforgeeks.org/word-embeddings-in-nlp/
https://www.geeksforgeeks.org/cosine-similarity/

"""
#---------- Import the libraries.
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
import matplotlib.pyplot as plt
import numpy as np

#---------- Function for displaying the result in Bar Graph
def disp_bar(data):
    x = []
    y = []
    for data in result:
        s = data[0][:data[0].index('.txt')]+' vs. '+data[1][:data[1].index('.txt')]
        x.insert(0,s)
        y.insert(0,data[2])
    h = 1/len(x)-1
    plt.barh(x,y,height=h)
    plt.title('Percentage of Plagiarism')
    plt.ylabel('%')
    plt.xlabel('')
    plt.show()

#---------- Function for comparing documents
def compare_docs():
    global doc_vectors
    tableScores = set()
    for doc_1, vector_1 in doc_vectors:
        list_vectors = doc_vectors.copy()
        index = list_vectors.index((doc_1, vector_1))
        print(str(index+1)+' Compairing the document in the file '+doc_1+' .....')
        del list_vectors[index]
        for doc_2, vector_2 in list_vectors:
            score = cosine_similarity([vector_1, vector_2])
            doc_pair = sorted((doc_1, doc_2))
            doc_pair_score = (doc_pair[0], doc_pair[1], 100*score[0][1])
            tableScores.add(doc_pair_score)
    return tableScores

#---------- Find the files having names Text_?.txt from the folder, named as 
#---------- Docs, under the current working directory and create the list of files.

list_files = [f for f in os.listdir('.\\Docs') if re.search("^Text_[0-9].txt",f)]
print('List of files found in the Docs folder under the current working directory .....')
print(list_files)

#---------- Load the documents from the text files.
list_docs = [open('.\\Docs\\'+f, encoding='utf-8').read()
                 for f in list_files]

#---------- Create the model of word vectors (document-term matrix) using 
#---------- TF-IDF for all the loaded documents. 
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(list_docs).toarray()

print('\nShape of the vectors : ',str(vectors.shape))
print('\nList of distinct words found in all the documents .....')
print(vectorizer.get_feature_names_out())
#print(vectors)

#---------- Create the list combining the vectors with the respective file names.
doc_vectors = list(zip(list_files, vectors))

#---------- Compute the score for checking the percentage of plagiarism among
#---------- the loaded text files.
result = compare_docs()

#---------- Display the result.
print('------------------------------ Result ------------------------------')
print('\t {:27} \t\t\t {}'.format('       Document Pairs', "Plagiarism %"))
print('\t------------------------------------------------------')
for data in result:
    print('\t {:11}  vs.  {:11} \t\t\t {:8.5} %'.format(data[0], data[1], data[2]))

#---------- Draw the Bar Graph against the result.
disp_bar(data)
