# CodeClause_project2_Plagiarism-Checker
AI Project on Plagiarism Check
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
