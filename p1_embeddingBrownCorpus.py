from nltk.corpus import brown
import string

# Remove stop words
from nltk.corpus import stopwords
stop_words = stopwords.words(fileids='english')
listWords = brown.words()
sentence = 'The quick brown fox was too quick for the white fox brown quick \
fox'
listWords = sentence.split()
brown_noStopWords = [w for w in listWords if w.lower() not in stop_words]

# Remove punctuation
import re
brown_noPunctuation = [re.sub(r'[^\w]','',w) for w in brown_noStopWords]
# Remove the leftover empty strings, so that they don't get counted in the
# window operation
brown_noPunctuation = [w for w in brown_noPunctuation if w != '']

# stem the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
brown_preprocessed = [ stemmer.stem(w) for w in brown_noPunctuation ]
print('Brown corpus with stop words and punctuation removed, remaining words \
      stemmed')
print(brown_preprocessed[:50])

# The 5000 most common words are the vocabulary and the 1000 most common words
# are the Context.
VOCAB_SIZE = 4
CONTEXT_SIZE = 4
dictWordCnt = {}
for token in brown_preprocessed:
    if token in dictWordCnt.keys():
        dictWordCnt[token] += 1
    else:
        dictWordCnt[token] = 1
# sort the dictionary
cnt = 0
listSortedKeys = sorted(dictWordCnt, key=dictWordCnt.get, reverse=True)
print('Top 20 most frequest words in the brown corpus')
for w in listSortedKeys:
    cnt += 1
    print (w, ':', dictWordCnt[w])
    if cnt >= 20:
        break

# The end goal is to create a cooccurence matrix, so creating dictionaries
# mapping the words to indices and the indices to words
dictVocab2idx = {}
dictContext2idx = {}
dictIdx2Vocab = {}
dictIdx2Context = {}
for idx,token in enumerate(listSortedKeys[:VOCAB_SIZE]):
    dictVocab2idx[token] = idx
    dictIdx2Vocab[idx] = token
    if idx < CONTEXT_SIZE:
        dictIdx2Context[idx] = token
        dictContext2idx[token] = idx

cnt = 0
print('Top 20 words in the vocab')
for key,val in dictIdx2Vocab.items():
    print(key,val)
    cnt += 1
    if cnt > 20:
        break

# The cooccurence matrix represents how often the context words occur in a
# 4 word window around the words in the vocabulary.
# The co-occurence matrix has a dimension of |V|*|C| and each element (w,c)
# represents the number of times c occurs in a window around w.
import numpy as np
def print_2dec(a):
    print(np.round(a,2))

cooccurence_matrix = np.zeros(shape=(VOCAB_SIZE,CONTEXT_SIZE))
for index,w in enumerate(brown_preprocessed[2:-2]):
    print(w)
    actIndex = index + 2    # Actual index is 2 more than index
    # Check if the word is in Vocab, then check the two words prior to it and
    # the 2 words after it to see if they belong to the Context set. Increment
    # the (w,c) entry if they do.
    if w in dictVocab2idx:
        window_words = (brown_preprocessed[actIndex-2],brown_preprocessed[actIndex-1],
         brown_preprocessed[actIndex+1],brown_preprocessed[actIndex+2])
        print(window_words)
        for c in window_words:
            if c in dictContext2idx:
                cooccurence_matrix[dictVocab2idx[w]][dictContext2idx[c]] += 1
print('cooccurence_matrix:')
print(cooccurence_matrix)
# Using the co-occurrence matrix, we can compute the probability distribution
# Pr(c|w) of context word c around w as well as the overall probability
# distribution of each context word c with Pr(c).

cwPDF = np.zeros(shape=(VOCAB_SIZE,CONTEXT_SIZE))
for wIdx,cwArray in enumerate(cooccurence_matrix):
    sumCw = sum(cwArray)
    for cIdx,val in enumerate(cwArray):
        cwPDF[wIdx][cIdx] = val/sumCw

print('cwPDF')
print_2dec(cwPDF)

sumC = np.sum(np.transpose(cooccurence_matrix),axis=1)
cPDF = sumC/np.sum(cooccurence_matrix)

print('cPDF')
print_2dec(cPDF)

# Now you can represent each vocabulary word as a |C| dimensional vector using this equation:
# Vector(w)= max(0, log (Pr(c|w)/Pr(c)))

embedding = np.log(cwPDF/cPDF[:,None])
embedding = np.maximum(0,embedding)

print('embedding')
print_2dec(embedding)

# 1.5 is the analysis of the embedding in order to check if the findings make
# sense. This involves using k-means to find clusters in the vocabulary and
# also finding the nearest neighbors for the top-20 words.


