from nltk.corpus import brown
import string

# Remove stop words
from nltk.corpus import stopwords
stop_words = stopwords.words(fileids='english')
listWords = brown.words()
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
print(brown_preprocessed[:50])
