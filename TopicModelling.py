import sys
import keras
import gensim
from keras.datasets import imdb
from nltk.tokenize import SpaceTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from gensim import corpora, models
# Set up log to terminal
import logging
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Console Logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#NUM_WORDS=1000 # only use top 1000 words
#INDEX_FROM=3

tokenizer = SpaceTokenizer()
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

"""
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
"""

texts = []

#id_to_word = {value:key for key,value in word_to_id.items()}
file_object  = open("Gone_Girl.txt", "r")
for i in range(0,7441):
    item2 = file_object.readline() #item2 = ' '.join(id_to_word[id] for id in x_train[i])
    tokens = tokenizer.tokenize(item2)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    text = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(text)

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=int(sys.argv[2]), id2word = dictionary, passes=int(sys.argv[1]))

output = open("LDA_OUTPUT.txt",'w')

output.write(ldamodel.print_topics(num_topics=int(sys.argv[2]), num_words=int(sys.argv[3])))

output.close()
