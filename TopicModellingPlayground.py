from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import operator
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
from textblob import TextBlob as tb
from nltk.corpus import wordnet as wn
import time
import nltk
t0 = time.time()
stop_words = stopwords.words('english')

# How Many Features?
no_features = 1000
no_topics = 8

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}



lines = []
with open("data/B000I68BD4.txt", 'r+') as myfile:
    for line in myfile.readlines():
        t0 = time.time()
        tokens = nltk.word_tokenize(line)
        tagged = nltk.pos_tag(tokens)
        nouns = [word for word, pos in tagged \
                 if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
        downcased = [stemmer.stem(lemmatizer.lemmatize(x.lower())) for x in nouns]
        line = ' '.join(downcased)
        lines.append(line)

print(len(lines))

tfidf_vectorizer = TfidfVectorizer(max_df=.95, min_df=2, max_features=no_features, stop_words='english', norm=None)
tfidf_vectorizer.fit_transform(lines)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
tfidf_scores = sorted(dict(zip(tfidf_feature_names,tfidf_vectorizer.idf_)).items(),key=operator.itemgetter(1))
print(time.time() - t0)
print(tfidf_scores)

"""
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(lines)
tf_feature_names = tf_vectorizer.get_feature_names()
# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
print()
display_topics(lda, tf_feature_names, no_top_words)
"""