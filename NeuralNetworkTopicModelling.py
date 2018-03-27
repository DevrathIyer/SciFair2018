"""

import csv
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import operator

lines = open('data/reuters/TestTopics.txt').readlines()
outfile = open('TestingData.csv','w+')
stop_words = stopwords.words('english')
data = []
topics = []
writer = csv.writer(outfile)
for line in lines:
    filename = line.split(' ')[0]
    file = open('data/reuters/{}'.format(filename))
    data.append(file.read().replace('\n', ''))

    for item in line.split(' ')[1:]:
        topics.append(item)

tfidf_vectorizer = TfidfVectorizer(max_df=.95, min_df=2, max_features=1000, stop_words='english', norm=None)
tfidf_vectorizer.fit_transform(data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
tfidf_scores = dict(sorted(dict(zip(tfidf_feature_names,tfidf_vectorizer.idf_)).items(),key=operator.itemgetter(1)))

print(tfidf_scores)
for key in tfidf_scores:
    writer.writerow([tfidf_scores[key], int(key in topics)])
"""


from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
dataset = numpy.loadtxt("TrainingData.csv", delimiter=",")
X = dataset[:,0:1]
Y = dataset[:,1]

model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)