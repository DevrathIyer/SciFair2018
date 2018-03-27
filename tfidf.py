import math
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from lxml import html
import csv
import sys
import time
import gzip
from pathlib import Path
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
start_time = time.time()
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

is_noun = lambda pos: pos[:2] == 'NN'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
      yield eval(l)

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def n_times(word, bloblist):
    count = 0
    for blob in bloblist:
        for words in blob.words:
            if words == word:
                count+=1
    return count

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

tfidf_dict = {}
bloblist = []

#outfile = open('data/models/%s_tfidf3.csv' % (dataset), "w+")
#myFields = ['word', 'score']
#writer = csv.DictWriter(outfile, fieldnames=myFields)
#writer.writeheader()

dataset = 'B007WTAJTO'
with open("data/B007WTAJTO.txt", 'r+') as myfile:
    lines = myfile.readlines()
    for line in lines:
        blob = tb(line)
        bloblist.append(blob)

    print("Generating TF-IDF Data...")
    print(len(bloblist))
    print(n_times("card",bloblist))
    for i, blob in enumerate(bloblist):
        scores = {word: tfidf(word, blob,bloblist) for word in blob.words}
        for word, score in scores.items():
            if word not in tfidf_dict:
                tfidf_dict[word] = 0
            tfidf_dict[word] += score

    print(len(tfidf_dict))
    #for word,score in tfidf_dict.items():
    #    tfidf_dict[word] = score*idf(word,bloblist)

    print("Writing Output data...")
    sortedList = sorted(tfidf_dict.items(), key=lambda x:x[1])

    print(sortedList)

    tfidf_list = []
    for stuple in sortedList[:100]:
        if(stuple[0] not in stop_words and stuple[0] in nouns):
            tfidf_list.append(stuple[0])

    print(tfidf_list)


    page = requests.get("https://www.amazon.com/dp/%s" % (str(dataset)), headers=headers)

    out = str(dataset) + ' '
    doc = html.fromstring(page.content)
    title = (''.join(doc.xpath('//*[@id="productTitle"]//text()')).strip())
    title = (title[:50] + '..') if len(title) > 50 else title
    out += title + ' '
    LighthouseTermsRAW = '//span[@class="cr-lighthouse-term "]//text()'
    LighthouseTerms = ''.join(doc.xpath(LighthouseTermsRAW)).strip().split()

    plottedArray = [0 for i in range(0,100)]
    count = 0

    print(LighthouseTerms)
    for i,word in enumerate(tfidf_list):
        if(word in LighthouseTerms):
            count+=1
        plottedArray[i] = count
    plt.plot(plottedArray)
    plt.ylabel('some numbers')
    plt.show()

    myfile.close()

print("--- %s seconds ---" % (time.time() - start_time))
    #outfile.close()