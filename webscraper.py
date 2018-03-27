import sys
import gzip
import csv
import requests
import operator
import math
from lxml import html
from nltk.stem.porter import *
from nltk.corpus import stopwords
from textblob import TextBlob as tb


headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}

#GZIP PARSER
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
      yield eval(l)

#IFIDF CODE
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


for i in range(1,len(sys.argv)):
    page = requests.get("https://www.amazon.com/dp/%s" % (str(sys.argv[i])), headers=headers)

    out = str(sys.argv[i]) + ' '
    doc = html.fromstring(page.content)
    title = (''.join(doc.xpath('//*[@id="productTitle"]//text()')).strip())
    title = (title[:50] + '..') if len(title) > 50 else title
    out += title + ' '
    LighthouseTermsRAW = '//span[@class="cr-lighthouse-term "]//text()'
    LighthouseTerms = ''.join(doc.xpath(LighthouseTermsRAW)).strip().split()

    out = out + " ".join(LighthouseTerms)
    print(out)

"""
for asin,count in reviews:

    bloblist = []
    dataset = str(sys.argv[i])
    print (dataset)
    lines = open('data/%s_topics_in.txt' % (dataset)).readlines()

    outfile = open('data/models/%s_tfidf2.csv' % (dataset), "w+")
    myFields = ['word', 'score']
    writer = csv.DictWriter(outfile, fieldnames=myFields)
    writer.writeheader()

    for line in lines:
        if(len(line) < 200):
            continue
        blob = tb(line.replace("'", ""))
        lista = [word for word in blob.noun_phrases if word not in stopwords.words('english')]
        line = ' '.join(lista)
        blob = tb(line)
        bloblist.append(blob)

    for i, blob in enumerate(bloblist):
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:3]:
            writer.writerow({'word': '{}'.format(word), 'score': '{}'.format(round(score, 5))})
page = requests.get("https://www.amazon.com/dp/B01DFKC2SO", headers = headers)

doc = html.fromstring(page.content)
LighthouseTermsRAW = '//span[@class="cr-lighthouse-term "]//text()'
LighthouseTerms = ''.join(doc.xpath(LighthouseTermsRAW)).strip().split()

for term in LighthouseTerms:
    print(term)

"""