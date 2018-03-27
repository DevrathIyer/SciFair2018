import gzip
from nltk.stem.porter import *
import operator
import operator

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
      yield eval(l)

reviews = {}
ps = PorterStemmer()

outfile = open('data/Tools.txt','w+',newline='\n')
for i in parse("data/reviews_Tools_and_Home_Improvement_5.json.gz"):
  if i['asin'] in reviews:
      reviews[i['asin']] += 1
  else:
      reviews[i['asin']] = 1
print(sorted(reviews.items(),key = operator.itemgetter(1),reverse=True)[:10])