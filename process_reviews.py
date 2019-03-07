import gzip
import json

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
      yield eval(l)

reviews = {}
num = 0
outfile = open('B009A5204K.txt','w+',newline='\n')
for i in parse("reviews_Electronics_5.json.gz"):
  if i['asin'] == "B009A5204K":
      outfile.write(json.dumps(i['reviewText']) + '\n')#', "qas": [{"answers": [{"answer_start": 177, "text": "Denver Broncos"}], "question": "How did they feel about the headphones?", "id": "56be4db0acb8001400a502ec'+str(num)+ '"}]},\n')
      num+=1
outfile.close()