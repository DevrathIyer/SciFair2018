import sys
import re
import json
from stanfordcorenlp import StanfordCoreNLP
from textblob import TextBlob as tb

#NLP Properties and server object
properties = {"annotators": "tokenize,ssplit,pos,parse,depparse,openie","outputFormat": "json"}
nlp = StanfordCoreNLP('http://corenlp.run', port=80)

#Read in plaintext data
dataset = str(sys.argv[1])
lines = open('data/{}.txt'.format(dataset))

#Read in tfidf terms from cmd line
tfidf_terms = []
for x in range(2,len(sys.argv)):
    tfidf_terms.append(str(sys.argv[x]))

#Iterate through each line in plaintext data
for line in lines:

    #Stip line of apostrophes, they just make it more confusing
    line = line.strip("'")
    line = str(tb(line).correct())

    #Break up review into individual sentences based on punctuation('.' and ';')
    sentences = list(filter(None, re.split('[.;!?\n]',line.lower())))

    #Iterate through each sentencs
    for sentence in sentences:

        #Query CoreNLP server
        ReturnedJson = json.loads(nlp.annotate(sentence,properties=properties))['sentences'][0]
        #print(ReturnedJson)

        #Parsing Tokens
        sentenceTokens = [None]
        reverseDict = {}
        for token in ReturnedJson['tokens']:
            reverseDict[token['word']] = token['index']
            sentenceTokens.append([token['word'],token['lemma'],token['pos']])

        #Parsing Dependencies
        dependencies = {}
        for dependency in ReturnedJson['enhancedPlusPlusDependencies']:
            if dependency['dep'] == 'amod' or dependency['dep'] == 'advmod' or dependency['dep'] == 'nummod' or dependency['dep'] == 'neg':
                if dependency['governor'] not in dependencies:
                    dependencies[dependency['governor']] = []
                dependencies[dependency['governor']].append(dependency['dependent'])

        #Parsing OpenIE Relations
        relations = []
        print(sentence)
        for relation in ReturnedJson['openie']:
            relations.append({'subject': relation['subjectSpan'],
                              'subjectSpan': relation['subjectSpan'][1] - relation['subjectSpan'][0],
                              'relation': relation['relationSpan'], 'object': relation['objectSpan'],
                              'objectSpan': relation['objectSpan'][1] - relation['objectSpan'][0]})
        print(relations)

            #print(relations)