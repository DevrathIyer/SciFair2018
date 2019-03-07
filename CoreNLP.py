from stanfordcorenlp import StanfordCoreNLP
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import math
import re

NEGATE = {"aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
 "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "d`oesn't",
 "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
 "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
 "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
 "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
 "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
 "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"}

CHANGE_WORDS = {'though','but','although'}
#GETTING SENTIMENT ANALYSIS LEXICON
sid = SentimentIntensityAnalyzer()
lexicon = sid.lexicon

lexicon2 = {}
file = open('data/SentiWords_1.0.txt').readlines()
for line in file:
    word, measure = line.split()
    word, pos = word.split('#')
    if word not in lexicon2:
        lexicon2[word] = {}
    lexicon2[word] = measure

#SIGMOID FUNCTION
def sigmoid(x):
    return x/math.sqrt(math.pow(x,2) + 15)

#PARSE FUNCTION
def parse(expr):
    def _helper(iter):
        items = {}
        startString = ""
        resultString = ""
        for item in iter:
            if item == '(':
                result, closeparen = _helper(iter)
                if not closeparen:
                    raise ValueError("bad expression -- unbalanced parentheses")
                if result:
                    if startString not in items:
                        items[startString] = []
                    items[startString].append(result)
            elif item == " ":
                if resultString:
                    startString = resultString
                resultString = ""
            elif item == ')':
                if resultString:
                    if startString not in items:
                        items[startString] = []
                    items[startString].append(resultString)
                return items, True
            else:
               resultString += item.strip()
        return items, False

    return _helper(iter(expr))[0]

def checkScore(word,lemma,sentence):
    if(word in dependencies):
        if (word in NEGATE):
            currscore = -1
        elif(lemma not in lexicon and lemma not in lexicon2):
            return 0
        else:
            if(lemma in lexicon):
                currscore = lexicon[lemma]
            else:
                currscore = lexicon2[lemma]
        for dependency in dependencies[word]:
            for item in sentence['tokens']:
                if dependency == item['word']:
                    dependentLemma = item['lemma']
            if(word != dependency):
                dependencyScore = float(checkScore(dependency, dependentLemma, sentence))
            else:
                dependencyScore = .1
            #   used.append(dependency)
            #else:
            #dependencyScore = .1
            currscore = float(currscore)*dependencyScore*100
        return currscore
    elif lemma in lexicon:
        return float(lexicon[lemma])
    elif (word in NEGATE):
        return -1
    elif lemma in lexicon2:
        return float(lexicon2[lemma])
    else:
        return 0
    return 0

maderAccuracy = []
vaderAccuracy = []
sentences = open('data/amazonReviewSentiments.txt').readlines()
accuracy1 = 0
accuracy2 = 0
count = 0


for sentence in sentences:
    count += 1
    text = sentence.split('\t')[2]
    goal = float(sentence.split('\t')[1])
    maderScore = 0

    #GETTING PROPERTIES FROM NLP
    properties = {"annotators": "tokenize,ssplit,pos,parse,depparse,openie", "outputFormat": "json"}
    nlp = StanfordCoreNLP('http://corenlp.run', port=80)
    ReturnedJson = json.loads(
        nlp.annotate(text,
                     properties=properties))
    for sentenceJSON in ReturnedJson['sentences']:
        constituency = parse(sentenceJSON['parse'])[''][0]['ROOT']

        #PARSING DEPENDENCIES
        dependencies = {}
        for dependency in sentenceJSON['enhancedPlusPlusDependencies']:
            if(dependency['dep'] == 'amod' or dependency['dep'] == 'advmod' or dependency['dep'] == 'nummod' or dependency['dep'] == 'neg'):
                if dependency['governorGloss'] not in dependencies:
                    dependencies[dependency['governorGloss']] = []
                dependencies[dependency['governorGloss']].append(dependency['dependentGloss'])

        used = []
        score = 0
        for item in sentenceJSON['tokens']:
            if(item['word'] in used):
                continue
            if(item['word'] in CHANGE_WORDS):
                score *=.1
            else:
                score += sigmoid(checkScore(item['word'],item['lemma'],sentenceJSON))
                used.append(item['word'])
        maderScore += score
        vaderScore = sid.polarity_scores(text)['compound']

    #print("Trial #{}:".format(count))
    #print("MADER Score:{}".format(maderScore))
    #print("VADER Score:{}".format(vaderScore))
    if((maderScore < 0 and goal < 0) or (maderScore > 0 and goal > 0)):
        accuracy1+=1
    if((vaderScore < 0 and goal < 0) or (vaderScore > 0 and goal > 0)):
        accuracy2+=1

    print(count)
    print("MADER Accuracy: {}%".format(accuracy1 * 100 / count))
    print("VADER Accuracy: {}%".format(accuracy2 * 100 / count))

print("MADER Accuracy: {}%".format(accuracy1 * 100 / count))
print("VADER Accuracy: {}%".format(accuracy2 * 100 / count))