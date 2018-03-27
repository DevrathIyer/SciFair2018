from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk import word_tokenize, pos_tag
import nltk
sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
             "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
            "VADER is not very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
            "VADER is VERY SMART, handsome, and FUNNY.",# emphasis for ALLCAPS handled
            "I thought this product was going to be great, but it ended up being horrible!",
            "This is an amazing product!",
             "Total disappointment. DO NOT BUY!!!!",
            "My friend liked his product, but mine was a total disaster",
            "My son liked the colors, the sound, and the fit",
             "These earbuds have very poor bass and poor sensitivity",
             "These do just what was stated fit really nice, have different ear sets to tailor to your ears,and sound really good",
             "After a couple months, I'm really really bored",
             "It's not perfect of course but it works extremely well and it will only get better",
             "This is a very decent, inexpensive product",
             "Not only did my video playback improve but so did my signal strength",
             "I bought this and as soon as I set it up I got 4 bars and excellent service",
             "My reviews are usually long, but all I can say about this keyboard is it's truly a great product in every way",
             "It would have been nice to have a solar power feature, too",
             "it's not full sized and the F-keys were annoying until i found out logitech offers software to fix that",
             "This little keyboard is just what I was looking for",
             "Absolutely necessary if you purchase the Kindle since it is not included with it",
             "Amazon is doing something here that they rarely do, cheat the customer",
             "It's an obvious money-grubbing scheme",
             "I've had zero issues with it",
             "No issues with BluRay player or laptop that were used",
             "I'm no hdmi guru or techie but they do exaclty what I need them too and were a fraction of the price of other cables.",
]

lexicon = {}
file = open('data/SentiWords_1.0.txt').readlines()
for line in file:
    word, measure = line.split()
    word, pos = word.split('#')
    if word not in lexicon:
        lexicon[word] = {}
    lexicon[word] = measure


print(lexicon['issues'])
print(float(lexicon['very']['a']))
"""
for sentence in sentences:
    out = sentence + '\t'
    ss = sid.polarity_scores(sentence)
    out += str(ss['compound'])
    #for k in ss:
    #    out += '{}: {}'.format(k, ss[k]) + '\t'
    print(out)

tokenized_sentence = word_tokenize("This is very decent and inexpensive product,")
pos_tags = pos_tag(tokenized_sentence)
grammar = r""
  NP: {<DT|RB|PP|PRP|PDT|\$>?<JJ>*}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}
  V:  {<VB.>}# chunk sequences of proper nouns
""
"<CC>*<JJ>*<NN.?>"
cp = nltk.RegexpParser(grammar)
for parsed in cp.parse(pos_tags):
    print(parsed)"""
