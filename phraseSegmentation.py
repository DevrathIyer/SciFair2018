import nltk
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import sentimentAnalysis

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

reviews = [
"The design is near perfect for listening without the hassle of continually trying to maintain in your ear... The qaulity of these ear phone are excellent.",
"These little ear buds are great.  The sound is really good and they are actually durable.  I go on a lot of business trips (flying) and so I like to bring along some of my own ear buds for better sound quality and to save myself 2 bucks.  These ear buds have been exactly what I was looking for: very inexpensive, great sound and have lasted for nearly a year now with no failures.",
"My son liked the colors, the sound, and the fit.  I bought them to go with his ipod otter box that had similar colors.  He was very happy with both.  He wanted Beats which were too expensive but seemed to like these just fine.",
"I'm a careless person and usually obliterate earbuds in less than a month. These however, lasted 6 months (pretty good i spent less than 9 dollars on it). I loved the sound and it was comfortable.",
"Fits in ears with a tight sealed comfortable fit. I can hear partial traffic noise while walking home. The right side went off after 3 months of use. So , I bought a pair of buds from another company to replace these.",
"Needed another set and came across this.... For the price, it was worth the try and I can honestly say that these aren't bad at all. Decent sound quality with a good balance of treble and bass.... As far as the noise reducing factor.... Well, for the price, it isn't bad at all either... But keep this in mind..... Do not compare these to Bose earplugs because you will probably hate these... But for 100 dollars Bose, you expect perfect sound and noise reduction. I actually used these while I was vacuuming, and they actually held the sound out pretty well... Again, for the money, it's worth the try for yourself.",
"While the clich&eacute; title of this review might imply this product is clich&eacute;, don't make that mistake!This set of headphones offers excellent sound blocking with the flexible earplugs, the cord is plenty long, and for the price, you simply cannot go wrong here.Especially if you are upgrading from the default iPod earphones, or a cheap equivalent, you will be quite impressed when you can, once again, catch all the subtle, quieter nuances of your favorite songs.",
"I can't believe I never reviewed these earphones! I first got this pair 3.5 years ago, but they had a slight defect and eventually resulted in the left earbud not working. When I contacted JLab Audio support, they were extremely friendly and immediately shipped me a new pair (which fixed the previous issue as I noted that they reinforced the structure). Since then, I have been using this pair frequently, either keeping it on my desk for use at any time or storing it in my backpack for when I go on trips.Pros:Durable!Decent sound quality (great for the price)Looks decentGreat customer serviceCons:Just the mishap with the first pair, but I haven't encountered any issues with my current one!",
"Good sound and actually does keep out some sound (its not like BOSE noise-reduction, but for the price, one does not expect that).  Not great for working out as the buds do come out of the ear, but they nonetheless fit very comfortably.",
"I have some good ear buds that are significantly more impressive than these. but I use these more so i don't tear up my good ones. They are decent for an ear that doesn't know the difference. I am a musician and work a sound board. I know",
"for a cheap pair of headphones i'm a big fan of these. i'd definitely recommend these to anyone looking for a comfortable pair of ear buds at a good price.",
"I'd rather use the earphones that came with my iPod than use these.  It takes some real maneuvering to get them to stay in the ears.  Once they're in and secure, the sound quality is average.  I listen to different types of music and didn't hear anything great about these ear buds.",
"With 3 dogs and  a 13 year-old human child, it can get pretty loud around here.These ear buds are comfortable, block outside noise and seem well made.",
"They sounded pretty good at the beginning but it wasn't long before that changed.  The sound quality started going fast, and the cheap construction showed up in the fact that they started to literally fall apart.  I wasn't abusing them or overworking them or anythiing and I believe they should have stood up better than they did.  I wouldn't waste my money on them.  There are better buds out there.",
]
def n_containing(topics, review):
    tags = {}
    bloblist = []
    bloblist = [tb(' '.join(lemmatizer.lemmatize(word) for word in tb(phrase).lower().correct().split(" ")) + delimiter).strip() for phrase,delimiter in zip(re.split('[.,;!?:]', review), re.findall('[.,;!?:]', review))]

    for blob in bloblist:
        newBlob = blob.lower().correct();
        tags[blob] = []
        for topic in topics:
            if(topic in newBlob.words):
                tags[blob].append(topic)

    for blob, tag_list in tags.items():
        print("{}: {}".format(blob,tag_list))

    return tags

for review in reviews:
    n_containing(['quality', 'price', 'sound', 'bass', 'cord', 'bud', 'volume', 'problem', 'fit', 'hi'],review)
    print()