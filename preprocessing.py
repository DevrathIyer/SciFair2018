from textblob import TextBlob

with open('data/hardDrive_topics_in.txt','r+') as fi:
    with open('data/hardDrive2_topics_in.txt','a') as fo:
        line = fi.readline()
        while((line) != ""):
            blob = TextBlob(line)
            line = ""
            for item in blob.noun_phrases:
                line += item + " "
            fo.write(line + "\n")
            line = fi.readline()
