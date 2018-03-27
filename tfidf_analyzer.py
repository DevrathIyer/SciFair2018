import csv
import sys
import requests
from lxml import html
import time
import matplotlib.pyplot as plt

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}

for i in range(1,len(sys.argv)):
    dataset = sys.argv[i]

    page = requests.get("https://www.amazon.com/dp/%s" % (str(dataset)), headers=headers)

    doc = html.fromstring(page.content)
    LighthouseTermsRAW = '//span[@class="cr-lighthouse-term "]//text()'
    LighthouseTerms = ''.join(doc.xpath(LighthouseTermsRAW)).strip().split()
    if(len(LighthouseTerms) == 0):
        print(page.content)
        if(str(page) == "<Response [503]>"): print("COCKBLOCKED")
        continue

    time.sleep(3)

    with open('data/models/%s_tfidf3.csv' % (dataset), 'r+') as csvfile:
        with open('data/models/formatted/%s.csv' % (dataset), 'w+',newline='') as csvoutfile:
            tfidf_reader = csv.reader(csvfile,delimiter = ' ',quotechar='|')
            fieldnames = ['TFIDF_Terms', 'Lighthouse_Terms']
            writer = csv.DictWriter(csvoutfile, fieldnames=fieldnames)
            writer.writeheader()
            for i,row in enumerate(tfidf_reader):
                if(i > 100): continue
                amzterm = ''
                if(len(LighthouseTerms) > i):
                    amzterm = LighthouseTerms[i]
                writer.writerow({'TFIDF_Terms': row[0].split(',')[0], 'Lighthouse_Terms': amzterm})

    print("Done with: {}".format(dataset))
                #analyzed.append(count/len(LighthouseTerms))

    #if(len(analyzed) > 100):
    #    print(analyzed[100])
    #plt.subplot(9,5,i)
    #plt.plot(analyzed)
#plt.show()
