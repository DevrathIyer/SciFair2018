from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Analyzer:
    analyz = SentimentIntensityAnalyzer()

    def analyze(self, sentence):
        return self.analyz.polartity_scores(sentence)
