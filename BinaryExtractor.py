import spacy
from spacy import displacy

class BinaryExtractor:
    def __init__(self):
        self.nlp = spacy.load('en')
        self.positive = "Yes"
        self.negative = "No"

    def extractBinaryAnswer(self, question, relevantSentence):
        questionDoc = self.nlp(question)
        sentenceDoc = self.nlp(relevantSentence)

        # Spacy Dependency Parser
        negTokenQuestion = [tok for tok in questionDoc if tok.dep_ == 'neg']
        negTokenSentence = [tok for tok in sentenceDoc if tok.dep_ == 'neg']

        if len(negTokenQuestion) == 0 and len(negTokenSentence) == 0:
            return self.positive
        elif len(negTokenSentence) == 0:
            return self.negative
        elif len(negTokenQuestion) == 0:
            return self.negative
        else:
            negTokenHeadQuestion = [token.head for token in negTokenQuestion]
            negTokenHeadSentence = [token.head for token in negTokenSentence]
            sentenceNegHeads = []
            questionNegHeads = []
            for tok in negTokenHeadSentence:
                sentenceNegHeads.append(tok.lemma_)
                sentenceNegHeads.append(tok.head.lemma_)
            for tok in negTokenHeadQuestion:
                questionNegHeads.append(tok.lemma_)
                questionNegHeads.append(tok.head.lemma_)
            if len(set(questionNegHeads).intersection(sentenceNegHeads)) != 0:
                return self.positive
            else:
                return self.negative