import nltk
import numpy as np
from collections import OrderedDict


# Bag of Words
# Find least euclidean distance
class FindSimilarSentenceA:
    def __init__(self, trainData):
        self.dictionary = {}
        self.sentences = []
        self.features = []
        self.weightVec = None
        self.generateDictionaryAndSentences(trainData)
        self.generateFeaturesForSentences()

    def generateDictionaryAndSentences(self, trainData):
        self.dictionary["UNKNOWN"] = (1, 0)
        self.sentences = nltk.sent_tokenize(trainData)

        for sentence in self.sentences:
            words_token = nltk.word_tokenize(sentence)
            for word in words_token:
                if word not in self.dictionary:
                    self.dictionary[word] = [1, len(self.dictionary)]
                else:
                    self.dictionary[word][0] += 1

            self.weightVec = np.ones(len(self.dictionary))

            for word, value in self.dictionary.items():
                self.weightVec[value[1]] /= float(value[0])

    def generateFeaturesForSentences(self):
        for sentence in self.sentences:
            feature = self.getFeature(sentence)
            self.features.append(feature)

    def getFeature(self, sentence):
        feature = np.zeros(len(self.dictionary))
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word in self.dictionary:
                feature[self.dictionary[word][1]] += 1.0
            else:
                feature[self.dictionary["UNKNOWN"][1]] += 1.0
        feature /= float(len(words))
        return feature

    def getSimilarityScore(self, sentence, topK=1):
        feature = self.getFeature(sentence)

        candidates = []
        i = 0
        for targetFeature in self.features:
            similarity = 1.0/(self.getDistance(feature, targetFeature)+0.00000000001)
            candidates.append((similarity, self.sentences[i], i))
            i = i + 1

        candidates.sort(key=lambda tup: tup[0], reverse=True)

        return candidates[:topK]

    def getDistance(self, feature1, feature2):
        return np.linalg.norm((feature1 - feature2)*self.weightVec)