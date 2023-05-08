import nltk

# Fail safe in case answer extracion fails.
def getMostProbableAnswer(question, relevantSentence, dictionary):
    qWords = nltk.word_tokenize(question)
    rsWords = nltk.word_tokenize(relevantSentence)

    answer = None
    freq = -1

    for rsW in rsWords:
        if rsW not in qWords and rsW[0].isupper():
            if freq == -1 or dictionary[rsW][0] < freq:
                ans = rsW
                freq = dictionary[rsW][0]

    if answer is None: return relevantSentence
    else: return answer