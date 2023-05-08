import nltk
from nltk.tokenize import sent_tokenize

# Structure Matching
class FindSimilarSentenceB:
    def __init__(self, trainData):
    self.data = sent_tokenize(trainData)

  # Generate related sentence; usually for shorter questions when possibility 
  # of the words in question will show in the sentence high
  # Extract and sort the sentence based on frequency of target word in sentence
  def generateSentences(self, question):
    sanitized_question = question.replace("?", "")
    wh_words = ['when', 'what', 'why', 'which', 'who', 'whose', 'whom', 'where', 'how']
    for word in wh_words:
        sanitized_question = sanitized_question.replace(word, "")
        sanitized_question = sanitized_question.replace(word.capitalize(), "")
    
    sanitized_question = sanitized_question.split(" ")
    while "" in sanitized_question: 
        sanitized_question.remove("")
    while "the" in sanitized_question: 
        sanitized_question.remove("the")
    
    tagged_question = (nltk.pos_tag(nltk.word_tokenize(question)))
    verb = ""

    # Find the verb
    # Assumptions: single verb, categorizing the question and closest sentence
    for word in tagged_question:
        if ("VB" in word[1]):
        verb = word[0]
        break
    
    noun = ""

    # Find the noun
    # Assumptions: single proper noun, it is the main object of the question, 
    # and the main object in the answer
    for word in tagged_question:
        if ("NN" in word[1]): noun = word[0]
        break

    potential_sentence = []
    for sentence in self.data:
        count = 0
        # Simple counting of words
        for word in sanitized_question:
            if word in sentence: count = count + 1
      
      
        # If the sentence has the structure where the main noun appears before
        # the main verb, it is highly possible to be the correct answer based on 
        # the direct object assumption
        appeared = 0
        for word in sentence.split(" "):
            if word == noun: 
                appeared = 1
            if word == verb and appeared == 1:
                count = count + 1
                break
            if word == verb and appeared == 0:
                count = count - 0.5
                break

    tokenized_question = nltk.word_tokenize(question)
    tokenized_sentence = nltk.word_tokenize(sentence)
    for i in range(len(tokenized_question)):
    try:
        if tokenized_sentence[(tokenized_sentence.index(tokenized_question[i]))+1] == tokenized_question[i+1]:
        count = count + 1
    except:
        pass

    if (count != 0): potential_sentence.append((sentence, count))
    result = (sorted(list(potential_sentence), key = lambda x: x[1], reverse=True)) 

    return result  