import spacy
import nltk
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
import BinaryExtractor
import os

class Extract_Answer:
    def __init__(self, trainData):
    self.nlp = spacy.load("en_core_web_sm")

    content = sent_tokenize(trainData.strip())
    self.trainData = content
    self.binaryAnswerExtracter = BinaryExtractor.BinaryExtractor()

    def find_previous_sentence(self, sentence):
        for i in range(len(self.trainData)):
            # using similarity to check the sentence
            s = SequenceMatcher(None, self.trainData[i], sentence)
            if (s.ratio() >= 0.7 ):
            # He should not be used on the first sentence
            # in case of really special case
                try: return self.trainData[i-1]
                except: return ""
        return ""

    def find_most_relevant(self, question_type, question, potential_answer):
        doc = self.nlp(potential_answer)
        answer = []

        if question_type == "binary":
            return [self.binaryAnswerExtracter.extractBinaryAnswer(question, potential_answer)]
        # Information finding: get as much information as possible
        for entity in doc.ents:
            if (question_type == "when" and entity.label_ in ["DATE", "TIME"]):
                answer = answer + [entity.text]
            if (question_type == "what" and entity.label_ in ["PERSON", "ORG", "FAC", "EVENT", "WORK_OF_ART"]):
                answer = answer + [entity.text]
            if (question_type == "why"):
                answer = [potential_answer]
            if (question_type == "which" and entity.label_ in ["PERSON", "ORG", "FAC", "EVENT", "WORK_OF_ART"]):
                answer = answer + [entity.text]
            if (question_type in ["whose", "who", "whom"] and entity.label_ in ["PERSON", "ORG"]):
                answer = answer + [entity.text]
            if (question_type in ["where"] and entity.label_ in ["GPE", "FAC", "LOC"]):
                answer = answer + [entity.text]
            if (question_type in ["how"] and entity.label_ in ["MONEY", "QUANTITY", "CARDINAL", "PERCENT"]):
                answer = answer + [entity.text]

        # If (A's B) appear in the text, change (A) in the potential answer list to (A's B)
        for item in answer:
            if (item + "'s") in potential_answer: tokens = potential_answer.split(" ")
            
            for i in range(len(tokens)):
                if item in tokens[i]:
                try: 
                    answer = [word.replace(item, (item + "'s " + tokens[i+1])) for word in answer]
                except:
                    pass

        # Filter 1 -- precise wording
        # Assumption -- humans don't always ask questions that contain the 
        # precise wording of the entire answer.
        for item in answer:
            if item in question: answer.remove(item)


        # Filter 2A -- "what" question (subject)
        # Assumption -- when the what question is asking for the object, 
        # verb more likely to appear in the end of the question compared to noun
        if (question_type == "what"):
            tokenized = nltk.word_tokenize(question)
            tokenized.reverse()
            for item in nltk.pos_tag(tokenized):
                if "NN" in item[1]: break
                if "VB" in item[1]:
                    index = potential_answer.find(tokenized[1])
                    if (index != -1):
                        return [potential_answer[index+1+(len(tokenized[1])):]]

        # Filter 2B -- "what" question (object)
        # Assumption -- what question is asking for the object this time, 
        # return the main object
        tokenized = nltk.word_tokenize(question)
        for item in nltk.pos_tag(tokenized):
            if "NN" in item[1]: break
            if "VB" in item[1]:
            # find the first verb
            index = potential_answer.find(tokenized[1])
            if (index != -1):
                answer = potential_answer[0:index-1]
            if "it" in answer.lower() and "its" not in answer.lower():
                return self.find_most_relevant(question_type, question, self.find_previous_sentence(potential_answer))
            else: potential_answer = answer

        
  
        # Filter 3 -- what, who, which
        # Find the verb in the question, and try to find the answer with the 
        # closest relationship (item performs verb, or verb is being performed
        # by item) with the verb. Will remove information that it thinks is 
        # incorrect + return immediately if it thinks the information is correct
        if (question_type in ["what", "who", "whom", "which"]):
            sentence_chunks = []
            doc_s = self.nlp(potential_answer)
            for chunk in doc_s.noun_chunks:
                sentence_chunks.append((chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text))

            question_chunks = []
            doc_q = self.nlp(question)
            for chunk in doc_q.noun_chunks:
                question_chunks.append((chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text))
          
            # Same relation as main word
            for chunk in question_chunks:
                if ((chunk[0]).lower() in ["what", "who"]):
                for item in sentence_chunks:
                    if item[2] == chunk[2] and item[3] == chunk[3]:
                        if item[0].lower() in ["he", "she", "it", "they", "his", "her", "their"]:
                            return self.find_most_relevant(question_type, question, self.find_previous_sentence(potential_answer))
                        else: return [item[0]]

        # Filter 4 -- why
        # Hard to answer because need to understand meaning; but there are some 
        # words which indicate a explanation relationship, and thus we can  
        # assume some information based on those words.
        if (question_type == "why"): question_tokens = nltk.word_tokenize(potential_answer)

            # Try to get in-sentence logic
            indicator = ["because", "as", "since", "therefore", "consequently", "due to", "because of"]
            for i in range(len(question_tokens)):
                if question_tokens in indicator: return [potential_answer]

            # If no obvious in-sentence logic, 
            # return entire previous sentence plus current sentence      
            return [self.find_previous_sentence(potential_answer) + " " + potential_answer]

        # Filter 5 -- how
        # Filter out how much and how many questions bc asking for quantity
        # For other how questions, return the relevant adverb
        if ("how" in question_type): 
            if (len(answer) > 0): return [answer[0]]
            else: return [(potential_answer.split(" "))[0]]

        # Coreferencing pronouns of who class questions
        if (question_type in ["whose", "who", "whom", "what"]):
            pronoun = ["he", "she", "it", "they", "his", "her", "their"]
            for word in potential_answer.split(" "):
                if word.lower() in pronoun:
                return self.find_most_relevant(question_type, question, self.find_previous_sentence(potential_answer))

        # Last resort -- return the first noun found in the sentence
        noun = ""
        tagged_sentence = (nltk.pos_tag(nltk.word_tokenize(potential_answer)))
        for word in tagged_sentence:
            if ("NN" in word[1]):
                noun = word[0]
                return [noun]

        # Unable to extract info -- just return the potential answer
        return [potential_answer]
