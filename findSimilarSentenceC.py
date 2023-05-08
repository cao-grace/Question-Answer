import time
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


# Wordnet 
def penn_to_wn(tag):
    # Convert between a Penn Treebank tag to a simplified Wordnet tag
    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)

    if wn_tag is None: return None

    try: return wn.synsets(word, wn_tag)[0]
    except: return None

def sentence_similarity(sentence1, sentence2):
    # Compute the sentence similarity using Wordnet, tokenize and tag
    sentence_1 = pos_tag(word_tokenize(sentence_1))
    sentence_2 = pos_tag(word_tokenize(sentence_2))

    # Get the synsets for the tagged words
    synsets_1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence_1]
    synsets_2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence_2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets_1 if ss]
    synsets2 = [ss for ss in synsets_2 if ss]
    score, count = 0.0, 0

    # For each word in the first sentence, get the similarity value of 
    # the most similar word in the other sentence
    for synset in synsets_1:
        similarities = [synset.path_similarity(ss) for ss in synset_2]
        similarities  = [ss for ss in similarities if ss]

        # Check that the similarity could have been computed
        if(len(similarities) > 0):
            score += max(similarities )
            count += 1

    # Average the values
    if(count != 0):
        score /= count
        return score, count
    else: 
        return (0, 0)


class FindSimilarSentenceC:
    def __init__(self, trainData):
        self.sentences = nltk.sent_tokenize(trainData)

    def getSimilarityScore(self, sentence, topK=1):

        candidates = []
        i = 0
        for docuSentence in self.sentences:
            similarity1, count1 = sentence_similarity(docuSentence, sentence)
            similarity2, count2 = sentence_similarity(sentence, docuSentence)
            similarity = (similarity1 + similarity2)/2.0
            count = (count1 + count2)/2.0
            if (count > 2):
                candidates.append((similarity, docuSentence, i))
            i = i + 1

        if len(candidates)<topK:
            return None
        else:
            candidates.sort(key=lambda tup: tup[0], reverse=True)

            if topK==1:
                return candidates[0]
            else:
                return candidates[0:topK]
