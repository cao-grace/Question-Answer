import nltk

def read_file(filename):
  with open(filename, "r") as f:
    return f.read()

# Decide the type of the question: wh-question or a binary question
def question_type(question):
  wh_words = ['when', 'what', 'why', 'which', 'who', 'whose', 'whom', 'where', 'how']
  auxiliary_words = ['be', 'is', 'are', 'was', 'were', 'can', 'could', 'do', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'ought', 'should', 'shall', 'will', 'would']

  # Tokenize the question
  processed_question = nltk.word_tokenize(question.lower())

  for i in range(len(processed_question)):
    word = processed_question[i]

    # Special case: it is sort of wh word but it is a binary question
    if word == 'whether' or word in auxiliary_words: return 'binary'
    if word in wh_words: return word

  return 'UNKNOWN'
