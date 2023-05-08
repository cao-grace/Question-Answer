from stanfordcorenlp import StanfordCoreNLP

class CoreferenceResolution:
	def __init__(self):
		self.stanfordcorenlpLoaded = False
		try:
			self.nlp = StanfordCoreNLP('http://localhost', port=9000)
			self.stanfordcorenlpLoaded = True
		except:
			pass

	def getCoreferenceResolvedSentence(self, inputSentence, listContext):
		if not self.stanfordcorenlpLoaded: return inputSentence

		context = []
		for sent in listContext:
			if (sent != ""): context.insert(0, sent)

		if len(context) == 0: return inputSentence

		inputDoc = ""

		for sent in context:
			inputDoc = inputDoc + ' ' + sent

		inputDoc = inputDoc + ' ' + inputSentence
		sentCount = len(context) + 1
		corefs = None
		
		try:
			corefs = self.nlp.coref(inputDoc)
		except:
			return inputSentence

		outputSentence = ' ' + inputSentence

		for tcoref in corefs:
			fullCandidate = ""
			targetCandidate = None
			targetCandidateSpan = None

			for match in tcoref:
				sentNum = match[0]
				tCandidate = match[3]
				if (sentNum == sentCount):
					if targetCandidate is None:
						targetCandidate = tCandidate
						targetCandidateSpan = (match[1], match[2])
					elif len(tCandidate) < len(targetCandidate):
						targetCandidate = tCandidate 
						targetCandidateSpan = (match[1], match[2])
				elif (len(tCandidate)>len(fullCandidate)):
					fullCandidate = tCandidate

			if len(fullCandidate) != 0 and targetCandidate is not None and len(fullCandidate) > len(targetCandidate):
				outputSentence = outputSentence.replace(' ' + targetCandidate + ' ', ' ' + fullCandidate + ' ')

		return outputSentence[1:]
