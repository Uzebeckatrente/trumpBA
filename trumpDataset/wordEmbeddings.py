import fasttext
from .basisFuncs import *
from .part3funcs import unworthynessOfToken;
import spacy
nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!


def writeCleanedTotxtFile(fileName = "trumpDataset/clean.txt"):
	f = open(fileName,"w")
	mycursor.execute("select id, cleanedText from " + mainTable + " " + relevancyDateThreshhold + " and isRt = 0")
	tweets = mycursor.fetchall()
	for tweet in tweets:
		cleanText = tweet[1]
		f.write(cleanText+"\n")


def computeWordEmbeddings(cleanFile = "trumpDataset/clean.txt", embeddingsFile = "trumpDataset/cleanModel.bin"):
	writeCleanedTotxtFile()
	try:
		model = fasttext.load_model(embeddingsFile)
	except:
		model = fasttext.train_unsupervised(cleanFile, model='cbow', minCount = 1)
		for word in model.words[:20]:
			print(model[word])

		model.save_model(embeddingsFile)
	return model


def getSumOfVectorsForTweet(oneGrams, tfidfScores):
	if len(oneGrams) == 0:
		print(oneGrams);
		return np.zeros((300,));
	tokens = nlp(" ".join(oneGrams));
	tokenSum = np.zeros(tokens[0].vector.shape);
	for i in range(len(oneGrams)):
		tokenSum += tokens[i].vector*tfidfScores[i]
	tokenSum /= len(oneGrams);
	return tokenSum;




def getTopNVectorsForTweet(model, tweetText, n, vectorLen):
	vecs = []

	nlpTweetText = nlp(tweetText)

	words = [str(token) for token in nlpTweetText if not unworthynessOfToken(token)]

	wordsInDescendingFrequencyFile =open("trumpBA/trumpDataset/npStores/wordsInDescendingFrequency.p","rb")
	wordsInDescendingFrequency = pickle.load(wordsInDescendingFrequencyFile);
	wordsInDescendingFrequency = [word[0] for word in wordsInDescendingFrequency]
	topNIndices = []
	for word in words:
		if word in wordsInDescendingFrequency:
			topNIndices.append(wordsInDescendingFrequency.index(word))
		else:
			print("bust |"+word+"|")
	topNIndices.sort()
	topNIndices = topNIndices[:n]



	topNWords = [wordsInDescendingFrequency[index] for index in topNIndices]#TODO top n words for tweetText

	for word in topNWords:
		vec = model[word]
		vecs.append(vec)
	if len(vecs)<5:
		print("proppin that up what " ,len(vecs),topNWords)
	while len(vecs) < 5:
		vecs.append(np.zeros((vectorLen)))

	return vecs;






