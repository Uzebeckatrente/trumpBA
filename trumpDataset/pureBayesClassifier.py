from .stats import ols
from .favs import getMedianFavCountPresTweets, getMeanFavCountPresTweets, \
	calculateAndStorenGramsFavsProbabilities, loadnGramsWithFavsProbabilities
from .part3funcs import extractNGramsFromCleanedText, getnGramsWithOverMOccurences
from .media import removeMediaFromTweet
from .basisFuncs import *
from .visualization import graphTwoDataSetsTogether;

# from keras import Sequential
# from keras.layers import Dense


class PureBayesClassifier():
	def __init__(self, ns = [1,2,3,4],numBins = 2):
		'''
		TODO: make good documentation of this!
		TODO: probability = inverse numTokens?


		:param furthestDeviationWords: position i+1 represents how many i-grams that deviate furthest from mean
		:param meaningfulClassifiersFromTopBottomPercentages: position i+1 represents the number of most/least
		popular i-grams

		... will have an attached classifier
		:param filterOutliers: if set to -1, no outliers will be filtered. Otherwise, it is the number of SD's away
		from the median that will be excluded

		'''


		self.ns = ns;
		self.numBins = 2;



	def sourceNgramsForCurrentSavedSelfNsTweetsHash(self):

		self.nGramsWithBinProbabilitiesDict = loadnGramsWithFavsProbabilities(self.ns,self.numBins, self.tweetsHash)


		self.medianFavCount = getMedianFavCountPresTweets(self.trainingTweets)
		self.meanFavCount = getMeanFavCountPresTweets(self.trainingTweets)

		print("mean favs: ",self.meanFavCount," median favs: ",self.medianFavCount);

	def displayPredictionResults(self,sample = True):


		print("success rate: ",self.successRate)

		fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		ax1 = plt.subplot(1,1,1)
		# ax2 = plt.subplot(2,1,2)

		ax1.title.set_text("Naive Bayes Success Rate: " + str(self.successRate))
		ax1.scatter([tup[0] for tup in self.successScores], [tup[1] for tup in self.successScores], label="successes", color="red")
		ax1.scatter([tup[0] for tup in self.failureScores], [tup[1] for tup in self.failureScores], label= "failures",color="blue")

		# ax2.scatter(self.numberOfGramsSuccessRates.keys(), [self.numberOfGramsSuccessRates[i][0]/self.numberOfGramsSuccessRates[i][1] for i in self.numberOfGramsSuccessRates.keys()],label="success rate for n accepted tokens")
		# ax2.scatter(self.numberOfGramsSuccessRates.keys(), [self.numberOfGramsSuccessRates[i][1] for i in self.numberOfGramsSuccessRates.keys()],label="instances of n accepted tokens")


		ax1.plot([self.medianFavCount,self.medianFavCount],[-1*max([tup[1] for tup in self.successScores]),max([tup[1] for tup in self.successScores])],color="red")
		# ax2.plot([0,max(self.numberOfGramsSuccessRates.keys())],[self.successRate,self.successRate],label="average succ rate");

		ax1.set_xlim([0, 200000])

		ax1.plot([0, 200000],[0,0],color="cyan")
		ax1.legend()
		# ax2.legend()

		# ax1.set_yscale('log')
		plt.show()

		# graphTwoDataSetsTogether([self.numberOfGramsSuccessRates[i][0]/self.numberOfGramsSuccessRates[i][1] for i in self.numberOfGramsSuccessRates.keys()],"success rate for n accepted tokens", [self.numberOfGramsSuccessRates[i][1] for i in self.numberOfGramsSuccessRates.keys()],"instances of n accepted tokens")

	def selectRandomBatch(self, sourceTweets, batchSize):
		indices = np.random.permutation(len(sourceTweets))
		batchIndices = indices[:batchSize]
		batch = []
		for i in batchIndices:
			batch.append(sourceTweets[i])
		return batch;

	def train(self, trainingTweets,testData = None, epochs = 50,retabulate = False):
		'''
		:param trainingTweets: [(favCount, cleanedText, allCapsRatio),...]
		:return:
		'''
		self.trainingTweets = trainingTweets
		self.trainingTweetsFavsAndCleanedText = [tweet[0:2] for tweet in trainingTweets];
		self.tweetsHash = calculateAndStorenGramsFavsProbabilities(self.ns,self.trainingTweetsFavsAndCleanedText,self.numBins,retabulate);
		self.sourceNgramsForCurrentSavedSelfNsTweetsHash()



	def test(self, testTweets,title=""):


		meanFavCount = self.meanFavCount
		medianFavCount = self.medianFavCount
		self.successScores = []
		self.failureScores = []
		misClassifiedTweetsNaive = []

		self.numberOfGramsSuccessRates = {}

		# countWeight = False, stdWeight = False, boolean = False

		totalGuesses = 0;
		totalCorrect = 0;


		for tweet in testTweets:

			realFavCount = tweet[0]
			cleanedText = tweet[1]

			if len(cleanedText) > 2:
				prediction, confidence, numberOfAcceptedGrams = self.predict(cleanedText)
				if prediction == 404 or numberOfAcceptedGrams == 0:
					continue;

				success = int(prediction * (realFavCount - self.medianFavCount) > 0)
				totalCorrect += success;

				if success:
					self.successScores.append((realFavCount,confidence))
					try:
						self.numberOfGramsSuccessRates[numberOfAcceptedGrams][0] += 1
					except:
						self.numberOfGramsSuccessRates[numberOfAcceptedGrams] = [1,0]
				else:
					self.failureScores.append((realFavCount,confidence));
					misClassifiedTweetsNaive.append(tweet);

				try:
					self.numberOfGramsSuccessRates[numberOfAcceptedGrams][1] += 1
				except:
					self.numberOfGramsSuccessRates[numberOfAcceptedGrams] = [0, 1]


			totalGuesses += 1;

		self.successRate = totalCorrect/totalGuesses;
		self.displayPredictionResults()
		return [misClassifiedTweetsNaive]

	def predict(self, cleanedText, lowerLimitAccGram = 0, upperLimitAccGram = 100000):
		'''
		Predicting: P(C_i | cleanedText) = P(C_b)*\product p(nGram_i | C_b)
		:param cleanedText:
		:param countWeight:
		:param stdWeight:
		:param boolean:
		:return:
		'''

		nGrams = extractNGramsFromCleanedText(cleanedText, self.ns);

		meanScore = 0;
		medianScore = 0;
		numberOfAcceptedGrams = 0;
		probsClasses = [1]*self.numBins;



		for nGram in nGrams:
			if nGram in self.nGramsWithBinProbabilitiesDict:

				numberOfAcceptedGrams += 1
				myGramBinsProbs = self.nGramsWithBinProbabilitiesDict[nGram];
				for bin, prob in enumerate(myGramBinsProbs):
					probsClasses[bin] *= prob;
		# if not lowerLimitAccGram <= numberOfAcceptedGrams <= upperLimitAccGram:
		# 	return 404,404
		maxBin = np.argmax(probsClasses)
		probMaxBin = probsClasses[maxBin];
		probsClasses.remove(probMaxBin);
		if maxBin == 0:
			maxBin = -1;


		if probMaxBin > 0.01:
			print(cleanedText,probMaxBin)
		return maxBin, probMaxBin, numberOfAcceptedGrams#/(np.sum(probsClasses)/len(probsClasses))


	def regression(self):


		vocab = []
		for n in self.ns:
			myGrams = getnGramsWithOverMOccurences(n,1,self.tweetsHash)
			for grm in myGrams:
				vocab.append((grm[0]))
		print(len(vocab))

		xMatrix = np.zeros((len(vocab),len(self.cleanTexts)))
		yMatrix = np.zeros((len(self.cleanTexts)))
		for ctIndex, ct in enumerate(self.cleanTexts):
			myGrams = extractNGramsFromCleanedText(ct,self.ns);
			myVector = np.zeros((len(vocab)))
			for index,v in enumerate(vocab):
				myVector[index] = myGrams.count(v);
			yMatrix[ctIndex] = self.favs[ctIndex]
			xMatrix[:,ctIndex] = myVector

			if ctIndex%100 == 0:
				print(ctIndex/len(self.cleanTexts))

		self.m,self.c = ols(xMatrix,yMatrix)

