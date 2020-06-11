from .stats import ols
from .favs import getMedianFavCountPresTweets, getMeanFavCountPresTweets, \
	calculateAndStorenGramsFavsProbabilities, loadnGramsWithFavsProbabilities
from .part3funcs import extractNGramsFromCleanedText, getnGramsWithOverMOccurences
from .media import removeMediaFromTweet
from .basisFuncs import *
from .visualization import graphTwoDataSetsTogether;
from .deletedAndAllCaps import getAllCapsSkewForAboveThreshold;

# from keras import Sequential
# from keras.layers import Dense


class AllCapsClassifier():
	def __init__(self):
		'''
		TODO: make good documentation of this!
		TODO: probability = inverse numTokens?
		TODO: allCaps, media(type)


		:param furthestDeviationWords: position i+1 represents how many i-grams that deviate furthest from mean
		:param meaningfulClassifiersFromTopBottomPercentages: position i+1 represents the number of most/least
		popular i-grams

		... will have an attached classifier
		:param filterOutliers: if set to -1, no outliers will be filtered. Otherwise, it is the number of SD's away
		from the median that will be excluded

		'''



		self.learningRate = 0.05;
		self.allCapsLowerThreshhold = 0.7;




	def displayPredictionResults(self,title="",sample = True):


		print("success rate: ",self.successRate)

		fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		ax1 = plt.subplot(1,1,1)
		# ax2 = plt.subplot(2,1,2)

		ax1.title.set_text(title + str(self.successRate))
		ax1.scatter(self.successPredictions,self.successScores , label="successes", color="red")
		ax1.scatter(self.failurePredictions,self.failureScores, label= "failures",color="blue")

		# ax2.scatter(self.numberOfGramsSuccessRates.keys(), [self.numberOfGramsSuccessRates[i][0]/self.numberOfGramsSuccessRates[i][1] for i in self.numberOfGramsSuccessRates.keys()],label="success rate for n accepted tokens")
		# ax2.scatter(self.numberOfGramsSuccessRates.keys(), [self.numberOfGramsSuccessRates[i][1] for i in self.numberOfGramsSuccessRates.keys()],label="instances of n accepted tokens")

		# ax2.plot([0,max(self.numberOfGramsSuccessRates.keys())],[self.successRate,self.successRate],label="average succ rate");

		ax1.set_xlim([0, 1])

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
		self.medianFavCount = getMedianFavCountPresTweets(self.trainingTweets)
		self.tweetsHash = hashTweets(trainingTweets)


		self.trainingTweets.sort(key = lambda tup: tup[2])
		# graphTwoDataSetsTogether([t[0] for t in self.trainingTweets], "favs",[ t[2] for t in self.trainingTweets],"caps")
		#
		# exit()

		possibleThreshholds =np.array(list(range(50)))/67.5+0.26

		successRates = []
		totalGuesses = []
		for lowerThresh in possibleThreshholds:
			self.allCapsLowerThreshhold = lowerThresh;
			misClassifieds = self.test(self.trainingTweets,training=True)
			if self.totalGuesses > 50:
				successRates.append(self.successRate)
				totalGuesses.append(self.totalGuesses)

		maxSuccessRateIndex = np.argmax(successRates);
		print("max success rate: ",successRates[maxSuccessRateIndex]," at threshhold: ",possibleThreshholds[maxSuccessRateIndex])
		self.allCapsLowerThreshhold = possibleThreshholds[maxSuccessRateIndex]

		# graphTwoDataSetsTogether(successRates,"successRates",totalGuesses,"totalGuesses");




	def test(self, testTweets,title="", training = False):


		medianFavCount = self.medianFavCount
		self.successScores = []
		self.successPredictions = []
		self.failurePredictions = []
		self.failureScores = []
		misClassifiedTweetsNaive = []

		self.numberOfGramsSuccessRates = {}

		# countWeight = False, stdWeight = False, boolean = False

		totalGuesses = 0;
		totalCorrect = 0;


		for tweet in testTweets:


			realFavCount = tweet[0]

			allCapsRatio = tweet[2]


			prediction= self.predict(allCapsRatio, self.allCapsLowerThreshhold)
			if prediction == 0:
				continue;
			print("tweet: ", tweet[1],prediction);

			success = int(prediction * (realFavCount - self.medianFavCount) > 0)
			totalCorrect += success;

			if success:
				self.successScores.append(realFavCount)
				self.successPredictions.append(allCapsRatio)
			else:
				self.failureScores.append(realFavCount);
				misClassifiedTweetsNaive.append(tweet);
				self.failurePredictions.append(allCapsRatio)


			totalGuesses += 1;
		if totalGuesses == 0:
			self.totalGuesses = 0;
			self.successRate = 0;
			return []

		self.successRate = totalCorrect/totalGuesses;
		self.totalGuesses = totalGuesses;
		if not training:
			self.displayPredictionResults(title=title)
		return [misClassifiedTweetsNaive]

	def predict(self, allCapsRatio, lowerThreshhold):
		'''
		Predicting: P(C_i | cleanedText) = P(C_b)*\product p(nGram_i | C_b)
		:param cleanedText:
		:param countWeight:
		:param stdWeight:
		:param boolean:
		:return:
		'''

		if lowerThreshhold <= allCapsRatio :
			return 1;
		else:
			return 0;



'''
leaveout middle part
perceptron update?
CNN

https://github.com/bentrevett/pytorch-sentiment-analysis part 4

Observations: all-caps no-media is almost a guaranteed over-median signal


Question: is the favourite count increase due to changing vocabulary?
How to solve: subdivide tweets into n time blocks;
for each time block:
	calculate the most popular grams
	calculate (percent) shift between time blocks
for grams with big change, were they unpopular at first or simply irrelevant?
are there anachronistic terms (maga, crooked hillary, etc.) vs terms whose popularity comes and then goes (spicer, roy moore, etc.)?

'''