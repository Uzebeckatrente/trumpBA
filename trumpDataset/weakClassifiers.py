from .stats import ols
from .favs import getMedianFavCountPresTweets, getMeanFavCountPresTweets, \
	calculateAndStorenGramsFavsProbabilities, loadnGramsWithFavsProbabilities
from .part3funcs import extractNGramsFromCleanedText, getnGramsWithOverMOccurences, computeMostCommonnGrams
from .media import removeMediaFromTweet
from .basisFuncs import *
from .stats import zScore;
from .visualization import graphTwoDataSetsTogether;
from .deletedAndAllCaps import getAllCapsSkewForAboveThreshold;
import statsmodels.api as sm
from sklearn.kernel_ridge import KernelRidge


'''
this file contains several classifiers which I don't/didn't expect to perform super well:
all-caps;
OLS (50/50 or 10/90)
Poisson (rounded log count as bins)
'''



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

	def crossValidate(self, folds):
		numFolds = len(folds);
		possibleThreshholds = np.array(list(range(50))) / 67.5 + 0.26
		totalGuess = [];
		testSuccessRates = []
		for t in possibleThreshholds:
			sumSuccessRate = 0;
			for holdOut in range(numFolds):
				flatList = []
				for fold in folds[0:holdOut] + folds[holdOut + 1:numFolds]:
					flatList.extend(fold);
				self.train(flatList, trainingThreshhold=t);
				self.test(folds[holdOut], training=True, title="train data")[0];
				sumSuccessRate += self.successRate;
			totalSuccess = sumSuccessRate / len(folds);
			totalGuess.append(self.totalGuesses);
			testSuccessRates.append(totalSuccess);
			print("success for ", t, " ", totalSuccess, self.totalGuesses)


		argMaxSuccessRate = np.argmax(testSuccessRates)
		print("best success: ", possibleThreshholds[argMaxSuccessRate], " at clip: ", testSuccessRates[argMaxSuccessRate], "with ", totalGuess[argMaxSuccessRate], " guesses");

	def train(self, trainingTweets,trainingThreshhold = -1,testData = None, epochs = 50,retabulate = False):
		'''
		:param trainingTweets: [(favCount, cleanedText, allCapsRatio),...]
		:return:
		'''

		self.trainingTweets = trainingTweets
		self.medianFavCount = getMedianFavCountPresTweets(self.trainingTweets)
		self.tweetsHash = hashTweets(trainingTweets)
		self.allCapsLowerThreshhold = 0.4525925925925926
		return;


		self.trainingTweets.sort(key = lambda tup: tup[2])
		# graphTwoDataSetsTogether([t[0] for t in self.trainingTweets], "favs",[ t[2] for t in self.trainingTweets],"caps")
		#
		# exit()

		if trainingThreshhold != -1:
			possibleThreshholds = [trainingThreshhold]
		else:
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
		# print("max success rate: ",successRates[maxSuccessRateIndex]," at threshhold: ",possibleThreshholds[maxSuccessRateIndex])
		self.allCapsLowerThreshhold = possibleThreshholds[maxSuccessRateIndex]

		# graphTwoDataSetsTogether(successRates,"successRates",totalGuesses,"totalGuesses",xes = possibleThreshholds);




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
			mediaType = tweet[3]


			prediction= self.predict(allCapsRatio, self.allCapsLowerThreshhold, mediaType)
			if prediction == 0:
				continue;

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

	def predict(self, allCapsRatio, lowerThreshhold, mediaType):
		'''
		Predicting: P(C_i | cleanedText) = P(C_b)*\product p(nGram_i | C_b)
		:param cleanedText:
		:param countWeight:
		:param stdWeight:
		:param boolean:
		:return:
		'''

		if lowerThreshhold <= allCapsRatio and mediaType == "none":
			return 1;
		else:
			return 0;





class OlsTry():

	'''
	todo: how to tell OLS to focus on parameters which have a lower std?
	'''

	def createDataAndTargetMatrices(self, tweets):
		favs = [t[0] for t in tweets];
		bigMatrix = np.zeros((len(tweets), len(self.allNGrams)));
		targetMatrix = np.ndarray((len(tweets), 1))

		for tweetIndex, tweet in enumerate(tweets):
			nGramsForTweet = extractNGramsFromCleanedText(tweet[1], self.ns);
			for nGram in nGramsForTweet:
				if nGram in self.nGramIndices:
					nGramIndex = self.nGramIndices[nGram]
					bigMatrix[tweetIndex][nGramIndex] = 1.#/self.allNGramsWithCountsDict[nGram];

				# if zScore(favs,tweet[0]) > 1:
				if tweet[0]> self.ninetiethPercentileFavCount:
					targetMatrix[tweetIndex] = 10
				else:
					targetMatrix[tweetIndex] = -10;

		return bigMatrix,targetMatrix
	def train(self,trainingTweets):
		trainingTweets.sort(key=lambda tuple: tuple[0], reverse=False)
		self.ninetiethPercentileFavCount = trainingTweets[int(len(trainingTweets)*0.9)][0];

		self.trainingTweets = trainingTweets;

		self.median = getMedianFavCountPresTweets(trainingTweets)



		allNGrams = set();
		self.nGramIndices = {}
		allNGramsWithCountsDict = {}
		self.ns = [1,2,3,4]#,3,4]
		for n in self.ns:
			computeMostCommonnGrams([tweet[0:2] for tweet in trainingTweets], n);
			myNGramsWithCounts = getnGramsWithOverMOccurences(n, 2, hashTweets([tweet[0:2] for tweet in trainingTweets]))
			myNGramsWithCountsDict = {nGram[0]:nGram[1] for nGram in myNGramsWithCounts}
			myNGrams = [nGram[0] for nGram in myNGramsWithCounts]
			allNGrams.update(myNGrams)
			allNGramsWithCountsDict.update(myNGramsWithCountsDict)

		# for tweet in self.trainingTweets:
		# 	cleanedText = tweet[1]
		# 	nGrams = extractNGramsFromCleanedText(cleanedText,ns)
		# 	allNGrams.update(nGrams);
		counter = 0;
		for nGram in allNGrams:
			self.nGramIndices[nGram] = counter
			counter += 1;

		self.allNGrams = allNGrams;
		self.allNGramsWithCountsDict = allNGramsWithCountsDict;
		bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets)
		print("dims: ", bigMatrix.shape);
		startTime = time.time()
		clf = KernelRidge(alpha=1.0, kernel='linear')
		clf.fit(bigMatrix, targetMatrix)
		self.clf = clf;
		if False:
			self.weights = np.linalg.lstsq(bigMatrix, targetMatrix)[0];
		print("trained in : ",time.time()-startTime)

		# self.weights = np.dot(np.dot(np.linalg.inv(np.dot(bigMatrix, targetMatrix)),bigMatrix),targetMatrix.T)


	def test(self,testTweets):

		testTweets.sort(key=lambda tup: tup[0]);

		numCorrect = 0;

		predictionMatrix = np.zeros((len(testTweets), len(self.allNGrams)));
		print("predicting dims: ", predictionMatrix.shape);

		predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets);
		actualFavs = [t[0] for t in testTweets];

		predictions =self.clf.predict(predictionMatrix);
		for i in range(predictions.shape[0]):
			if predictions[i]*targetMatrix[i] > 0:
				numCorrect += 1;


		xes = [i for i in range(len(testTweets))];
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		fig.suptitle('Predicted FavCounts and real FavCounts (log)')
		plt.plot(xes, targetMatrix, 'go-', label='Actual Counts')
		plt.plot(xes, predictions, 'ro-', label='Predicted Counts')
		plt.legend()
		plt.show()

		print(numCorrect/len(testTweets))



class LogRegressionSlashPoisson():

	'''
	todo: how to tell OLS to focus on parameters which have a lower std?
	'''

	def createDataAndTargetMatrices(self, tweets):
		bigMatrix = np.zeros((len(tweets), len(self.allNGrams)));
		targetMatrix = np.ndarray((len(tweets), 1))

		for tweetIndex, tweet in enumerate(tweets):
			nGramsForTweet = extractNGramsFromCleanedText(tweet[1], self.ns);
			for nGram in nGramsForTweet:
				if nGram in self.nGramIndices:
					nGramIndex = self.nGramIndices[nGram]
					bigMatrix[tweetIndex][nGramIndex] = 1.#/self.allNGramsWithCountsDict[nGram];


				targetMatrix[tweetIndex] = np.log(tweet[0]);
				# if tweet[0] - self.median > 0:
				# 	targetMatrix[tweetIndex] = 1
				# else:
				# 	targetMatrix[tweetIndex] = -1;

		return bigMatrix,targetMatrix


	def trainPoisson(self, trainingTweets, reload = False):


		self.trainingTweets = trainingTweets;
		hash = hashTweets(trainingTweets);
		self.ns = [1, 2]
		if not reload:
			try:
				fileName = "trumpBA/trumpDataset/classifiers/poisson" + str(self.ns) + hash + ".p";
				classifile = open(fileName, "rb")
				self.poisson_training_results = pickle.load(classifile);

				fileNameAllNGrams = "trumpBA/trumpDataset/classifiers/allNGrams" + str(self.ns) + hash + ".p";
				allNGramsFile = open(fileNameAllNGrams, "rb")
				self.allNGrams = pickle.load(allNGramsFile);
				return;
			except:
				print("reloading anyways :)")

		self.median = getMedianFavCountPresTweets(trainingTweets)

		allNGrams = set();
		self.nGramIndices = {}
		allNGramsWithCountsDict = {}
		#  # ,3,4]
		for n in self.ns:
			computeMostCommonnGrams([tweet[0:2] for tweet in trainingTweets], n);
			myNGramsWithCounts = getnGramsWithOverMOccurences(n, 25, hashTweets([tweet[0:2] for tweet in trainingTweets]))
			myNGramsWithCountsDict = {nGram[0]: nGram[1] for nGram in myNGramsWithCounts}
			myNGrams = [nGram[0] for nGram in myNGramsWithCounts]
			allNGrams.update(myNGrams)
			allNGramsWithCountsDict.update(myNGramsWithCountsDict)

		# for tweet in self.trainingTweets:
		# 	cleanedText = tweet[1]
		# 	nGrams = extractNGramsFromCleanedText(cleanedText,ns)
		# 	allNGrams.update(nGrams);
		counter = 0;
		for nGram in allNGrams:
			self.nGramIndices[nGram] = counter
			counter += 1;

		self.allNGrams = allNGrams;
		self.allNGramsWithCountsDict = allNGramsWithCountsDict;
		bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets)
		print("calculatin",bigMatrix.shape)
		self.poisson_training_results = sm.GLM(targetMatrix,bigMatrix, family=sm.families.Poisson()).fit()

		with open("trumpBA/trumpDataset/classifiers/poisson" + str(self.ns) + hash + ".p",'wb') as classifile:
			pickle.dump(self.poisson_training_results, classifile)
			classifile.close()
		with open("trumpBA/trumpDataset/classifiers/allNGrams" + str(self.ns) + hash + ".p",'wb') as nGramFile:
			pickle.dump(self.allNGrams, nGramFile)
			nGramFile.close()
		print(self.poisson_training_results.summary())


	def testPoisson(self, testTweets):
		self.nGramIndices = {}
		counter = 0;
		for nGram in self.allNGrams:
			self.nGramIndices[nGram] = counter
			counter += 1;

		numCorrect = 0;

		predictionMatrix = np.zeros((len(testTweets), len(self.allNGrams)));
		print("predicting dims: ", predictionMatrix.shape);

		predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets);

		poisson_predictions = self.poisson_training_results.get_prediction(predictionMatrix)
		# .summary_frame() returns a pandas DataFrame
		predictions_summary_frame = poisson_predictions.summary_frame()
		poisson_predictions = poisson_predictions.predicted_mean;


		predicted_counts = predictions_summary_frame['mean']
		actual_counts = [ac[0] for ac in list(targetMatrix)]



		for i in range(poisson_predictions.shape[0]):
			prediction = poisson_predictions[i];
			target = targetMatrix[i][0];
			if (round(prediction, 0) - round(target, 0)) == 0:
				numCorrect += 1;

		print(numCorrect / len(testTweets))

		xes = [i for i in range(poisson_predictions.shape[0])]

		# Mlot the predicted counts versus the actual counts for the test data.
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		fig.suptitle('Predicted FavCounts and real FavCounts (log)')
		plt.plot(xes, poisson_predictions, 'go-', label='Predicted counts')
		plt.plot(xes, actual_counts, 'ro-', label='Actual counts')
		plt.legend()
		plt.show()

	def train(self,trainingTweets):
		self.trainingTweets = trainingTweets;

		self.median = getMedianFavCountPresTweets(trainingTweets)

		allNGrams = set();
		self.nGramIndices = {}
		allNGramsWithCountsDict = {}
		self.ns = [1,2,3,4]#,3,4]
		for n in self.ns:
			computeMostCommonnGrams([tweet[0:2] for tweet in trainingTweets], n);
			myNGramsWithCounts = getnGramsWithOverMOccurences(n, 2, hashTweets([tweet[0:2] for tweet in trainingTweets]))
			myNGramsWithCountsDict = {nGram[0]:nGram[1] for nGram in myNGramsWithCounts}
			myNGrams = [nGram[0] for nGram in myNGramsWithCounts]
			allNGrams.update(myNGrams)
			allNGramsWithCountsDict.update(myNGramsWithCountsDict)

		# for tweet in self.trainingTweets:
		# 	cleanedText = tweet[1]
		# 	nGrams = extractNGramsFromCleanedText(cleanedText,ns)
		# 	allNGrams.update(nGrams);
		counter = 0;
		for nGram in allNGrams:
			self.nGramIndices[nGram] = counter
			counter += 1;

		self.allNGrams = allNGrams;
		self.allNGramsWithCountsDict = allNGramsWithCountsDict;
		bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets)
		print("dims: ", bigMatrix.shape);
		startTime = time.time()
		clf = KernelRidge(alpha=100.0, kernel='linear')
		clf.fit(bigMatrix, targetMatrix)

		# f = open("")

		self.clf = clf;
		if False:
			self.weights = np.linalg.lstsq(bigMatrix, targetMatrix)[0];
		print("trained in : ",time.time()-startTime)

		# self.weights = np.dot(np.dot(np.linalg.inv(np.dot(bigMatrix, targetMatrix)),bigMatrix),targetMatrix.T)


	def test(self,testTweets):

		numCorrect = 0;

		predictionMatrix = np.zeros((len(testTweets), len(self.allNGrams)));
		print("predicting dims: ", predictionMatrix.shape);

		predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets);


		predictions =self.clf.predict(predictionMatrix);
		for i in range(predictions.shape[0]):
			prediction = predictions[i][0];
			target = targetMatrix[i][0];
			if (round(prediction,0)-round(target,0)) == 0:
				numCorrect += 1;

		print(numCorrect/len(testTweets))








'''
leaveout middle part
perceptron update?
CNN
MLP nochmal
tsni

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