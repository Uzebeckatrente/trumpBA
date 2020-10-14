from matplotlib import patches

from .basisFuncs import *
from .part3funcs import normalize_headline, extractNGramsFromCleanedText, computeMostCommonnGrams, getnGramsWithOverMOccurences,computeTFIDFMatrix;
from .visualization import graphTwoDataSetsTogether;
from .stats import ols;
from .favs import getMedianFavCountPresTweets,loadnGramsWithFavsMinusMeanMedian,calculateAndStorenGramsWithFavsMinusMeanMedian
from sklearn.linear_model import *
import seaborn as sn
from sklearn.svm import SVC as svc
from sklearn.model_selection import cross_val_score;
from .daysOfTheWeek import determineDayOfWeek,determineSegmentOfDay,determineYearOfTweet
from sklearn.neural_network import MLPClassifier, MLPRegressor;
import sklearn.linear_model
from .wordEmbeddings import getSumOfVectorsForTweet, computeWordEmbeddingsDict, getSumOfGloveVectorsForTweet
from sklearn.metrics import plot_confusion_matrix,confusion_matrix
from sklearn.metrics import classification_report,r2_score,mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
XX, yy = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(XX, yy,random_state=1)

from keras import backend as K, regularizers, optimizers

from keras import Sequential
from keras.layers import Dense
from scipy.stats import zscore

def precisionAttenuatesWithBoxSize(classifiers,classifierNames, boxSizes, training, test):
	scores = []
	for numBoxes in boxSizes:
		for classifierIndex, classifier in enumerate(classifiers):
			if classifierIndex == 0:
				classifier.train(training, percentageNGrams=0.3, includeWordEmbeddingMatrix=False, alpha=20, transformation="log", numBoxes=numBoxes)
			elif classifierIndex == 1:
				classifier.train(training,percentageNGrams = 0.35, includeWordEmbeddingMatrix = False, alpha = 1,transformation="log", numIterationsMax = 200,numBoxes=numBoxes)
			elif classifierIndex == 2:
				classifier.train(training, percentageNGrams=0.35, includeWordEmbeddingMatrix=False, alpha=10, transformation="log", numBoxes=numBoxes)
			score = classifier.test(test, title="All Years Test Data", crossVal=True, targetNames=["box " + str(i) for i in range(numBoxes)]);
			try:
				scores[classifierIndex].append(score)
			except:
				scores.append([score])


	for classifierIndex in range(len(classifiers)):
		c = randomHexColor()
		xes = [i for i in range(len(scores[classifierIndex]))]
		plt.scatter(xes,scores[classifierIndex],label=classifierNames[classifierIndex],c=c)
		fitLineCoefficients = np.polyfit(xes, scores[classifierIndex], 1, full=False)
		slope = fitLineCoefficients[0];
		intercept = fitLineCoefficients[1]
		plt.ylabel("Precision of Classification", fontsize=15)
		plt.legend()
		# plt.xlabel("Time")
		plt.plot(xes, [0+ slope * x + intercept for x in xes],color=c)
	xes = [i for i in range(len(scores[0]))]
	plt.xticks(xes,[str(boxSize) for boxSize in boxSizes],fontsize=10);
	plt.xlabel("Number of Bins")
	plt.title("Precision Attenuates as Box Size Increases",fontsize=20);
	plt.show()


def precisionAttenuatesWithBoxWidth(classifiers,classifierNames, boxSizes, training, test):
	scores = []
	for boxSize in boxSizes:
		for classifierIndex, classifier in enumerate(classifiers):
			if classifierIndex == 0:
				classifier.train(training,percentageNGrams=0.25,includeWordEmbeddingMatrix=True,alpha=0.001,transformation="poisson",boxSize=boxSize);
			elif classifierIndex == 1:
				classifier.train(training,percentageNGrams=0.5,includeWordEmbeddingMatrix=True,alpha=0.0001,transformation="poisson",boxSize=boxSize);
			score = classifier.test(test, title="All Years Test Data", crossVal=True, targetNames=["box " + str(i) for i in range(boxSize)]);
			try:
				scores[classifierIndex].append(score)
			except:
				scores.append([score])


	for classifierIndex in range(len(classifiers)):
		c = randomHexColor()
		xes = [i for i in range(len(scores[classifierIndex]))]
		plt.scatter(xes,scores[classifierIndex],label=classifierNames[classifierIndex],c=c)
		fitLineCoefficients = np.polyfit(xes, scores[classifierIndex], 1, full=False)
		slope = fitLineCoefficients[0];
		intercept = fitLineCoefficients[1]
		plt.ylabel("Precision of Classification", fontsize=15)
		plt.legend()
		# plt.xlabel("Time")
		plt.plot(xes, [0+ slope * x + intercept for x in xes],color=c)
	xes = [i for i in range(len(scores[0]))]
	plt.xticks(xes,[str(boxSize) for boxSize in boxSizes],fontsize=10);
	plt.xlabel("Box Sizes")
	plt.title("Precision Attenuates as Box Size Decreases",fontsize=20);
	plt.show()

def graphConfusionMatrixContinuousMultipleInputs(testTweets, predictionsEs, targetMatrix, dataSetNames, acceptableRange=0.3, doGraph=False):
	if doGraph:
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)

	markers = ["v","P","x","o"];

	favs = [t[0] for t in testTweets];
	favs.sort()
	favsPercentiles = []
	custom_lines = []
	labels = []
	numInRange = 0;
	for dataSetIndex in range(len(dataSetNames)):
		print("di: ",dataSetIndex)
		predictions = predictionsEs[dataSetIndex]
		custom_lines.append(Line2D([0], [0], lw=4,linestyle="none",markersize=10, alpha=0.6,marker=markers[dataSetIndex], color='#000000'))
		labels.append(dataSetNames[dataSetIndex])
		for tweetIndex in range(len(testTweets)):
			print("ti: ",tweetIndex);
			target = targetMatrix[tweetIndex];
			prediction = predictions[tweetIndex];
			if acceptableRange <= 1:
				acceptableMin = (target * (1 - acceptableRange))
				acceptableMax = (target * (1 + acceptableRange))
			else:
				acceptableMin = (target - acceptableRange)
				acceptableMax = (target + acceptableRange)
			inAcceptableRange = acceptableMin <= prediction <= acceptableMax

			numInRange += int(inAcceptableRange)
			if doGraph:
				if inAcceptableRange:
					plt.scatter([tweetIndex], [prediction], c='#00ff00', marker=markers[dataSetIndex],s=16)
				else:
					plt.scatter([tweetIndex], [prediction], c='#ff0000', marker=markers[dataSetIndex],s=16)
				if dataSetIndex == 0:
					acceptableRect = patches.Rectangle((tweetIndex, acceptableMin), 1, acceptableMax - acceptableMin, linewidth=0.5, alpha=0.32, color="#aabbcc")
					plt.gca().add_patch(acceptableRect)

	xes = [i for i in range(predictionsEs[0].size)];

	# plt.scatter(xes, predictions, c='r', label='Predicted Counts')
	if doGraph:
		plt.plot(xes, targetMatrix, 'go-', markersize=0.1, c='#0000ff', label='Actual Counts')
		plt.ylim(min(targetMatrix) - 1, max(targetMatrix) + 1);
		# custom_lines.append(Line2D([0], [0], lw=4, alpha=0.32, color="#aabbcc"))
		labels.extend(["In Range","Out of Range","Actual Favourite Counts"])
		custom_lines.extend([Line2D([0], [0], lw=4, alpha=0.6, color='#00ff00'),Line2D([0], [0], lw=4, alpha=0.6, color='#ff0000'),Line2D([0], [0], lw=4, alpha=0.6, color='#0000ff')])
		plt.legend(custom_lines, labels,prop={'size': 12});
		plt.xticks([])
		plt.ylabel("Favourite Counts",fontsize=18)
		plt.yticks(fontsize=15)
		plt.title('Pure Regression of Each Model on Test Data',size=20)

		plt.show()
	return numInRange / len(testTweets)

def fourModelComparisonRegression(trainingLastYear,testLastYear,allPresTweetsFavCountAndCleanedTextByYear5):
	testLastYear.sort(key = lambda tup: tup[0])
	olsRegressor = RegressionModel("ols")
	mlpRegressor = RegressionModel("mlpRegressor")
	mlPoisson = RegressionModel("mlPoisson");
	poisson = RegressionModel("poisson");

	olsRegressor.train(trainingLastYear, percentageNGrams=0, includeWordEmbeddingMatrix=False, alpha=10)
	mlpRegressor.train(trainingLastYear, percentageNGrams=0.25, includeWordEmbeddingMatrix=False, alpha=100000, numIterationsMax=500);
	poisson.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=False,alpha=1000);
	mlPoisson.train(trainingLastYear, percentageNGrams=0.5, includeWordEmbeddingMatrix=False, alpha=100);



	predictionMatrixOLS, targetMatrix = olsRegressor.createDataAndTargetMatrices(testLastYear,training=False)
	predictionsOls = olsRegressor.model.predict(predictionMatrixOLS);

	predictionMatrixMLP, _ = mlpRegressor.createDataAndTargetMatrices(testLastYear,training=False)
	predictionsMLP = mlpRegressor.model.predict(predictionMatrixMLP);

	predictionMatrixPoisson, _ = poisson.createDataAndTargetMatrices(testLastYear,training=False)
	predictionsPoisson = poisson.model.predict(predictionMatrixPoisson);

	predictionMatrixMLPoisson, _ = mlPoisson.createDataAndTargetMatrices(testLastYear,training=False)
	predictionsMLPoisson = mlPoisson.model.predict(predictionMatrixMLPoisson);

	predictionsEs = [predictionsOls,predictionsMLP,predictionsPoisson,predictionsMLPoisson];
	datasetNames = ["Ridge Regression","Multilayer Perceptron","Poisson Regression","Poisson MLP"]
	#
	# predictionsEs = [predictionsOls];
	# datasetNames = ["Ridge Regression"];

	graphConfusionMatrixContinuousMultipleInputs(testLastYear, predictionsEs, targetMatrix, datasetNames, acceptableRange=0.3, doGraph=True);







class RegressionModel():

	'''
	todo: how to tell OLS to focus on parameters which have a lower std?
	'''
	def __init__(self,learner,task="class"):
		print(Fore.RED,"Initializing "+str(learner)+"!\n",Style.RESET_ALL)

		self.learner = learner;
		self.task =task;
		self.ns = [1, 2, 3, 4]  # ,3,4]

	def computeWordEmbeddingMatrix(self, trainingTweets):
		'''
		TODO: fix to proper word embeddings
		:param trainingTweets:
		:return:
		'''
		tweetsHash = hashTweets(trainingTweets)
		try:

			return np.load("trumpBA/trumpDataset/npStores/embeddingMatrix/wordEmbeddingMatrix" + tweetsHash + ".npy")
		except:
			pass;
		print(Fore.MAGENTA, "computing word embedding matrix", Fore.RESET);

		wordEmbeddingsDict = computeWordEmbeddingsDict()




		wordEmbeddingMatrix = np.zeros((len(trainingTweets), 25));

		for i in range(len(trainingTweets)):
			sentenceEmbedding = getSumOfGloveVectorsForTweet(trainingTweets[i][1], wordEmbeddingsDict);
			wordEmbeddingMatrix[i] = sentenceEmbedding;
			if i % 100 == 0:
				print(i / len(trainingTweets));
		np.save("trumpBA/trumpDataset/npStores/embeddingMatrix/wordEmbeddingMatrix" + tweetsHash + ".npy", wordEmbeddingMatrix)
		return wordEmbeddingMatrix

	def computeWordEmbeddingForTweet(self,tweet, wordEmbeddingsDict):
		sentenceEmbedding = getSumOfGloveVectorsForTweet(tweet[1], wordEmbeddingsDict);
		return sentenceEmbedding

	def computeWordEmbeddingMatrixDepricated(self, trainingTweets):
		'''
		TODO: fix to proper word embeddings
		:param trainingTweets:
		:return:
		'''
		tweetsHash = hashTweets(trainingTweets)
		try:return np.load("trumpBA/trumpDataset/npStores/embeddingMatrix/wordEmbeddingMatrix"+tweetsHash+".npy")
		except:pass;
		print(Fore.MAGENTA,"computing word embedding matrix",Fore.RESET);
		
		cleanedTexts = [t[1] for t in trainingTweets];
		tfidfMatrix, vocab = computeTFIDFMatrix(cleanedTexts)
		vocabDict = {};
		for index, v in enumerate(vocab):
			vocabDict[v] = index;


		wordEmbeddingMatrix = np.zeros((len(trainingTweets),300));
		for i in range(len(trainingTweets)):
			tweetTime = time.time();
			tweet = trainingTweets[i];

			nGramsForTweet = extractNGramsFromCleanedText(tweet[1], self.ns);

			tfidfRow = tfidfMatrix[i];
			tfidfScores = [];
			tfidfNGrams = [];
			for nGram in nGramsForTweet:
				if nGram in vocabDict:
					vocabIndex = vocabDict[nGram];
					tfidfScores.append(tfidfRow[vocabIndex]);
					tfidfNGrams.append(nGram);
			extractTime = time.time();
			wordEmbeddingForTweet = getSumOfVectorsForTweet(tfidfNGrams,tfidfScores);
			extractTime = extractTime - time.time();
			wordEmbeddingMatrix[i] = wordEmbeddingForTweet;
			tweetTime = tweetTime-time.time();
			# print(extractTime/tweetTime)
			if i%100 == 0:
				print(i/len(trainingTweets));
		np.save("trumpBA/trumpDataset/npStores/embeddingMatrix/wordEmbeddingMatrix"+tweetsHash+".npy", wordEmbeddingMatrix)
		return wordEmbeddingMatrix

	def createDataAndTargetMatrices(self, tweets,training = True, randomness= False):
		'''
		•n-grams (m-hot)
		•all-caps percentage
		•length
		•time of day
		•day of week
		•sentence embedding
		:param tweets:
		:param extraParameters:
		:param randomness:
		:return:
		'''







		#####create data matrices
		targetMatrix = np.ndarray((len(tweets), 1))
		sizeOfBigMatrix = 0;
		sizeOfBigMatrix += len(self.allNGrams)
		if self.includeWordEmbeddingMatrix:
			print("hiya!")
			myWordEmbeddingMatrix = self.computeWordEmbeddingMatrix(tweets);
			sizeOfBigMatrix += myWordEmbeddingMatrix.shape[1];
			# wordEmbeddingsDict = computeWordEmbeddingsDict()
		for toInclude in self.extraParamDict.values():
			if toInclude:
				sizeOfBigMatrix += 1;
		bigMatrix = np.zeros((len(tweets),sizeOfBigMatrix));

		years = 4;
		times = [t[4] for t in tweets];
		minTime = min(times);
		maxTime = max(times);


		logs = [np.log(np.log(tweet[0])) if np.log(tweet[0]) > 0 else 0 for tweet in tweets];
		zscores = [round(zs,2) for zs in zscore(logs)];

		poissonFavs = [int(round(t[0]/self.boxSize,0)) for t in tweets];


		if training:
			minZscore = np.min(zscores)
			maxZscore = np.max(zscores)
			self.starts = [round(i * (maxZscore - minZscore) / (self.numBoxes) + minZscore, 2) for i in range(self.numBoxes + 1)]

		favs = [t[0] for t in tweets];
		maxFav = np.max(favs);
		minFav = np.min(favs);


		# bigMatrix = np.zeros((len(tweets), len(self.allNGrams)+len(extraParameters)+(wordEmbeddingMatrix.shape[1] if self.wordEmbeddingMatrix else 0)));
		#####\create data matrices


		#####fill bigMatrix
		offSet = len(self.nGramIndices);
		for tweetIndex, tweet in enumerate(tweets):
			nGramsForTweet = extractNGramsFromCleanedText(tweet[1], self.ns);
			allCapsForTweet = [t.strip() for t in tweet[5].lower().split(" ") if len(t.strip()) > 0];

			#### fill m-hot vector
			for nGram in nGramsForTweet:
				if nGram in self.nGramIndices:
					nGramIndex = self.nGramIndices[nGram]
					if nGram in allCapsForTweet:
						bigMatrix[tweetIndex][nGramIndex] = 1+self.allCapsBoost;
					else:
						bigMatrix[tweetIndex][nGramIndex] = 1.#/self.allNGramsWithCountsDict[nGram];


			#### fill wordEmbeddings
			startingIndexForNextEntry = len(self.allNGrams);
			if self.includeWordEmbeddingMatrix:
				# sentenceEmbeddings = self.computeWordEmbeddingForTweet(tweet, wordEmbeddingsDict);
				sentenceEmbedding = myWordEmbeddingMatrix[tweetIndex]
				bigMatrix[tweetIndex][startingIndexForNextEntry:startingIndexForNextEntry+myWordEmbeddingMatrix.shape[1]] = sentenceEmbedding
				startingIndexForNextEntry += myWordEmbeddingMatrix.shape[1]

			#### fill extra params
			if self.extraParamDict["allCapsPercentage"]:
				bigMatrix[tweetIndex][startingIndexForNextEntry] = tweet[2]
				startingIndexForNextEntry += 1;
			if self.extraParamDict["dayOfWeek"]:
				bigMatrix[tweetIndex][startingIndexForNextEntry] = determineDayOfWeek(tweet[4])
				startingIndexForNextEntry += 1;
			if self.extraParamDict["timeOfDay"]:
				bigMatrix[tweetIndex][startingIndexForNextEntry] = determineSegmentOfDay(tweet[4])
				startingIndexForNextEntry += 1;
			if self.extraParamDict["yearOfTweeting"]:
				bigMatrix[tweetIndex][startingIndexForNextEntry] = determineYearOfTweet(minTime, maxTime, years=years, t=tweet[4])
				startingIndexForNextEntry += 1;
			if self.extraParamDict["length"]:
				numRealTokens = len(tweet[6].split(" "))
				length = -1;
				if numRealTokens <= 2:
					length = 0;
				elif numRealTokens <= 4:
					length = 1;
				elif numRealTokens <= 8:
					length = 2;
				elif numRealTokens <= 14:
					length = 3;
				else:
					length = 4;
				bigMatrix[tweetIndex][startingIndexForNextEntry] = length




			#####\fill m-hot vector



		#####fill target
			if self.task == "class":
				if self.transformation == "log":
					myZscore = zscores[tweetIndex];
					if self.numBoxes != -1:
						try:

							# starts = [(((maxZscore - minZscore)) * i / self.numBoxes - (maxZscore - minZscore)/2) for i in range(self.numBoxes + 1)];
							myBox = 0;
							if myZscore < self.starts[0]:
								pass;
							elif myZscore > self.starts[-1]:
								myBox = len(self.starts-1);
							else:
								while not self.starts[myBox] <= myZscore <= self.starts[myBox + 1]:
									myBox += 1;
							targetMatrix[tweetIndex] = myBox
						except:
							print(self.starts,myZscore)
							raise(Exception("n00b"))
				elif self.transformation == "poisson":

					targetMatrix[tweetIndex] = poissonFavs[tweetIndex]

				else:
					#binary classification!
					classI = 0;
					while tweet[0] >= self.twoClassBarrierCounts[classI+1]:
						classI += 1;
					print("classI: ",classI)
					targetMatrix[tweetIndex] = classI;
			else:
				#regression!
				#todo: richte boxes ein
				if self.transformation == "log":
					myZscore = zscores[tweetIndex];
					if self.numBoxes != -1:
						starts = [((maxZscore - minZscore)) * i / self.numBoxes for i in range(self.numBoxes + 1)];
						myBox = 0;

						while not starts[myBox] <= myZscore <= starts[myBox + 1]:
							myBox += 1;
						targetMatrix[tweetIndex] = myBox

					else:
						targetMatrix[tweetIndex] = myZscore
				elif self.transformation == "countReduce":
					myCountReduce = tweet[0]/50000;

					if self.numBoxes != -1:
						starts = [((maxFav - minFav)/50000) * i / self.numBoxes for i in range(self.numBoxes + 1)];
						myBox = 0;

						while not starts[myBox] <= myCountReduce <= starts[myBox + 1]:
							myBox += 1;
						targetMatrix[tweetIndex] = myBox
					else:
						targetMatrix[tweetIndex] = myCountReduce
				elif self.transformation == "custom":
					raise Exception("Not implemented lmao")

				else:
					#no transformation
					# targetMatrix[tweetIndex] = 17;
					targetMatrix[tweetIndex] = tweet[0]




		return bigMatrix,targetMatrix



	def getExtraParams(self,tweets, extraParamDict):
		self.years = 5;
		times = [t[4] for t in tweets];
		self.minTime = min(times);
		self.maxTime = max(times);
		allCapsRatios = [t[2] for t in tweets];
		daysOfWeek = [determineDayOfWeek(t[4]) for t in tweets]
		timesOfDay = [determineSegmentOfDay(t[4]) for t in tweets]

		years = [determineYearOfTweet(self.minTime, self.maxTime, years=self.years, t=t[4]) for t in tweets]
		extraParams = [allCapsRatios,daysOfWeek,timesOfDay,years]
		extraParamsToInclude = [];
		self.extraParamNamesToInclude = []
		for i in range(len(extraParams)):
			if extraParamNum % 2 == 0:
				extraParamsToInclude.append(extraParams[i]);
				self.extraParamNamesToInclude.append(extraParamNames[i]);
			extraParamNum = int(extraParamNum / 2);

		return extraParamsToInclude

	def poissonLoss(self,yTrue,lam):
		err = 0;

		'''
		if arrays, but apparently not here...
		'''

		err = lam-yTrue*K.log(lam)

		return err;

	def poissonActivation(self,x):
		# return x;
		return K.exp(x);

	def mlpRegressorActivation(self,x):
		# return x;
		return K.exp(x);

	def normalizeClassSizes(self, tweets, distancePercentileFromSplit=-1):
		favs = [t[0] for t in tweets];
		favs.sort();
		# indexOfPercentileCount = favs.index(self.twoClassBarrierCount);
		# if distancePercentileFromSplit != -1:
		# 	bottomIndex = indexOfPercentileCount - int(indexOfPercentileCount * distancePercentileFromSplit);
		# 	topIndex = indexOfPercentileCount + int(indexOfPercentileCount * (1 - distancePercentileFromSplit))
		# 	# trainingTweets = trainingTweets[:bottomIndex] + trainingTweets[topIndex:];

		theseClassElements = []
		for i in range(len(self.twoClassBarrierCounts)-1):
			theseElements = [t for t in tweets if self.twoClassBarrierCounts[i] <= t[0] < self.twoClassBarrierCounts[i+1]]
			theseClassElements.append(theseElements);


		minClassSize = np.min([len(c) for c in theseClassElements]);
		classTrainingIDX = [np.random.choice(len(c), minClassSize, replace=False) for c in theseClassElements]
		theseClassElementsTaken = []
		for classNumber, theseClassesTakenIndices in enumerate(classTrainingIDX):
			theseClassElementsTaken.append([theseClassElements[classNumber][i] for i in theseClassesTakenIndices])

		trainingTweets = []
		for classElements in theseClassElementsTaken:
			trainingTweets += classElements

		# print(Fore.MAGENTA, "ratio of positives to negatives in learning sample: ", Fore.RESET)
		# trainingTweets = myNegs + thesePositives;
		return trainingTweets;

	def train(self, trainingTweets, extraParamDict="default", hiddenLayers = (3,), allCapsBoost=0.5, alpha=0.1, percentileNegativePositive=1, percentileSplits = (0.5,), includeWordEmbeddingMatrix=True, twoClassBarrierCount=-1, distancePercentileFromSplit = 0.4, percentageNGrams = 0.5, transformation = None, numBoxes = 5, numIterationsMax = 500, learningRate = 1,boxSize = 100000):
		'''

		:param trainingTweets: tweets in format ["favCount","cleanedText, allCapsRatio, mediaType, publishTime","allCapsWords"]
		:param extraParamDict: Dict with keys {"allCapsRatios","daysOfWeek","timesOfDay","years","length"}; values boolean
		:param hiddenLayers: tuple of hidden layer widths for model if MLP-based
		:param allCapsBoost: extra weight for words that are spelled all-caps
		:param alpha: regularization parameter
		:param percentileNegativePositive: ratio of positive to negative tweets in training set, for classification
		:param percentileSplits: fulcrum for 2-class classification
		:param includeWordEmbeddingMatrix: boolean, if we are using word embeddings in our trainer (later becomes a matrix)
		:param twoClassBarrierCount: private parameter, normally set to -1, otherwise to indicate the count used for the fulcrum
		:param distancePercentileFromSplit: not implemented, would indicate the training tweets which we ignore that are close to the fulcrum
		:param percentageNGrams: 0 to 0.5, if 0 no n-grams are used, if 0.5 all are used, if in between only the most/least popular n-grams are used
		:param transformation: None, "log","countReduce". if log, favourite counts are log'd and rounded to create basically normal distribution; if countReduce, they are divided by 100000 or so and rounded
		:param numBoxes: int, decides the number of boxes for hybrid regression.
		:return:
		'''
		trainingTweets.sort(key=lambda tuple: tuple[0], reverse=False)
		'''
		init
		'''
		if self.learner in ["poisson","ols","mlPoisson","mlpRegressor"]:
			self.task = "reg"
		else:
			self.task = "class";
		if self.task == "class":
			if alpha == None:
				alpha = 4;
			if allCapsBoost == None:
				allCapsBoost = 0.5
		if self.learner == "poisson" or self.learner == "poissonClassifier":
			self.model = PoissonRegressor(alpha=alpha)
		elif self.learner == "ols" or self.learner == "olsClassifier":
			self.model = Ridge(alpha=alpha);
		elif self.learner == "svm":
			self.model = svc(C=alpha,verbose=True)
		elif self.learner == "mlpClassifier":
			self.model = MLPClassifier(hidden_layer_sizes=hiddenLayers,alpha=alpha,max_iter=numIterationsMax,verbose=True,);
		elif self.learner == "mlpRegressor":
			self.model = MLPRegressor(alpha=alpha,max_iter = numIterationsMax,hidden_layer_sizes=hiddenLayers,verbose=True,learning_rate_init=learningRate,learning_rate="adaptive");
			print("nalpha: ",alpha)

			# self.model = MLPRegressor(random_state=1,hidden_layer_sizes=(200), max_iter=10000);
			# self.model = Sequential();
		elif self.learner == "mlPoisson" or self.learner == "mlPoissonClassifier":
			self.model = Sequential();
		else:
			alpha = 0.5;


		if extraParamDict == "default":
			extraParamDict={"allCapsPercentage": True, "timeOfDay": True, "dayOfWeek": True,"yearOfTweeting":True, "length": True}
		for key, value in extraParamDict.items():
			if type(value) != bool or key not in ["allCapsPercentage","timeOfDay", "dayOfWeek","yearOfTweeting","length"]:
				raise Exception("extraParamDict formatting is wrong!")






		self.includeWordEmbeddingMatrix = includeWordEmbeddingMatrix;
		self.transformation = transformation;
		self.percentileNegativePositive = percentileNegativePositive;
		self.percentileSplit = percentileSplits;
		if twoClassBarrierCount == -1:
			self.twoClassBarrierCounts = [0]+[trainingTweets[int(len(trainingTweets) * self.percentileSplit[i])][0] for i in range(len(self.percentileSplit))]+[10000000];

		self.extraParamDict = extraParamDict;
		self.allCapsBoost = allCapsBoost;
		self.numBoxes = numBoxes;
		self.boxSize = boxSize;





		if self.task == "class" and self.percentileNegativePositive != -1:

			trainingTweets = self.normalizeClassSizes(trainingTweets, -1);
			print(trainingTweets);

		allNGrams = set();
		self.nGramIndices = {}
		allNGramsWithCountsDict = {}
		allNGramsWithSkewsDict = {}


		calculateAndStorenGramsWithFavsMinusMeanMedian(self.ns,[tweet[0:2] for tweet in trainingTweets],minCount=2,retabulate=False)
		nGramsWithFavsMinusMean, nGramsWithFavsMinusMedian = loadnGramsWithFavsMinusMeanMedian(self.ns, hashTweets([tweet[0:2] for tweet in trainingTweets]))
		nGramsWithFavsMinusMean.sort(key = lambda tup: tup[nGramStorageIndicesSkew["skew"]]);
		nGramsWithFavsMinusMedian.sort(key=lambda tup: tup[nGramStorageIndicesSkew["skew"]]);
		percentageBound = int(len(nGramsWithFavsMinusMedian)*percentageNGrams);
		if percentageNGrams == 0:
			acceptableNGrams = [];
		else:
			acceptableNGrams = [tup[nGramStorageIndicesSkew["nGram"]] for tup in nGramsWithFavsMinusMedian[:percentageBound]+nGramsWithFavsMinusMedian[-percentageBound:]]

		# acceptableNGrams = [];
		for n in self.ns:
			computeMostCommonnGrams([tweet[0:2] for tweet in trainingTweets], n,retabulate=True);
			# if self.learner == "poisson":
			# 	myNGramsWithCounts = getnGramsWithOverMOccurences(n, 2, hashTweets([tweet[0:2] for tweet in trainingTweets]))
			# else:# self.learner == "ols":
			myNGramsWithCounts = getnGramsWithOverMOccurences(n, 2, hashTweets([tweet[0:2] for tweet in trainingTweets]))
			rejects = [m for m in myNGramsWithCounts if m[0] not in acceptableNGrams];
			myNGramsWithCounts = [m for m in myNGramsWithCounts if m[0] in acceptableNGrams];
			print("numRej: ",len(rejects)," numTaken: ",len(myNGramsWithCounts));

			myNGramsWithCountsDict = {nGram[0]:nGram[1] for nGram in myNGramsWithCounts}
			myNGrams = [nGram[0] for nGram in myNGramsWithCounts]
			allNGrams.update(myNGrams)
			allNGramsWithCountsDict.update(myNGramsWithCountsDict)


		counter = 0;
		for nGram in allNGrams:
			self.nGramIndices[nGram] = counter
			counter += 1;

		self.allNGrams = allNGrams;
		self.allNGramsWithCountsDict = allNGramsWithCountsDict;

		# extraParams = self.getExtraParams(trainingTweets, extraParamNum);





		# bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets,[[np.log(t[0]) for t in trainingTweets],[(t[0]/100000) for t in trainingTweets]]);

		bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets,training=True);

		print("dims: ", bigMatrix.shape);
		print(targetMatrix)



		startTime = time.time()

		if self.learner == "mlPoisson" or self.learner == "mlPoissonClassifier":
			# bigMatrix = np.ones((1000,1))
			self.model.add(Dense(hiddenLayers[0], input_dim=bigMatrix.shape[1], activation='sigmoid', use_bias=True,kernel_regularizer=regularizers.l2(alpha)))
			for h in range(1,len(hiddenLayers)):
				self.model.add(Dense(hiddenLayers[h], activation='tanh', use_bias=True,kernel_regularizer=regularizers.l2(alpha)))
			# self.model.add(Dense(1, activation=self.poissonActivation, use_bias=True,kernel_regularizer=regularizers.l2(alpha)))
			self.model.add(Dense(1, use_bias=True,activation=self.poissonActivation,kernel_regularizer=regularizers.l2(alpha)))
			# self.model.add(Dense(1, use_bias=True, activation=self.poissonActivation,kernel_regularizer=regularizers.l2(alpha)))
			self.model.compile(optimizer="adam", metrics=['accuracy'], loss=self.poissonLoss);
			# self.model.compile(optimizer=optimizers.Adam(lr=0.01),metrics=['accuracy']);

			self.model.fit(bigMatrix, targetMatrix, epochs=numIterationsMax, batch_size=100)
		elif self.learner == "mlpRegressor" and False:
			self.model.add(Dense(hiddenLayers[0], input_dim=bigMatrix.shape[1], activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(alpha)))
			for h in range(1, len(hiddenLayers)):
				self.model.add(Dense(hiddenLayers[h], activation='tanh', use_bias=True, kernel_regularizer=regularizers.l2(alpha)))
			self.model.add(Dense(1, use_bias=True,activation=self.mlpRegressorActivation, kernel_regularizer=regularizers.l2(alpha)))
			self.model.compile(optimizer="adam", metrics=['accuracy']);
			self.model.fit(bigMatrix, targetMatrix, epochs=numIterationsMax, batch_size=100)
		else:
			np.sum(bigMatrix);
			np.sum(targetMatrix);
			self.model.fit(bigMatrix, targetMatrix);

		print(Fore.MAGENTA,"fitted in ",time.time()-startTime,Style.RESET_ALL);

		return list([key for key in self.extraParamDict.keys() if extraParamDict[key]])
		# return ", ".join(self.extraParamNamesToInclude);

	def graphConfusionMatrixDiscrete(self, testTweets, predictions, targetMatrix, score, title, percentiles = (0, 0.1, 0.9, 1)):
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		fig.suptitle('Predicted FavCounts and real FavCounts for ' + title + " correlation coef: " + str(round(score, 2)))

		favs = [t[0] for t in testTweets];
		favs.sort()
		favsPercentiles = []
		for percentileIndex in range(len(percentiles)-1):
			medianFavCountForPercentile = favs[int(len(favs)*(percentiles[percentileIndex]+percentiles[percentileIndex+1])*0.5)]
			lowestFavCountInRange = favs[int(len(favs)*(percentiles[percentileIndex]))]
			highestFavCountInRange = favs[int(len(favs) * (percentiles[percentileIndex+1]))-1]
			favsPercentiles.append(medianFavCountForPercentile)
			leftCount = int(len(favs)*percentiles[percentileIndex])
			rightCount = int(len(favs) * percentiles[percentileIndex+1])

			relevantPredictions = list(predictions[leftCount:rightCount])
			accuratePredictions = [t for t in relevantPredictions if lowestFavCountInRange <= t <= highestFavCountInRange];
			accurateIndices = [leftCount+relevantPredictions.index(t) for t in accuratePredictions]
			inaccuratePredictions = [t for t in relevantPredictions if not lowestFavCountInRange <= t <= highestFavCountInRange]
			inaccurateIndices = [leftCount + relevantPredictions.index(t) for t in inaccuratePredictions]
			plt.scatter(accurateIndices, accuratePredictions, c='#00ff00',marker="+", label='Predicted Counts (Accurate)')
			plt.scatter(inaccurateIndices, inaccuratePredictions, c='#ff0000',marker="x", label='Predicted Counts (Inaccurate)')


			plt.plot([leftCount,rightCount], [medianFavCountForPercentile,medianFavCountForPercentile],label="median for percentile " + str(percentileIndex))
			rect = patches.Rectangle((leftCount, lowestFavCountInRange),(rightCount-leftCount), highestFavCountInRange-lowestFavCountInRange, linewidth=5, edgecolor='r', alpha=0.32)
			plt.gca().add_patch(rect)

			rx, ry = rect.get_xy()
			cx = rx + rect.get_width() / 2.0
			cy = ry + rect.get_height() * 0.85
			patchAnnot = "Percentile ("+str(percentiles[percentileIndex])+","+str(percentiles[percentileIndex+1])+")"
			plt.gca().annotate(patchAnnot, (cx, cy), color='w', weight='bold',fontsize=6, ha='center', va='center')

		xes = [i for i in range(predictions.size)];

		# plt.scatter(xes, predictions, c='r', label='Predicted Counts')
		plt.plot(xes, targetMatrix, 'go-', markersize=0.1,c='#0000ff', label='Actual Counts')
		plt.ylim(min(targetMatrix) - 1, max(targetMatrix) + 1);
		plt.legend()
		# print(numCorrect / len(testTweets))
		plt.show()


	def graphConfusionMatrixContinuous(self, testTweets, predictions, targetMatrix, score, title, acceptableRange = 0.3, doGraph = False):
		if doGraph:
			fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
			fig.subplots_adjust(left=0.06, right=0.94)


		favs = [t[0] for t in testTweets];
		favs.sort()
		favsPercentiles = []
		numInRange = 0;
		for tweetIndex in range(len(testTweets)):
			target = targetMatrix[tweetIndex];
			prediction = predictions[tweetIndex];
			if acceptableRange <= 1:
				acceptableMin = (target*(1-acceptableRange))
				acceptableMax = (target*(1+acceptableRange))
			else:
				acceptableMin = (target - acceptableRange)
				acceptableMax = (target + acceptableRange)
			inAcceptableRange =  acceptableMin<= prediction <= acceptableMax

			numInRange += int(inAcceptableRange)
			if doGraph:
				if inAcceptableRange:
					plt.scatter([tweetIndex], [prediction], c='#00ff00',marker="+")
				else:
					plt.scatter([tweetIndex], [prediction], c='#ff0000',marker="x",)

				acceptableRect = patches.Rectangle((tweetIndex, acceptableMin), 1, acceptableMax - acceptableMin, linewidth=0.5, edgecolor='r', alpha=0.32, color="#aabbcc")
				plt.gca().add_patch(acceptableRect)


		xes = [i for i in range(predictions.size)];

		# plt.scatter(xes, predictions, c='r', label='Predicted Counts')
		if doGraph:
			plt.plot(xes, targetMatrix, 'go-', markersize=0.1,c='#0000ff', label='Actual Counts')
			plt.ylim(min(targetMatrix) - 1, max(targetMatrix) + 1);
			custom_lines = [Line2D([0], [0], color="#00ff00", lw=4),
							Line2D([0], [0], color="#ff0000", lw=4),
							Line2D([0],[0],lw=4, alpha=0.32,color="#aabbcc")]
			plt.legend(custom_lines, ["within tolerance", "not within tolerance","acceptance range"]);
			fig.suptitle('Predicted FavCounts and real FavCounts for ' + title + " correlation coef: " + str(round(score, 2)) + "; percentage within range: "+str(round(100*numInRange/len(testTweets),2))+"%")

			plt.show()
		return numInRange/len(testTweets)


	def graphClassification(self, predictionMatrix, predictions, targetMatrix, dataset,targetNames, doGraph = False):

		if doGraph:
			fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
			fig.subplots_adjust(left=0.06, right=0.94)




		maxClass = int(max(round(np.max(targetMatrix),0),round(np.max(predictions),0)))
		print("maximus: ", np.max(targetMatrix), np.max(predictions))
		minClass = int(min(np.min(targetMatrix), np.min(predictions)))

		correctPredictionsAggregator = [0 for _ in range(maxClass+1)]
		predictionsAggregator = [0 for _ in range(maxClass+1)]

		shouldHaveBeenFirstIndexActuallyWasSecondIndex = []
		for i in range(maxClass+1):
			shouldHaveBeenFirstIndexActuallyWasSecondIndex.append([]);
			for _ in range(maxClass+1):
				shouldHaveBeenFirstIndexActuallyWasSecondIndex[i].append(0);
		print("uniqueness: ",np.unique(targetMatrix))




		for i in range(predictions.shape[0]):
			prediction = int(round(float(predictions[i]),0));
			target = int(targetMatrix[i][0]);
			try:
				shouldHaveBeenFirstIndexActuallyWasSecondIndex[target][prediction] += 1;
			except:
				print(target, prediction,shouldHaveBeenFirstIndexActuallyWasSecondIndex)
				raise(Exception("n00b"))
			prod = prediction * target;

			if np.isclose(prediction,target):
				correctPredictionsAggregator[target] += 1;
				if doGraph:
					plt.scatter(i, prediction, c="#00ff00");

			else:
				if doGraph:
					plt.scatter(i,prediction,c="#ff0000");

			predictionsAggregator[target] += 1;
		accuracy = np.mean([correctPredictionsAggregator[i]/predictionsAggregator[i] for i in range(len(correctPredictionsAggregator)) if predictionsAggregator[i] > 0])
		# accuracy = 0.5*(correctPredictionsAggregator[0]/predictionsAggregator[0])+0.5*(correctPredictionsAggregator[1]/predictionsAggregator[1]);
		if doGraph:
			title = "Classification on " + dataset + " using " + self.learner + "; Classification Accuracy " + str(round(accuracy,2))
			plt.title(title)
			plt.show()

			N = len(correctPredictionsAggregator)
			ind = np.arange(N)  # the x locations for the groups
			width = 1/(len(correctPredictionsAggregator)+1)  # the width of the bars

			fig = plt.figure()
			ax = fig.add_subplot(111)

			bars = [];
			barLabels = []
			indexLabels = []
			for shouldHaveBeenThisClass in range(len(shouldHaveBeenFirstIndexActuallyWasSecondIndex)):
				actualyWasThisClass = shouldHaveBeenFirstIndexActuallyWasSecondIndex[shouldHaveBeenThisClass];
				bars.append(ax.bar(ind+width*shouldHaveBeenThisClass+width*0.5,actualyWasThisClass,width))
				barLabels.append("Classifier Guessed "+str(shouldHaveBeenThisClass))
				indexLabels.append("Target: "+str(shouldHaveBeenThisClass))
			ax.bar(ind+width*len(shouldHaveBeenFirstIndexActuallyWasSecondIndex)*0.5,[np.sum(x) for x in shouldHaveBeenFirstIndexActuallyWasSecondIndex],width*len(shouldHaveBeenFirstIndexActuallyWasSecondIndex),alpha = 0.5,label="Actual Occurence Count for Each Class")

			ax.set_ylabel('Target/Prediction Count for Each Class')
			ax.set_xticks(ind + width*len(shouldHaveBeenFirstIndexActuallyWasSecondIndex)*0.5)
			ax.set_xticklabels(indexLabels)
			ax.legend(bars,barLabels)

			# def autolabel(rects):
			# 	for rect in rects:
			# 		h = rect.get_height()
			# 		ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
			# 				ha='center', va='bottom')
			#
			# autolabel(rects1)
			# autolabel(rects2)
			# autolabel(rects3)

			plt.show()



			data = confusion_matrix(predictions, targetMatrix)
			for i in range(len(predictions)):
				print(predictions[i],targetMatrix[i]);
			df_cm = pd.DataFrame(data, columns=targetNames, index=targetNames)
			df_cm.index.name = 'Actual'
			df_cm.columns.name = 'Predicted'
			plt.figure(figsize=(10, 7))
			sn.set(font_scale=1.4)  # for label size
			sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16},fmt='d')
			plt.title(title)
			plt.show()

		return accuracy



	def test(self,testTweets,title="",distancePercentileFromSplit = 0, targetNames = ("0-30","70-100"),crossVal = False):

		testTweets.sort(key=lambda tup: tup[0]);


		favs = [t[0] for t in testTweets];

		# if self.percentileNegativePositive != -1:
		# 	indexOfPercentileCount = favs.index(min(favs, key=lambda x:abs(x-self.twoClassBarrierCount)));
		# 	bottomIndex = indexOfPercentileCount - int(indexOfPercentileCount * distancePercentileFromSplit);
		# 	topIndex = indexOfPercentileCount + int(distancePercentileFromSplit*len(testTweets))-indexOfPercentileCount
		# 	print("prev: ",len(testTweets));
		# 	testTweets = testTweets[:bottomIndex] + testTweets[topIndex:];
		# 	print("post: ", len(testTweets));
		numCorrect = 0;



		# extraParams = self.getExtraParams(testTweets,self.extraParamNum);


		# predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets,[[np.log(t[0]) for t in testTweets],[(t[0]/100000) for t in testTweets]]);

		predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets,training=False)

		print("targetMaxAndMin ",min(targetMatrix),max(targetMatrix));
		actualFavs = [t[0] for t in testTweets];

		predictions = self.model.predict(predictionMatrix);
		print("predictions: ",predictions);
		# plt.plot([i for i in range(len(predictions))],predictions);
		# plt.show();

		if self.task == "class":


			classificationAccuracy = self.graphClassification(predictionMatrix,predictions,targetMatrix,title,targetNames=targetNames,doGraph=not crossVal)
			print(Fore.MAGENTA, "classification report: \n", Fore.RESET);
			print("classificationAccuracy: ", classificationAccuracy)
			# print(classification_report(targetMatrix, predictions.round(), target_names=targetNames));
			print("and that was the classification report. Good night!")

			return classificationAccuracy
		else:
			correlationCoefficient = r2_score(targetMatrix,predictions);
			print("rex regis")
			percentWithinRange = self.graphConfusionMatrixContinuous(testTweets,predictions,targetMatrix,correlationCoefficient,title,doGraph=not crossVal,acceptableRange=0.3)
			# percentWithinRange = self.graphConfusionMatrixDiscrete(testTweets, predictions, targetMatrix, correlationCoefficient, title)
			return correlationCoefficient, percentWithinRange

		# return "new: ",score, "old way: ",scoreOldWay;


	def crossValNumHiddenLayers(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		# self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		hiddenLayerPoss = [1,3,5,7,9,20,30,40,50,100];
		index = 0;
		for numHoldoutFold in range(numFolds):
			trainFlat, holdOut = flattenFolds(folds, numHoldoutFold)
			for numHiddenLayers in hiddenLayerPoss:
				for numHiddenLayers2 in [0]:#[0]+hiddenLayerPoss:
					if numHiddenLayers2 == 0:
						self.train(trainFlat,hiddenLayers=(numHiddenLayers,));
					else:
						self.train(trainFlat, hiddenLayers=(numHiddenLayers,numHiddenLayers2));
					score = self.test(holdOut,crossVal=True);
					try:
						scoresDict[str((numHiddenLayers,numHiddenLayers2))].append(score);
					except:
						scoresDict[str((numHiddenLayers,numHiddenLayers2))] = [score];
					index += 1;

		for combo in scoresDict.keys():
			scoresDict[combo] = np.mean(scoresDict[combo]);
			scores.append(scoresDict[combo])

		bestScore = 0;
		bestCombo = "";
		for combo in scoresDict.keys():
			if scoresDict[combo] > bestScore:
				bestScore = scoresDict[combo];
				bestCombo = combo;
		print("best combo: ",bestCombo," with success: ",bestScore);

		scores = np.array(scores);
		print(scores)
		print(resultDict);
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



	def crossValRegularisatzia(self,tweets,alphaPossibilities = None):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		# self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		if alphaPossibilities == None:
			alphaPossibilities = [5,10,15,20,30]#,40,50,100];
		index = 0;
		for alpha in alphaPossibilities:
			scoresDict[alpha] = []
			for numHoldoutFold in range(numFolds):
				train = folds[0:numHoldoutFold] + folds[numHoldoutFold + 1:];
				holdOut = folds[numHoldoutFold];
				trainFlat = [item for sublist in train for item in sublist]



				self.train(trainFlat,alpha=alpha);
				score = self.test(holdOut,crossVal=True);
				scoresDict[alpha].append(score);
				index += 1;

				score = np.mean(scoresDict[alpha]);
				print(Fore.YELLOW, "done with ", index / (len(alphaPossibilities) * (len(alphaPossibilities) + 1)), str(alpha),score, Style.RESET_ALL);
				scores.append(score);
				resultDict[alpha] = score;

		bestScore = 0;
		bestCombo = "";
		for combo in resultDict.keys():
			if resultDict[combo] > bestScore:
				bestScore = resultDict[combo];
				bestCombo = combo;
		print("best combo: ",bestCombo," with success: ",bestScore);

		scores = np.array(scores);
		print(scores)
		print(resultDict);
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




	def crossValDistancePercentilFromSplitTesting(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		alphaPossibilities = [0.001,0.01,0.1,0.5]#,0.5,1,10]#,5,7,20,30,40,50,100];
		index = 0;

		for numHoldoutFold in range(numFolds):
			train = folds[0:numHoldoutFold] + folds[numHoldoutFold + 1:];
			holdOut = folds[numHoldoutFold];
			trainFlat = [item for sublist in train for item in sublist]

			training = trainFlat;
			self.train(training,crossVal=False);

			for distancePercentileFromSplit in [i/20 for i in range(20)]:
				score = self.test(holdOut,crossVal=True,distancePercentileFromSplit=distancePercentileFromSplit);
				try:
					scoresDict[distancePercentileFromSplit].append(score);
				except:
					scoresDict[distancePercentileFromSplit]=[score]
				index += 1;

		for distancePercentileFromSplit in [i / 20 for i in range(20)]:
			score = np.mean(scoresDict[distancePercentileFromSplit]);
			scores.append(score);
			resultDict[distancePercentileFromSplit] = score;

		bestScore = 0;
		bestCombo = "";
		for combo in resultDict.keys():
			if resultDict[combo] > bestScore:
				bestScore = resultDict[combo];
				bestCombo = combo;
		print("best combo: ",bestCombo," with success: ",bestScore);

		scores = np.array(scores);
		print(scores)
		print(resultDict);
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	def crossValAllCapsBoostTargetMatrix(self,tweets):
		print("doing xval for capsBoost :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		index = 0;
		allCapsBonuses = [0.2,0.5,1,1.3,1]#
		for numHoldoutFold in range(numFolds):
			train = folds[0:numHoldoutFold] + folds[numHoldoutFold + 1:];
			holdOut = folds[numHoldoutFold];
			trainFlat = [item for sublist in train for item in sublist]
			training = trainFlat;
			for allCapsBonus in allCapsBonuses:
				self.train(training, allCapsBoost=allCapsBonus,crossVal=False);
				score = self.test(holdOut,crossVal=True);
				try:
					scoresDict[allCapsBonus].append(score);
				except:
					scoresDict[allCapsBonus]=[score]
				index += 1;

		for allCapsBonus in allCapsBonuses:
			score = np.mean(scoresDict[allCapsBonus]);
			scores.append(score);
			resultDict[allCapsBonus] = score;

		bestScore = 0;
		bestCombo = "";
		for combo in resultDict.keys():
			if resultDict[combo] > bestScore:
				bestScore = resultDict[combo];
				bestCombo = combo;
		print("best combo: ",bestCombo," with success: ",bestScore);

		scores = np.array(scores);
		print(scores)
		print(resultDict);
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


	def crossValPercentileOfTweets(self, tweets):
		print("Breaking percentile xval doing xval :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		percentilePossibilities = [0.1,0.15,0.2,0.25,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]#,0.5,1,3,5,7]#,50,100,500,1000]#,3,5,7,9]#,20,30,40,50,100];
		index = 0;
		for percentile in percentilePossibilities:
			scoresDict[percentile] = []
			for numHoldoutFold in range(numFolds):
				train = folds[0:numHoldoutFold] + folds[numHoldoutFold + 1:];
				trainFlat = [item for sublist in train for item in sublist]
				holdOut = folds[numHoldoutFold];
				trainFlat.sort(key=lambda t: t[0]);
				theseNegatives = [t for t in trainFlat if t[0] < self.twoClassBarrierCount]
				thesePositives = [t for t in trainFlat if t[0] >= self.twoClassBarrierCount]


				negativeTrainingIDX = np.random.choice(len(theseNegatives), len(thesePositives), replace=False);
				myNegs = []
				for i in range(len(theseNegatives)):
					if i in negativeTrainingIDX:
						myNegs.append(theseNegatives[i])
				lennus = ((len(myNegs), len(thesePositives)));
				print("schnitt: ", [x for x in myNegs if x in thesePositives]);
				training = myNegs + thesePositives;



				self.train(training,percentilePassed=percentile,crossVal=True);
				score = self.test(holdOut,crossVal=True);
				scoresDict[percentile].append(score);
				index += 1;

				score = np.mean(scoresDict[percentile]);
				print(Fore.YELLOW, "done with ", index / (len(percentilePossibilities) * (len(percentilePossibilities) + 1)), str(percentile),score, Style.RESET_ALL);
				scores.append(score);
				resultDict[percentile] = score;

		bestScore = 0;
		bestCombo = "";
		for combo in resultDict.keys():
			if resultDict[combo] > bestScore:
				bestScore = resultDict[combo];
				bestCombo = combo;
		print("best combo: ",bestCombo," with success: ",bestScore);

		scores = np.array(scores);
		print(scores)
		print(resultDict);
		plt.plot(percentilePossibilities,scores)
		plt.title("breaking percentile scores")
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	def crossValTweetRatioInTraining(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key=lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		scores = []
		scoresDict = {}
		maxNeg = 0;
		for i in range(numFolds):
			trainNotFlat = folds[:i] + folds[i + 1:];
			trainFlat = [item for sublist in trainNotFlat for item in sublist]
			theseNegatives = [t for t in trainFlat if t[0] < self.twoClassBarrierCount]
			maxNeg = max(len(theseNegatives),maxNeg);


		positives = [t for t in tweets if t[0] >= self.twoClassBarrierCount]
		negatives = [t for t in tweets if t[0] < self.twoClassBarrierCount]
		lennus = (len(positives),len(negatives));
		resultDict = {}
		possNumNegs = list(range(int(len(positives)/2),maxNeg,int((len(negatives)-len(positives))/10)));
		possNumNegs.append(len(positives));
		possNumNegs.sort();
		for numNegs in possNumNegs:
			scoresDict[numNegs] = [];
			for i in range(numFolds):
				trainNotFlat = folds[:i]+folds[i+1:];
				trainFlat = [item for sublist in trainNotFlat for item in sublist]
				trainFlat.sort(key = lambda t: t[0]);
				theseNegatives = [t for t in trainFlat if t[0] < self.twoClassBarrierCount]
				thesePositives = [t for t in trainFlat if t[0] >= self.twoClassBarrierCount]

				holdout = folds[i];
				negativeTrainingIDX = np.random.choice(len(theseNegatives),numNegs,replace=False);
				myNegs = []
				for i in range(len(theseNegatives)):
					if i in negativeTrainingIDX:
						myNegs.append(theseNegatives[i])

				lennus = ((len(myNegs),len(thesePositives)));
				training = myNegs+thesePositives;

				self.train(training,crossVal=True);
				score = self.test(holdout,crossVal=True);
				scoresDict[numNegs].append(score);
			score = np.mean(scoresDict[numNegs]);
			resultDict[numNegs] = score;
			scores.append(score);
			print("numNegs: ",numNegs," (against approx",len(positives),") score: ",score);



		scores = np.array(scores);
		print(scores)
		print(resultDict);
		plt.plot(possNumNegs,scores);
		plt.show();
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


	def crossValDistanceFromTwoClassBarrierCount(self, tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key=lambda t: t[0]);
		favs = [t[0] for t in tweets];
		scoresDict = {}
		resultDict = {}
		scores = [];
		distancesFromClassBarrier = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		indexOfPercentileCount = favs.index(self.twoClassBarrierCount)
		for distance in distancesFromClassBarrier:
			indexOfPercentileCount = favs.index(min(favs, key=lambda x: abs(x - self.twoClassBarrierCount)));
			bottomIndex = indexOfPercentileCount - int(indexOfPercentileCount * distance);
			topIndex = indexOfPercentileCount + int(distance * len(tweets)) - indexOfPercentileCount
			print("prev: ", len(tweets));
			theseTweets = tweets[:bottomIndex] + tweets[topIndex:];
			train, test, folds = splitTrainingTest(theseTweets, numFolds);
			scoresDict[distance] = [];
			for i in range(numFolds):
				trainNotFlat = folds[:i] + folds[i + 1:];
				trainFlat = [item for sublist in trainNotFlat for item in sublist]
				trainFlat.sort(key=lambda t: t[0]);

				holdout = folds[i];
				self.train(trainFlat, crossVal=False);
				score = self.test(holdout, crossVal=True);
				scoresDict[distance].append(score);
			score = np.mean(scoresDict[distance]);
			resultDict[distance] = score;
			scores.append(score);
			print("percentile: ", distance, " score: ", score);



		scores = np.array(scores);
		print(scores)
		print(resultDict);
		plt.plot(distancesFromClassBarrier,scores);
		plt.show();
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


	def crossValWhichExtraParams(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key=lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		scores = []
		paramNamesInOrder = []
		scoresDict = {}
		maxNeg = 0;
		for i in range(numFolds):
			trainNotFlat = folds[:i] + folds[i + 1:];
			trainFlat = [item for sublist in trainNotFlat for item in sublist]
			theseNegatives = [t for t in trainFlat if t[0] < self.twoClassBarrierCount]
			maxNeg = max(len(theseNegatives),maxNeg);


		positives = [t for t in tweets if t[0] >= self.twoClassBarrierCount]
		negatives = [t for t in tweets if t[0] < self.twoClassBarrierCount]
		lennus = (len(positives),len(negatives));
		resultDict = {}


		for paramNum in range(2**len(extraParamNames)):
			paranNames = None;
			for i in range(numFolds):
				trainNotFlat = folds[:i]+folds[i+1:];
				trainFlat = [item for sublist in trainNotFlat for item in sublist]
				trainFlat.sort(key = lambda t: t[0]);
				theseNegatives = [t for t in trainFlat if t[0] < self.twoClassBarrierCount]
				thesePositives = [t for t in trainFlat if t[0] >= self.twoClassBarrierCount]
				numNegs = len(thesePositives);
				holdout = folds[i];
				negativeTrainingIDX = np.random.choice(len(theseNegatives),numNegs,replace=False);
				myNegs = []
				for i in range(len(theseNegatives)):
					if i in negativeTrainingIDX:
						myNegs.append(theseNegatives[i])

				lennus = ((len(myNegs),len(thesePositives)));
				training = myNegs+thesePositives;

				paramNames = self.train(training,crossVal=True,extraParamNum=paramNum);
				score = self.test(holdout,crossVal=True);
				try:
					scoresDict[paramNames].append(score);
				except:
					scoresDict[paramNames] = [score];
			score = np.mean(scoresDict[paramNames]);
			resultDict[paramNames] = score;
			scores.append(score);
			paramNamesInOrder.append(paramNames);
			print("paramNames: ",paramNames," score: ",score, " done with: ",paramNum/2**len(extraParamNames));

		bestScore = 0;
		bestCombo = "";
		for key in resultDict.keys():
			if resultDict[key] > bestScore:
				bestScore = resultDict[key];
				bestCombo=key;
		print("best combo: ",bestCombo," with score: ",bestScore);
		scores = np.array(scores);
		print(scores)
		print(resultDict);
		plt.plot(list(range(2**len(extraParamNames))),scores);
		plt.xticks(list(range(2**len(extraParamNames))),paramNamesInOrder)
		plt.show();
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	def crossValPercentageOfGramsToTake(self,tweets,alphasPoss= None):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key=lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scoresCorrelationCoefficient = []
		scoresPercentageWithinRange = []
		scoresDictCorrelationCoefficient = {}
		scoresDictPercentageWithinRange = {}

		resultsDictCorrelationCoefficient = {}
		resultsDictPercentageWithinRange = {}

		percentagesPoss=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
		if alphasPoss == None:
			alphasPoss = [0.1,1,10,50,100]
		for index,percentage in enumerate(percentagesPoss):
			for alphaIndex, alpha in enumerate(alphasPoss):
				scoresDictCorrelationCoefficient[(percentage,alpha)] = [];
				scoresDictPercentageWithinRange[(percentage, alpha)] = [];
				for i in range(numFolds):
					trainNotFlat = folds[:i]+folds[i+1:];
					trainFlat = [item for sublist in trainNotFlat for item in sublist]
					holdout = folds[i];



					self.train(trainFlat,percentageNGrams=percentage,alpha=alpha,numIterationsMax=1500,includeWordEmbeddingMatrix=False,percentileSplits=(0.9,));
					if self.task == "reg":
						correlationCoefficient, percentWithinRange = self.test(holdout,crossVal=True);
						scoresDictCorrelationCoefficient[(percentage, alpha)].append(correlationCoefficient);
						scoresDictPercentageWithinRange[(percentage, alpha)].append(percentWithinRange)
					else:
						accuracy = self.test(holdout, crossVal=True);
						scoresDictCorrelationCoefficient[(percentage, alpha)].append(accuracy);


				if self.task == "reg":
					correlationCoefficient = np.mean(scoresDictCorrelationCoefficient[(percentage,alpha)])
					percentWithinRange = np.mean(scoresDictPercentageWithinRange[(percentage,alpha)])
					resultsDictCorrelationCoefficient[(percentage, alpha)] = correlationCoefficient;
					resultsDictPercentageWithinRange[(percentage, alpha)] = percentWithinRange
					scoresCorrelationCoefficient.append(correlationCoefficient);
					scoresPercentageWithinRange.append(percentWithinRange);
				else:
					accuracy = np.mean(scoresDictCorrelationCoefficient[(percentage, alpha)])
					resultsDictCorrelationCoefficient[(percentage, alpha)] = accuracy;
					scoresCorrelationCoefficient.append(accuracy);
				if self.task == "reg":
					print(Fore.YELLOW,"percentage: ",percentage," alpha: ",alpha," (correlation coefficient, percent within range): ",correlationCoefficient, percentWithinRange, " done with: ",index/len(percentagesPoss));
				else:
					print(Fore.YELLOW, "percentage: ", percentage, " alpha: ", alpha, " accuracy: ", accuracy, " done with: ", index / len(percentagesPoss));

		bestScoreCorrelation = 0;
		bestComboCorrelation = "";
		if self.task == "reg":
			bestScorePercentWithinRange = 0;
			bestComboPercentWithinRange = "";
			for key in resultsDictPercentageWithinRange.keys():
				if resultsDictPercentageWithinRange[key] > bestScorePercentWithinRange:
					bestScorePercentWithinRange = resultsDictPercentageWithinRange[key];
					bestComboPercentWithinRange = key;
			print("best combo: ", bestComboPercentWithinRange, " with score: ", bestScorePercentWithinRange);
			scoresPercentageWithinRange = np.array(scoresPercentageWithinRange);
			print(scoresPercentageWithinRange)
			resultDictPercentageWithinRangeSorted = list(resultsDictPercentageWithinRange.items())
			resultDictPercentageWithinRangeSorted.sort(key=lambda tup: tup[1])
			print("percentage win range ", resultDictPercentageWithinRangeSorted)
			plt.plot([i for i in range(len(scoresPercentageWithinRange))], scoresPercentageWithinRange, label="percentage within range");

		for key in resultsDictCorrelationCoefficient.keys():
			if resultsDictCorrelationCoefficient[key] > bestScoreCorrelation:
				bestScoreCorrelation = resultsDictCorrelationCoefficient[key];
				bestComboCorrelation = key;
		print("best combo: ", bestComboCorrelation, " with score: ", bestScoreCorrelation);



		scoresCorrelationCoefficient = np.array(scoresCorrelationCoefficient);

		print(scoresCorrelationCoefficient)
		resultDictCorrelationCoefficientSorted = list(resultsDictCorrelationCoefficient.items())
		resultDictCorrelationCoefficientSorted.sort(key = lambda tup: tup[1])
		if self.task == "reg":
			print("correlation coefficietns: ",resultDictCorrelationCoefficientSorted)
			plt.plot([i for i in range(len(scoresCorrelationCoefficient))], scoresCorrelationCoefficient, label="correlation coefficient");

		else:
			print("classification accuracies: ", resultDictCorrelationCoefficientSorted)
			plt.plot([i for i in range(len(scoresCorrelationCoefficient))], scoresCorrelationCoefficient, label="Classification Accuracies");

		plt.legend()
		plt.show();
		# print("Accuracy: %0.2f (+/- %0.2f)" % (scoresCorrelationCoefficient.mean(), scoresCorrelationCoefficient.std() * 2))


	def crossValEverything(self,tweets):

		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key=lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets, numFolds);
		scores = []
		scoresDict = {}

		resultDict = {}

		percentagesPoss = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
		alphasPoss = [1, 10, 50, 100,500,1000]
		for index, percentage in enumerate(percentagesPoss):
			for alpha in alphasPoss:
				for includeSentenceEmbeddings in [True,False]:
					for hiddenLayerPoss in [1,3,5,7,9,20,30,40,50,100]:
						for numIterationsMax in [100,200,400,600]:
							scoresDict[(percentage, alpha,includeSentenceEmbeddings,hiddenLayerPoss,numIterationsMax)] = [];
							for i in range(numFolds):
								trainNotFlat = folds[:i] + folds[i + 1:];
								trainFlat = [item for sublist in trainNotFlat for item in sublist]
								holdout = folds[i];

								self.train(trainFlat, percentageNGrams=percentage, alpha=alpha,includeWordEmbeddingMatrix=includeSentenceEmbeddings,hiddenLayers=(hiddenLayerPoss,),numIterationsMax=numIterationsMax);
								score = self.test(holdout, crossVal=True);

								scoresDict[(percentage, alpha,includeSentenceEmbeddings,hiddenLayerPoss,numIterationsMax)].append(score);

							score = np.mean(scoresDict[(percentage, alpha,includeSentenceEmbeddings,hiddenLayerPoss,numIterationsMax)]);
							resultDict[(percentage, alpha,includeSentenceEmbeddings,hiddenLayerPoss,numIterationsMax)] = score;
							scores.append(score);
							print(Fore.YELLOW, "percentage: ", percentage, " alpha: ", alpha, " score: ", score, " done with: ", index / len(percentagesPoss));

		bestScore = 0;
		bestCombo = "";
		for key in resultDict.keys():
			if resultDict[key] > bestScore:
				bestScore = resultDict[key];
				bestCombo = key;
		print("best combo: ", bestCombo, " with score: ", bestScore);
		scores = np.array(scores);
		print(scores)
		print(resultDict);
		plt.plot([i for i in range(len(scores))], scores);
		plt.show();
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





'''
XVAL trials:
• determining pos/neg ratio
• determining hidden layer width/number
• determining how many ngrams (top/bottom n percent?) to take
• determining what type of other params are helpful



for hidden layers:
{'(1, 0)': 0.6630683461155342, '(1, 1)': 0.6060404363312661, '(1, 3)': 0.6974648189631985, '(1, 5)': 0.6911130497415368, '(1, 7)': 0.6918964807612382, '(1, 9)': 0.6640817682526752, '(3, 0)': 0.7023378095872852, '(3, 1)': 0.5414801318757592, '(3, 3)': 0.6928830871681333, '(3, 5)': 0.6780286492357941, '(3, 7)': 0.6965268555065811, '(3, 9)': 0.6744378523818619, '(5, 0)': 0.6940552841735776, '(5, 1)': 0.6360191771591696, '(5, 3)': 0.6719501821302483, '(5, 5)': 0.6866496441403518, '(5, 7)': 0.6810968898601057, '(5, 9)': 0.669193381830002, '(7, 0)': 0.694725462776083, '(7, 1)': 0.5696218724733351, '(7, 3)': 0.6806567850557197, '(7, 5)': 0.6770015397435507, '(7, 7)': 0.6790266206599973, '(7, 9)': 0.6753521878141827, '(9, 0)': 0.6873181932335275, '(9, 1)': 0.5727417106478189, '(9, 3)': 0.6397592044837519, '(9, 5)': 0.6568780249686592, '(9, 7)': 0.6665382753157478, '(9, 9)': 0.6728383650954253}
best combo:  (3, 0)  with success:  0.7023378095872852

for extraParams:
best combo:  allCapsRatios, years  with score:  0.7114033513522435
[0.65428974 0.62070467 0.68504782 0.69201905 0.69952953 0.70555917
 0.71140335 0.69588754 0.65536796 0.62984037 0.67313504 0.66948288
 0.67118622 0.65610531 0.65961511 0.68231941]
{'allCapsRatios, daysOfWeek, timesOfDay, years': 0.6542897447159495, 'daysOfWeek, timesOfDay, years': 0.6207046743647459, 'allCapsRatios, timesOfDay, years': 0.6850478158087151, 'timesOfDay, years': 0.6920190486802875, 'allCapsRatios, daysOfWeek, years': 0.6995295277107156, 'daysOfWeek, years': 0.7055591666628226, 'allCapsRatios, years': 0.7114033513522435, 'years': 0.6958875421158262, 'allCapsRatios, daysOfWeek, timesOfDay': 0.6553679566359951, 'daysOfWeek, timesOfDay': 0.6298403719310357, 'allCapsRatios, timesOfDay': 0.6731350354901863, 'timesOfDay': 0.6694828802504307, 'allCapsRatios, daysOfWeek': 0.6711862224608707, 'daysOfWeek': 0.6561053093042588, 'allCapsRatios': 0.6596151106414132, '': 0.6823194084589833}


for percentageNGrams:
best combo:  0.35  with score:  0.7175251423510199
[0.65684991 0.70882275 0.70069501 0.70654044 0.7019691  0.70709029
 0.71752514 0.69234235 0.69553161 0.71434023]
{0: 0.6568499072548706, 0.1: 0.7088227492978056, 0.15: 0.7006950119587376, 0.2: 0.7065404422047996, 0.25: 0.7019691023860316, 0.3: 0.7070902868737129, 0.35: 0.7175251423510199, 0.4: 0.6923423473370558, 0.45: 0.6955316088839528, 0.5: 0.7143402283777653}

Bonus for words appearing in caps:
{0: 0.67859148247568, 1: 0.7172941312976047, 2: 0.7119929060521837, 5: 0.672376713605003, 10: 0.5647923023158914, 100: 0.6617589895855891}


TODO:
Classification report/f1 score

precision is how many of the items classified a certain way were correct
recall is how many of the possible items were identified
Wikipedia says: precision is "how useful the search results are", and recall is "how complete the results are"
F1 is harmonic mean of precision & recall

Implement Poisson paper
Random features - is classification ~ 0.5?
split data test/(train/xval)



normalize fav counts from earlier epochs, try to use all data
Otherwise use good reason why not to use all data
'''


'''
Writeup clasifier/regression

Classifier:
MLP;
input:
	n-grams appearing at least twice in training set
	all-caps ratio; years (x-val'ed)
1 hidden layer with 3 neurons
1 output
input dimensions for last year (1/5):
	~678 tweets, 1454 dimensions each
input dimensions for all training:
	2860 tweets, 5876 dimensions each
	
accuracy for last year (1/5):
	about 75%
	
	
Problem: performs worse on all data than on 1/5 of data.
TODO: normalize all data to be as effective as 1/5 of data!

Normalization output:
Median, 30 day normalization window
{'allNormalized': 0.6887179405076264, 'allRegular': 0.6868387440694683, 'fifthNormalized': 0.6789939171207627, 'fifthRegular': 0.742883442516559}

Mean, 30 day normalization window
{'allNormalized': 0.6901158505754267, 'allRegular': 0.6859857486661065, 'fifthNormalized': 0.6585034922558279, 'fifthRegular': 0.7241664969025731}

Mean, 10 day normalization window
and that was the classification report. Good night!
{'allNormalized': 0.6899247523654606, 'allRegular': 0.6862917124847009, 'fifthNormalized': 0.6745167589861432, 'fifthRegular': 0.7324958333868847}

Mean, 90 day normalization window
{'allNormalized': 0.6902812174509178, 'allRegular': 0.6860567185245792, 'fifthNormalized': 0.6593163598657943, 'fifthRegular': 0.7264147277750503}


Fazit: normalisation does not help this model in classifying
	
Regressor:
mlPoisson

normalizing seems not to help, investigate further tomorrow.
Furthermore: investigate replies as extra param
furthermore: investigate other normalization strategies
furthermore: xVal for regression


Question: what are we trying to analyze? whether a tweet stands out against its immediate peers or its overall peers?
it probably doesn't make sense to train with whole dataset even normalized, because possible contanimation
of no longer relevant features are anyway bad. Possible amelioration to model, to add the temporal changing of features.
In any case, probably most relevant to train on only temporally relevant data

What stands out on the whole is the question? Then perhaps normalizing and the running should work...
	
	
	
	
TODO: twitter based word embeddings? send Stephi normalize/embedding plots
Check out DHD

Week: Topic Model

'''