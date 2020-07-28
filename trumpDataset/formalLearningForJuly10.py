from .basisFuncs import *
from .part3funcs import normalize_headline, extractNGramsFromCleanedText, computeMostCommonnGrams, getnGramsWithOverMOccurences,computeTFIDFMatrix;
from .visualization import graphTwoDataSetsTogether;
from .stats import ols;
from .favs import getMedianFavCountPresTweets,loadnGramsWithFavsMinusMeanMedian,calculateAndStorenGramsWithFavsMinusMeanMedian
from sklearn.linear_model import *
from sklearn.model_selection import cross_val_score;
from .daysOfTheWeek import determineDayOfWeek,determineSegmentOfDay,determineYearOfTweet
from sklearn.neural_network import MLPClassifier, MLPRegressor;
import sklearn.linear_model
from .wordEmbeddings import getSumOfVectorsForTweet;
from sklearn.metrics import classification_report,r2_score

from keras import backend as K, regularizers, optimizers

from keras import Sequential
from keras.layers import Dense

class RegressionModel():

	'''
	todo: how to tell OLS to focus on parameters which have a lower std?
	'''
	def __init__(self,learner,task=None):
		print(Fore.RED,"Initializing "+str(learner)+"!\n",Style.RESET_ALL)

		self.learner = learner;
		self.task =task;
		self.ns = [1, 2, 3, 4]  # ,3,4]

	def computeWordEmbeddingMatrix(self, trainingTweets):
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

	def createDataAndTargetMatrices(self, tweets, extraParameters = [], randomness= False):
		
		if self.wordEmbeddingMatrix:
			wordEmbeddingMatrix = self.computeWordEmbeddingMatrix(tweets);

		favs = [t[0] for t in tweets];
		bigMatrix = np.zeros((len(tweets), len(self.allNGrams)+len(extraParameters)+(wordEmbeddingMatrix.shape[1] if self.wordEmbeddingMatrix else 0)));
		targetMatrix = np.ndarray((len(tweets), 1))
		offSet = len(self.nGramIndices);

		for tweetIndex, tweet in enumerate(tweets):
			nGramsForTweet = extractNGramsFromCleanedText(tweet[1], self.ns);

			allCapsForTweet = [t.strip() for t in tweet[5].lower().split(" ") if len(t.strip()) > 0];
			for nGram in nGramsForTweet:
				if nGram in self.nGramIndices:
					nGramIndex = self.nGramIndices[nGram]
					if nGram in allCapsForTweet:
						bigMatrix[tweetIndex][nGramIndex] = 1+self.allCapsBoost;
					else:
						bigMatrix[tweetIndex][nGramIndex] = 1.#/self.allNGramsWithCountsDict[nGram];

			if self.task == "class":
				#classification!
				if tweet[0]> self.twoClassBarrierCount:
					targetMatrix[tweetIndex] = 1
				# elif tweet[0] < self.twoClassBarrierCountUnder:
				# 	targetMatrix[tweetIndex] = -10;
				else:
					targetMatrix[tweetIndex] = 0;
			else:
				#regression!
				if self.logReg and False:

					targetMatrix[tweetIndex] = int(tweet[0]/10000);
					# print(int(np.log(np.log(tweet[0]*100))));
					# targetMatrix[tweetIndex] = int(np.log(np.log(tweet[0]*100)));
				else:

					targetMatrix[tweetIndex] = tweet[0];
					# targetMatrix[tweetIndex] = 10;


			for param in range(len(extraParameters)):
				bigMatrix[tweetIndex][offSet + param] = extraParameters[param][tweetIndex];
		
		if self.wordEmbeddingMatrix:
			bigMatrix[:,offSet+len(extraParameters):] = wordEmbeddingMatrix;

		# bigMatrix = np.zeros((len(tweets), 101));#len(self.allNGrams) + len(extraParameters) + (wordEmbeddingMatrix.shape[1] if self.wordEmbeddingMatrix else 0)));
		# for i in range(len(tweets)):
		# 	bigMatrix[i][69] = 2;
		return bigMatrix,targetMatrix



	def getExtraParams(self,tweets, extraParamNum):
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

	def normalizeNegPos(self,tweets, distancePercentileFromSplit=-1):
		favs = [t[0] for t in tweets];
		favs.sort();
		indexOfPercentileCount = favs.index(self.twoClassBarrierCount);
		if distancePercentileFromSplit != -1:
			bottomIndex = indexOfPercentileCount - int(indexOfPercentileCount * distancePercentileFromSplit);
			topIndex = indexOfPercentileCount + int(indexOfPercentileCount * (1 - distancePercentileFromSplit))
			# trainingTweets = trainingTweets[:bottomIndex] + trainingTweets[topIndex:];

		theseNegatives = [t for t in tweets if t[0] < self.twoClassBarrierCount]
		thesePositives = [t for t in tweets if t[0] >= self.twoClassBarrierCount]

		negativeTrainingIDX = np.random.choice(len(theseNegatives), min(len(theseNegatives), len(thesePositives) * self.percentileNegativePositive), replace=False);
		myNegs = []
		for i in range(len(theseNegatives)):
			if i in negativeTrainingIDX:
				myNegs.append(theseNegatives[i])
		lennus = ((len(myNegs), len(thesePositives)));
		print(Fore.MAGENTA, "ratio of positives to negatives in learning sample: ", lennus, Fore.RESET)
		trainingTweets = myNegs + thesePositives;
		return trainingTweets;

	def train(self,trainingTweets,extraParamNum=6,hiddenLayers = (3,),allCapsBoost=0.5,alpha=0.1,percentileNegativePositive=1,percentileSplit = 0.75,wordEmbeddingMatrix=False, twoClassBarrierCount=-1,logReg = False,distancePercentileFromSplit = 0.4, percentageNGrams = 0.5):
		'''

		:param trainingTweets: tweets in format ["favCount","cleanedText, allCapsRatio, mediaType, publishTime","allCapsWords"]
		:param extraParamNum: \in [0,15], describes the additional parameters used by the model, Horner theorem counts down; ["allCapsRatios","daysOfWeek","timesOfDay","years"]
		:param hiddenLayers: tuple of hidden layer widths for model if MLP-based
		:param allCapsBoost: extra weight for words that are spelled all-caps
		:param alpha: regularization parameter
		:param percentileNegativePositive: ratio of positive to negative tweets in training set, for classification
		:param percentileSplit: fulcrum for 2-class classification
		:param wordEmbeddingMatrix: boolean, if we are using word embeddings in our trainer (later becomes a matrix)
		:param twoClassBarrierCount: private parameter, normally set to -1, otherwise to indicate the count used for the fulcrum
		:param logReg: regularize the fav counts
		:param distancePercentileFromSplit: not implemented, would indicate the training tweets which we ignore that are close to the fulcrum
		:param percentageNGrams: 0 to 0.5, if 0 no n-grams are used, if 0.5 all are used, if in between only the most/least popular n-grams are used
		:return:
		'''
		trainingTweets.sort(key=lambda tuple: tuple[0], reverse=False)
		'''
		init
		'''
		if self.learner in ["poisson","ols","mlPoisson"]:
			self.task = "reg"
		else:
			self.task = "class";
		if self.task == "class":
			if alpha == None:
				alpha = 4;
			if allCapsBoost == None:
				allCapsBoost = 0.5
		if self.learner == "poisson":
			self.model = PoissonRegressor(alpha=10)
		elif self.learner == "ols":
			self.model = LinearRegression();
		elif self.learner == "mlpClassifier":
			self.model = MLPClassifier(hidden_layer_sizes=hiddenLayers,alpha=alpha);
		elif self.learner == "mlPoisson":
			self.model = Sequential();



		else:
			alpha = 0.5;

		self.wordEmbeddingMatrix = wordEmbeddingMatrix;
		self.logReg = logReg;
		self.percentileNegativePositive = percentileNegativePositive;
		self.percentileSplit = percentileSplit;
		if twoClassBarrierCount == -1:
			self.twoClassBarrierCount = trainingTweets[int(len(trainingTweets) * self.percentileSplit)][0];

		self.extraParamNum = extraParamNum;
		self.allCapsBoost = allCapsBoost;





		if self.task == "class" and self.percentileNegativePositive != -1:

			trainingTweets = self.normalizeNegPos(trainingTweets,-1);

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

		extraParams = self.getExtraParams(trainingTweets, extraParamNum);





		# bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets,[[np.log(t[0]) for t in trainingTweets],[(t[0]/100000) for t in trainingTweets]]);
		bigMatrix, targetMatrix = self.createDataAndTargetMatrices(trainingTweets,extraParams);
		print("dims: ", bigMatrix.shape);



		startTime = time.time()

		if self.learner == "mlPoisson":
			# bigMatrix = np.ones((1000,1))
			self.model.add(Dense(hiddenLayers[0], input_dim=bigMatrix.shape[1], activation='sigmoid', use_bias=True,kernel_regularizer=regularizers.l2(alpha)))
			for h in range(1,len(hiddenLayers)):
				self.model.add(Dense(hiddenLayers[h], activation='tanh', use_bias=True,kernel_regularizer=regularizers.l2(alpha)))
			# self.model.add(Dense(1, activation=self.poissonActivation, use_bias=True,kernel_regularizer=regularizers.l2(alpha)))
			self.model.add(Dense(1, use_bias=True,activation=self.poissonActivation,kernel_regularizer=regularizers.l2(alpha)))
			self.model.add(Dense(1, use_bias=True, activation=self.poissonActivation,kernel_regularizer=regularizers.l2(alpha)))
			self.model.compile(optimizer="adam", metrics=['accuracy'], loss=self.poissonLoss);
			# self.model.compile(optimizer=optimizers.Adam(lr=0.01),metrics=['accuracy']);

			self.model.fit(bigMatrix, targetMatrix, epochs=200, batch_size=100)
		else:
			np.sum(bigMatrix);
			np.sum(targetMatrix);
			self.model.fit(bigMatrix, targetMatrix);
		print(Fore.MAGENTA,"fitted in ",time.time()-startTime,Style.RESET_ALL);


		return ", ".join(self.extraParamNamesToInclude);



	def test(self,testTweets,title="",distancePercentileFromSplit = 0, crossVal = False):

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

		predictionMatrix = np.zeros((len(testTweets), len(self.allNGrams)));
		print("predicting dims: ", predictionMatrix.shape);

		extraParams = self.getExtraParams(testTweets,self.extraParamNum);


		# predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets,[[np.log(t[0]) for t in testTweets],[(t[0]/100000) for t in testTweets]]);
		predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets,extraParams)
		print("targetMaxAndMin ",min(targetMatrix),max(targetMatrix));
		actualFavs = [t[0] for t in testTweets];

		predictions = self.model.predict(predictionMatrix);
		print("predictions: ",predictions);
		plt.plot([i for i in range(len(predictions))],predictions);
		plt.show();

		if self.task == "class":

			correctPredictionsAggregator = {0: 0, 1: 0};
			predictionsAggregator = {0:0,1:0};


			for i in range(predictions.shape[0]):
				prediction = predictions[i];
				target = targetMatrix[i][0]
				prod = prediction * target;


				if prediction == 0 and target == 0 or prediction > 0 and target > 0:
					correctPredictionsAggregator[target] += 1;

				predictionsAggregator[target] += 1;
			score = 0.5*(correctPredictionsAggregator[0]/predictionsAggregator[0])+0.5*(correctPredictionsAggregator[1]/predictionsAggregator[1]);
		elif self.learner == "mlPoisson":
			print("lennus: ",targetMatrix.shape,predictions.shape)
			score = r2_score(targetMatrix,predictions);
		else:
			score = self.model.score(predictionMatrix,np.ravel(targetMatrix));
		#
		if self.task == "class":
			print(Fore.MAGENTA,"classification report: \n",Fore.RESET);
			print(classification_report(targetMatrix, predictions, target_names=["under " + str(self.percentileNegativePositive), "over " + str(self.percentileNegativePositive)]));
			print("and that was the classification report. Good night!")
		else:
			print(Fore.MAGENTA, "classification report: \n", Fore.RESET);
			print(r2_score(targetMatrix, predictions));
			print("and that was the classification report. Good night!")

		if not crossVal:


			xes = [i for i in range(len(testTweets))];
			fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
			fig.subplots_adjust(left=0.06, right=0.94)
			fig.suptitle('Predicted FavCounts and real FavCounts for '+title+ " score: " + str(score))
			plt.scatter(xes, predictions, c='r',label='Predicted Counts')
			plt.plot(xes, targetMatrix, 'go-',markersize=0.1, label='Actual Counts')
			plt.ylim(min(targetMatrix)-1,max(targetMatrix)+1);
			plt.legend()
			print(numCorrect / len(testTweets))
			plt.show()

		return score
		# return "new: ",score, "old way: ",scoreOldWay;


	def crossValNumHiddenLayers(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		hiddenLayerPoss = [3]#[1,3,5,7,9]#,20,30,40,50,100];
		index = 0;
		for numHoldoutFold in range(numFolds):
			trainFlat, holdOut = flattenFolds(folds, numHoldoutFold)
			for numHiddenLayers in hiddenLayerPoss:
				for numHiddenLayers2 in [0]:#[0]+hiddenLayerPoss:
					if numHiddenLayers2 == 0:
						self.train(trainFlat,hiddenLayers=(numHiddenLayers,),crossVal=True);
					else:
						self.train(trainFlat, hiddenLayers=(numHiddenLayers,numHiddenLayers2),crossVal=True);
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



	def crossValRegularisatzia(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key = lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {};
		resultDict = {}
		self.twoClassBarrierCount = tweets[int(len(tweets) * self.percentileNegativePositive)][0];
		alphaPossibilities = [1,2,3,4,5]#,5,7,20,30,40,50,100];
		index = 0;
		for alpha in alphaPossibilities:
			scoresDict[alpha] = []
			for numHoldoutFold in range(numFolds):
				train = folds[0:numHoldoutFold] + folds[numHoldoutFold + 1:];
				holdOut = folds[numHoldoutFold];
				trainFlat = [item for sublist in train for item in sublist]
				if self.percentileNegativePositive != -1:


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
				else:
					training = trainFlat;



				self.train(training,alpha=alpha,crossVal=True);
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

	def crossValPercentageOfGramsToTake(self,tweets):
		print("doing xval :)")
		numFolds = 5;
		tweets.sort(key=lambda t: t[0]);
		train, test, folds = splitTrainingTest(tweets,numFolds);
		scores = []
		scoresDict = {}


		resultDict = {}

		percentagesPoss=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
		for index,percentage in enumerate(percentagesPoss):
			scoresDict[percentage] = [];
			for i in range(numFolds):
				trainNotFlat = folds[:i]+folds[i+1:];
				trainFlat = [item for sublist in trainNotFlat for item in sublist]
				holdout = folds[i];



				self.train(trainFlat,percentageNGrams=percentage,wordEmbeddingMatrix=True);
				score = self.test(holdout,crossVal=True);

				scoresDict[percentage].append(score);

			score = np.mean(scoresDict[percentage]);
			resultDict[percentage] = score;
			scores.append(score);
			print(Fore.YELLOW,"percentage: ",percentage," score: ",score, " done with: ",index/len(percentagesPoss));

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
		plt.plot(percentagesPoss,scores);
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