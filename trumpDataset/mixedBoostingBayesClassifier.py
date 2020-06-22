from .stats import ols
from .favs import getMedianFavCountPresTweets, getMeanFavCountPresTweets, \
	calculateAndStorenGramsWithFavsMinusMeanMedian, loadnGramsWithFavsMinusMeanMedian
from .part3funcs import extractNGramsFromCleanedText, getnGramsWithOverMOccurences, computeMostCommonnGrams
from .media import removeMediaFromTweet
from .basisFuncs import *
from .deletedAndAllCaps import getAllCapsSkewForAboveThreshold;

from keras import Sequential
# from keras import initializers;
from keras.layers import Dense


class MLPPopularity():

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
					targetMatrix[tweetIndex] = 1
				else:
					targetMatrix[tweetIndex] = 0;

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

		model = Sequential()
		# model.add(Dense(12, input_dim=inputDim, activation='relu'))
		model.add(Dense(12, input_dim=bigMatrix.shape[1], activation='relu',use_bias=True))
		model.add(Dense(1, activation='sigmoid',use_bias=True))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(bigMatrix, targetMatrix, epochs=50, batch_size=int(bigMatrix.shape[0]/5))

		B_Input_Hidden = model.layers[0].get_weights()[1]
		B_Output_Hidden = model.layers[1].get_weights()[1]

		self.model = model;

		print("trained in : ",time.time()-startTime)

		# self.weights = np.dot(np.dot(np.linalg.inv(np.dot(bigMatrix, targetMatrix)),bigMatrix),targetMatrix.T)


	def test(self,testTweets):


		testTweets.sort(key=lambda tup: tup[0]);

		numCorrect = 0;

		predictionMatrix = np.zeros((len(testTweets), len(self.allNGrams)));
		print("predicting dims: ", predictionMatrix.shape);

		predictionMatrix, targetMatrix = self.createDataAndTargetMatrices(testTweets);
		actualFavs = [t[0] for t in testTweets];
		# predictionMatrix = np.array([-123]*predictionMatrix.shape[0]);
		predictions = self.model.predict_classes(predictionMatrix)
		for i in range(predictions.shape[0]):
			prediction = predictions[i][0];
			target = targetMatrix[i][0]
			prod = prediction*target;
			if prediction == 0 and target == 0 or prediction > 0 and target > 0:
				numCorrect += 1;



		xes = [i for i in range(len(testTweets))];
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		fig.suptitle('Predicted FavCounts and real FavCounts (log)')
		plt.plot(xes, targetMatrix, 'g', label='Actual Counts')
		plt.plot(xes, predictions, 'r', label='Predicted Counts')
		plt.legend()
		plt.show()

		print(numCorrect/len(testTweets))





def trainFavoriteCountGuesser():
	'''
	DEPRICATED
	nn params: keywords with over 80 (subject to change) mentions, 25 bigrams
	length of original tweet
	hasArticleMedia, hasImageMedia, videoLength (0 if none)
	= (80)+(25)+1+1+3


	:return:
	'''


	def addToFeatureVector(featureVector,contents, featureVectorIndexCounter):
		featureVector[featureVectorIndexCounter[0]] = contents;
		featureVectorIndexCounter[0] += 1


	nMostPopularWordsWithCounts, mMostPopularBigramsWithCounts = getMostPopularWordsOverCountNMostPopularBigramsOverCount(125,25)
	##idea: vectors are too sparse: reduce dimenisonality significantly
	nMostPopularWordsJustWords = [w[0] for w in nMostPopularWordsWithCounts]
	mMostPopulaBigramsJustBigrams = [bg[0] for bg in mMostPopularBigramsWithCounts]



	mycursor.execute("select tweetText, cleanedText, mediaType, favCount from tta2 " + relevancyDateThreshhold + " and isRt = 0")
	presidentialTweets = mycursor.fetchall();
	inputDim = len(nMostPopularWordsWithCounts) + len(mMostPopularBigramsWithCounts) + 1 + 3

	INPUT = np.zeros((inputDim,len(presidentialTweets)))
	OUTPUT = np.zeros(len(presidentialTweets))

	for tweetIndex,t in enumerate(presidentialTweets):
		featureVectorIndexCounter = [0]

		featureVector = np.zeros((inputDim))

		cleanedTextWords = t[1].split(" ")
		cleanedBigrams = computeBigrams(cleanedTextWords)
		wordCountsClean = np.ndarray((len(nMostPopularWordsWithCounts)))
		bigramsCountClean = np.ndarray((len(mMostPopularBigramsWithCounts)))
		for index, word in enumerate(nMostPopularWordsJustWords):
			count = cleanedTextWords.count(word)
			wordCountsClean[index] = count
			addToFeatureVector(featureVector,count,featureVectorIndexCounter)


		for index, bigram in enumerate(mMostPopulaBigramsJustBigrams):
			count = cleanedBigrams.count(bigram)
			bigramsCountClean[index] = count;
			addToFeatureVector(featureVector, count, featureVectorIndexCounter)

		lenTweet = len(removeMediaFromTweet(t[0]))
		addToFeatureVector(featureVector,lenTweet,featureVectorIndexCounter)

		hasArticleMedia= int(t[2] == "articleLink")
		addToFeatureVector(featureVector, hasArticleMedia, featureVectorIndexCounter)
		hasImageMedia = int(t[2] == "photo")
		addToFeatureVector(featureVector, hasImageMedia, featureVectorIndexCounter)
		videoLen = t[2].split("#")
		if len(videoLen) == 2:
			videoLen = np.log(int(videoLen[1]))
		else:
			videoLen = 0;
		addToFeatureVector(featureVector, videoLen, featureVectorIndexCounter)

		INPUT[:,tweetIndex] = featureVector
		# OUTPUT[tweetIndex] = int(t[3]) > 80884
		OUTPUT[tweetIndex] = int(t[3])

	###pcaExperiment
	# meanDiffs = []
	# cov = np.cov(INPUT)
	# eigenvalues, eigenvectors = np.linalg.eigh(cov)
	# idx = np.argsort(eigenvalues)[::-1]  ##indices of eigenvalues by size
	# eigenvectors = eigenvectors[:, idx]
	# eigenvalues = eigenvalues[idx]# ith column is the ith eigenvector
	# for i in range(1,inputDim,5):
	# 	print("pca i",i)
	# 	eigenvectorsI = eigenvectors[:, :i]  ###only the first n
	# 	reducedData = np.dot(eigenvectorsI.T, INPUT)
	# 	reconstructedInput = np.dot(eigenvectorsI,reducedData)
	# 	means=np.mean(INPUT, axis=1)
	# 	reconstructedInput=  (reconstructedInput.T+means).T
	# 	diffs = [np.linalg.norm(reconstructedInput[:,j]-INPUT[:,j]) for j in range(INPUT.shape[1])]
	# 	meanDiffs.append(np.mean(diffs))
	#
	# plt.plot([i for i in range(1,inputDim,5)],meanDiffs)
	# print(meanDiffs)
	# plt.show()

	INPUT = reduceDimensPCA(INPUT,1)

	graphTwoDataSetsTogether(list(INPUT)[0],"pc1",list(OUTPUT),"favs")
	# graphTwoDataSetsTogether(list(OUTPUT), "pc1", list(OUTPUT), "favs")

	print("the x's have been created. Now we shall train!")

	model = Sequential()
	# model.add(Dense(12, input_dim=inputDim, activation='relu'))
	model.add(Dense(12, input_dim=1, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(INPUT.T, OUTPUT, epochs=400, batch_size=10000)

	predictions = model.predict_classes(INPUT.T)

	diffs = []

	for i in range(len(presidentialTweets)):
		diffs.append(np.fabs(predictions[i]- OUTPUT[i]))
		# print('%s => %d (expected %d)' % (presidentialTweets[i], predictions[i], OUTPUT[i]))
	print(np.mean(diffs))
	diffs = sorted(diffs)

	color1 = plt.cm.viridis(0)
	color2 = plt.cm.viridis(0.5)
	plt.plot([i for i in range(len(diffs))], diffs, color=color1)
	plt.yscale("log")
	plt.show()

class MixedBoostingBayesClassifier():
	def __init__(self, furthestDeviationWords = [100,100], meaningfulClassifiersFromTopBottomPercentages = [100,100], \
				 filterOutliers = -1, ns = [1,2,3,4]):
		'''
		TODO: make good documentation of this!

		Algorithm:

		take all n-grams of tweets in training set. Find the mean and median fav count for each n-gram.
		Find the difference between the n-gram's mean/media fav count and the global mean/median fav count.
		We call this the skew. We then show whether each n-gram has a globally positive or negative effect
		on fav count. To predict, we iterate over each n-gram and add the skews.

		:param furthestDeviationWords: position i+1 represents how many i-grams that deviate furthest from mean
		:param meaningfulClassifiersFromTopBottomPercentages: position i+1 represents the number of most/least
		popular i-grams

		... will have an attached classifier
		:param filterOutliers: if set to -1, no outliers will be filtered. Otherwise, it is the number of SD's away
		from the median that will be excluded

		'''

		self.tweets = getTweetsFromDB(purePres=True,returnParams=["favCount","cleanedText"])
		self.cleanTexts = [t[1] for t in self.tweets]
		self.favs = [t[0] for t in self.tweets]
		self.tweetsHash = hashTweets(self.tweets);
		self.ns = ns;
		self.furthestDeviationWords = furthestDeviationWords;
		self.meaningfulClassifiersFromTopBottomPercentages = meaningfulClassifiersFromTopBottomPercentages;
		self.filterOutliers = filterOutliers;
		self.learningRate = 0.0005;
		self.allCapsThreshhold = 0.7;

	def plotMedianDistribution(self):
		ax1= plt.subplot(2,1,1)
		ax2 = plt.subplot(2, 1, 2)
		ax1.scatter(list(range(len(self.distancesMedian))),[d for d in self.distancesMedian]);
		ax2.scatter(list(range(len(self.distancesMedian))), [np.sign(d)*np.log(np.fabs(d)) for d in self.distancesMedian]);
		ax1.plot([0,len(self.distancesMedian)],[0,0])
		ax2.plot([0, len(self.distancesMedian)], [0, 0])
		print(len([x for x in self.distancesMedian if x >= 0]) / len(self.distancesMedian))
		plt.show()



		# self.sourceNgramsForCurrentSavedSelfNsTweetsHash();

	def sourceMeanMedianInternalVars(self):
		self.nGramsWithFavsMinusMeanDict = {}
		for nGram in self.nGramsWithFavsMinusMean:
			self.nGramsWithFavsMinusMeanDict[nGram[0]] = nGram

		self.nGramsWithFavsMinusMedianDict = {}

		distancesMedian = [np.fabs(nGram[nGramStorageIndicesSkew["skew"]]) for nGram in self.nGramsWithFavsMinusMedian]
		self.distancesMedian = [nGram[nGramStorageIndicesSkew["skew"]] for nGram in self.nGramsWithFavsMinusMedian]
		distancesMean = [np.fabs(nGram[nGramStorageIndicesSkew["skew"]]) for nGram in self.nGramsWithFavsMinusMean]

		self.avgMeanSkew = np.mean(distancesMean)
		self.avgMedianSkew = np.mean(distancesMedian)

		self.nGramMedianWeights = {}

		for nGram in self.nGramsWithFavsMinusMedian:
			self.nGramsWithFavsMinusMedianDict[nGram[0]] = nGram
			self.nGramMedianWeights[nGram[0]] =10000/(nGram[nGramStorageIndicesSkew["count"]]*(nGram[nGramStorageIndicesSkew["std"]]));# nGram[nGramStorageIndices["skew"]]/self.avgMedianSkew;
			# self.nGramMedianWeights[nGram[0]] = 1;
		self.nGramMedianWeights["allCaps"] = 1;





		appearancesForEachNGram = [int(nGram[nGramStorageIndicesSkew["count"]]) for nGram in self.nGramsWithFavsMinusMedian]
		self.medianAppearances = np.median(appearancesForEachNGram)
		self.meanAppearances = np.mean(appearancesForEachNGram)

		self.meanStd = np.mean([nGram[nGramStorageIndicesSkew["std"]] for nGram in self.nGramsWithFavsMinusMedian])
		return

	def sourceNgramsForCurrentSavedSelfNsTweetsHash(self):
		t = time.time()

		self.nGramsWithFavsMinusMean, self.nGramsWithFavsMinusMedian = loadnGramsWithFavsMinusMeanMedian(self.ns, self.tweetsHash)
		self.sourceMeanMedianInternalVars()

		print("sourced in : ",time.time()-t)

		self.medianFavCount = getMedianFavCountPresTweets(self.trainingTweets)
		self.meanFavCount = getMeanFavCountPresTweets(self.trainingTweets)

		print("mean favs: ",self.meanFavCount," median favs: ",self.medianFavCount);

	def displayPredictionResults(self, title,sample = True):


		maxIndex = np.iinfo(int).max
		if sample:
			maxIndex = 10;


		dataFrameDict = {}
		dataFrameDict["tweets"] = self.seenTweets[:maxIndex];
		dataFrameDict["guessed score median"] = self.guessedScoresMedian[:maxIndex]
		dataFrameDict["guessed score mean"] = self.guessedScoresMean[:maxIndex]
		dataFrameDict["correctness mean"] = self.correctnessMean[:maxIndex]
		dataFrameDict["guessed score median"] = self.correctnessMedian[:maxIndex]
		df = pd.DataFrame(dataFrameDict)
		print(df)
		medianSuccess = sum(self.correctnessMedian)/len(self.correctnessMedian);
		meanSuccess = sum(self.correctnessMean) / len(self.correctnessMean);
		print("median success: ",medianSuccess)
		print("mean success: ", meanSuccess)
		print("mean success word count: ",np.mean(self.lensSuccessMeans)," and failure word count: ",np.mean(self.lensFailureMeans))
		print("medians success word count: ", np.mean(self.lensSuccessMedians), " and failure word count: ",
			  np.mean(self.lensFailureMedians))
		print("percentage of positive median guesses: ",self.positiveMedianGuesses/len(self.seenTweets),"; mean guesses: ",self.positiveMeanGuesses/len(self.seenTweets))
		print("percentage of positive median tweets: ",self.actuallyPositiveTweetsMedian/len(self.seenTweets),"; mean tweets: ",self.actuallyPositiveTweetsMean/len(self.seenTweets))

		print("average median score: ",np.mean(self.guessedScoresMedian), " ||", len(self.guessedScoresMedian))
		print("average mean score: ", np.mean(self.guessedScoresMean), " ||", len(self.guessedScoresMedian))
		fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		ax1 = plt.subplot(2,1,1)
		ax2 = plt.subplot(2,1,2)
		ax1.title.set_text("means: "+title+ "  " + str(meanSuccess))
		ax2.title.set_text("medians: "+title +"  " + str(medianSuccess));
		ax1.scatter([tup[0] for tup in self.successScoresMean], [tup[1] for tup in self.successScoresMean], label="successes", color="red")
		ax1.scatter([tup[0] for tup in self.failureScoresMean], [tup[1] for tup in self.failureScoresMean], label= "failures",color="blue")

		ax2.scatter([tup[0] for tup in self.successScoresMedian], [tup[1] for tup in self.successScoresMedian],label="successes", color="red")
		ax2.scatter([tup[0] for tup in self.failureScoresMedian], [tup[1] for tup in self.failureScoresMedian],label="failures", color="blue")
		ax1.plot([self.medianFavCount,self.medianFavCount],[-1*max([tup[1] for tup in self.successScoresMean]),max([tup[1] for tup in self.successScoresMean])],color="red")
		ax2.plot([self.medianFavCount, self.medianFavCount], [-1*max([tup[1] for tup in self.successScoresMedian]),max([tup[1] for tup in self.successScoresMedian])], color="blue")

		ax1.set_xlim([0, 200000])
		ax2.set_xlim([0, 200000])

		ax1.plot([0, 200000],[0,0],color="cyan")
		ax2.plot([0, 200000], [0, 0],color="cyan")
		ax1.legend()
		ax2.legend()
		plt.show()

	def selectRandomBatch(self, sourceTweets, batchSize):
		indices = np.random.permutation(len(sourceTweets))
		batchIndices = indices[:batchSize]
		batch = []
		for i in batchIndices:
			batch.append(sourceTweets[i])
		return batch;

	def train(self, trainingTweets,testData = None, epochs = 40,retabulate = False):
		'''
		:param trainingTweets: [(favCount, cleanedText, allCapsRatio),...]
		:return:
		'''
		self.trainingTweets = trainingTweets
		self.trainingTweetsFavsAndCleanedText = [tweet[0:2] for tweet in trainingTweets];
		self.tweetsHash = calculateAndStorenGramsWithFavsMinusMeanMedian(self.ns,self.trainingTweetsFavsAndCleanedText,retabulate);
		self.sourceNgramsForCurrentSavedSelfNsTweetsHash()

		'''
		compute median medianScore 
		'''
		self.meanBias = 0;
		self.medianBias = 0;
		for e in range(epochs):
			medianScores = []
			meanScores = []
			updateTuples = []
			validCounter = 0;
			totalSuccesses = 0;
			totalFailures = 0;
			totalGuesses = 0;
			print(Fore.LIGHTGREEN_EX,"beginning epoch: ",e,Style.RESET_ALL);

			batch = self.selectRandomBatch(self.trainingTweets, batchSize = len(self.trainingTweets));
			for tweet in batch:
				realFavCount = tweet[0]
				cleanedText = tweet[1]
				allCapsRatio = tweet[2]
				mediaType = tweet[3]
				if len(tweet[1])> 2:
					meanScore, medianScore, numGramsUsed = self.predict(cleanedText, allCapsRatio,mediaType,training=True)

					meanSuccess = int(meanScore * (realFavCount - self.meanFavCount) > 0)
					medianSuccess = int(medianScore * (realFavCount - self.medianFavCount) > 0)
					if not medianSuccess:
						updateTuples.append((np.sign(realFavCount - self.medianFavCount),cleanedText, allCapsRatio))
						totalFailures += 1;
					else:
						totalSuccesses += 1;
					totalGuesses += 1;
					medianScores.append(medianScore)
					meanScores.append(meanScore)
					validCounter += 1
			if e == 0:

				self.meanBias = np.mean(meanScores)
				self.medianBias = np.median(medianScores)
			self.learningRate *= 0.99
			self.updateWeights(updateTuples)
			wouldBeBias = np.mean(medianScores);



			'''
			take 2 :)
			'''
			totalSuccessesTake2 = 0;
			totalGuessesTake2 = 0;
			for tweet in batch:
				realFavCount = tweet[0]
				cleanedText = tweet[1]
				allCapsRatio = tweet[2]
				mediaType = tweet[3]
				if len(tweet[1])> 2:
					_, medianScore, _ = self.predict(cleanedText, allCapsRatio,mediaType,training=True)

					medianSuccess = int(medianScore * (realFavCount - self.medianFavCount) > 0)
					if medianSuccess: totalSuccessesTake2 += 1;
					totalGuessesTake2 += 1;




			print("epoch ",e," finished; training accuracy: ",totalSuccesses/totalGuesses,"training inaccuracy: ",totalFailures/totalGuesses," median score: ", wouldBeBias,"; updating for: ",len(updateTuples)," tweets")
			print("second time around, ",totalSuccessesTake2/totalGuessesTake2)

			# self.test(testData,title="epoch: "+str(e));
	def updateWeights(self, updateTuples):#supposedToClass, cleanedText, allCapsRatio, ):
		'''
		updateTuples = [(supposedToClass, cleanedText, allCapsRatio),...]
		:param cleanedText:
		:param allCapsRatio:
		:param supposedToClass:
		:return:
		'''

		# print("updating dos weights")

		updateDirections = {}
		prevWeights = self.nGramMedianWeights.copy()
		shouldaBeenPositive = 0;
		for tuple in updateTuples:
			supposedToClass = tuple[0]
			cleanedText = tuple[1]
			allCapsRatio = tuple[2]
			nGrams = extractNGramsFromCleanedText(cleanedText, self.ns);
			shouldaBeenPositive += int(supposedToClass == 1);

			for nGram in nGrams:

				if nGram in self.nGramMedianWeights:

					self.nGramMedianWeights[nGram] += supposedToClass*self.learningRate;

		diffs = []
		for nGram in self.nGramMedianWeights:
			diffs.append((nGram,np.fabs(self.nGramMedianWeights[nGram]-prevWeights[nGram])))
		diffs.sort(key=lambda tup: tup[1],reverse=True);
		print("biggest shakers: ",diffs[:10]);


		print("pointing pos: ",shouldaBeenPositive/len(updateTuples),self.nGramMedianWeights["great"])
					# 	try:
					# 		updateDirections[nGram].append(1)
					# 	except:
					# 		updateDirections[nGram] = [1];
					# else:
					# 	try:
					# 		updateDirections[nGram].append(-1);
					# 	except:
					# 		updateDirections[nGram]=[-1];

		# 	if allCapsRatio > self.allCapsThreshhold:
		# 		self.
		# 		if supposedToClass == 1:
		# 			try:
		# 				updateDirections["allCaps"].append(1)
		# 			except:
		# 				updateDirections["allCaps"] = [1];
		# 		else:
		# 			try:
		# 				updateDirections["allCaps"].append(-1);
		# 			except:
		# 				updateDirections["allCaps"] = [-1];
		#
		# for nGram in updateDirections.keys():
		# 	updateDirections[nGram] = (np.std(updateDirections[nGram]),np.mean(updateDirections[nGram]));
		#
		# sortedUpdateDirections = {k: v for k, v in sorted(updateDirections.items(), key=lambda item: item[1][0])}
		# # sortedUpdateDirections=sortedUpdateDirections[:int(len(sortedUpdateDirections) * 0.2)]
		#
		#
		# nGramCounter = 0;
		# numPos = 0;
		# numTotal = 0;
		# for nGram in sortedUpdateDirections.keys():
		# 	if nGramCounter >= len(sortedUpdateDirections)*0.2:
		# 		break;
		# 	if sortedUpdateDirections[nGram][1] > 0:
		# 		numPos += 1
		# 		self.nGramMedianWeights[nGram] += self.learningRate
		# 	else:
		# 		self.nGramMedianWeights[nGram] -= self.learningRate
		# 	numTotal += 1;

		# print("numPos/numTotal: ",numPos/numTotal);
	def test(self, testTweets,title=""):

		print("training and the final predict weight: ",self.nGramMedianWeights["allCaps"]);

		self.seenTweets = []
		self.guessedScoresMean = []
		self.guessedScoresMedian = []
		self.correctnessMean = []
		self.correctnessMedian = []
		self.lensSuccessMeans = []
		self.lensFailureMeans = []
		self.lensSuccessMedians = []
		self.lensFailureMedians = []

		self.successScoresMean = []
		self.successScoresMedian = []
		self.failureScoresMean = []
		self.failureScoresMedian = []
		self.positiveMedianGuesses = 0;
		self.positiveMeanGuesses = 0;
		self.actuallyPositiveTweetsMean = 0;
		self.actuallyPositiveTweetsMedian = 0;

		misClassifiedTweetsMedian = []
		misClassifiedTweetsMean = []


		meanFavCount = self.meanFavCount
		medianFavCount = self.medianFavCount

		# countWeight = False, stdWeight = False, boolean = False




		for tweet in testTweets:

			realFavCount = tweet[0]
			cleanedText = tweet[1]
			allCapsRatio = tweet[2]
			mediaType = tweet[3]

			if len(cleanedText) > 2:
				meanScore, medianScore, numberOfAcceptedGrams = self.predict(cleanedText,allCapsRatio, mediaType)

				if numberOfAcceptedGrams == 0:
					continue;
				self.seenTweets.append(cleanedText)
				self.guessedScoresMean.append(round(meanScore, 2))
				self.guessedScoresMedian.append(round(medianScore, 2))
				self.positiveMedianGuesses += int(medianScore>0)
				self.positiveMeanGuesses += int(meanScore > 0)
				self.actuallyPositiveTweetsMean += int(realFavCount > self.medianFavCount)
				self.actuallyPositiveTweetsMedian += int(realFavCount > self.medianFavCount)
				meanSuccess = int(meanScore * (realFavCount - self.medianFavCount) > 0)
				medianSuccess = int(medianScore * (realFavCount - self.medianFavCount) > 0)

				self.correctnessMean.append(meanSuccess)
				self.correctnessMedian.append(medianSuccess)

				if meanSuccess:
					self.lensSuccessMeans.append(numberOfAcceptedGrams)
					self.successScoresMean.append((realFavCount,meanScore))
				else:
					self.lensFailureMeans.append(numberOfAcceptedGrams)
					self.failureScoresMean.append((realFavCount,meanScore))
					misClassifiedTweetsMean.append(tweet)

				if medianSuccess:
					self.lensSuccessMedians.append(numberOfAcceptedGrams)
					self.successScoresMedian.append((realFavCount,medianScore))
				else:
					self.lensFailureMedians.append(numberOfAcceptedGrams)
					self.failureScoresMedian.append((realFavCount,medianScore))
					misClassifiedTweetsMedian.append(tweet)


		self.displayPredictionResults(title=title)
		return [misClassifiedTweetsMean,misClassifiedTweetsMedian]

	def predictNgrams(self, cleanedText,countWeight=False, stdWeight=False, boolean=False):

		nGrams = extractNGramsFromCleanedText(cleanedText, self.ns);

		meanScore = 0;
		medianScore = 0;
		numberOfAcceptedGrams = 0;
		for nGram in nGrams:
			if nGram in self.nGramMedianWeights:
				numberOfAcceptedGrams += 1
				myGram = self.nGramsWithFavsMinusMedianDict[nGram];
				myGramSkew = myGram[nGramStorageIndicesSkew["skew"]] / self.avgMedianSkew;
				myWeight = self.nGramMedianWeights[nGram];
				medianScore += myGramSkew * myWeight;
			# if countWeight:
			# 	distanceVectorMedian *= self.nGramsWithFavsMinusMeanDict[nGram][nGramStorageIndices["count"]]/self.medianAppearances
			# if stdWeight:
			# 	distanceVectorMedian *= self.meanStd/self.nGramsWithFavsMinusMeanDict[nGram][nGramStorageIndices["std"]]
			# if boolean:
			# 	medianScore += np.sign(distanceVectorMedian);
			# else:
			# 	medianScore += distanceVectorMedian;
			# if nGram in self.nGramsWithFavsMinusMedianDict:
			# 	distanceVectorMedian = self.nGramsWithFavsMinusMedianDict[nGram][nGramStorageIndices["skew"]] / self.avgMedianSkew
			#
			# 	medianScore += distanceVectorMedian;
			if nGram in self.nGramsWithFavsMinusMeanDict:
				distanceVectorMean = self.nGramsWithFavsMinusMeanDict[nGram][
										 nGramStorageIndicesSkew["skew"]] / self.avgMeanSkew
				if countWeight:
					distanceVectorMean *= int(
						self.nGramsWithFavsMinusMeanDict[nGram][nGramStorageIndicesSkew["count"]]) / self.meanAppearances
				if stdWeight:
					distanceVectorMean *= self.meanStd / self.nGramsWithFavsMinusMeanDict[nGram][
						nGramStorageIndicesSkew["std"]]
				if boolean:
					meanScore += np.sign(distanceVectorMean)
				else:
					meanScore += distanceVectorMean;
			continue

		# print("tweet: ",cleanedText," median score said: ",round(medianScore,2)," mean score said: ",round(meanScore,2)," survey said: ",realFavCount)
		meanScore -= self.meanBias
		medianScore -= self.medianBias
		# if allCapsRatio > self.allCapsThreshhold:
		# 	medianScore += self.allCapsWeight

		return meanScore, medianScore, numberOfAcceptedGrams


	def predictAllCaps(self,allCapsRatio, mediaType,countWeight = False, stdWeight = False, boolean = False):


		if allCapsRatio > self.allCapsThreshhold and mediaType == "none":
			return 1;
		return -1;


	def predict(self,cleanedText,allCapsRatio, mediaType, training = False,countWeight = False, stdWeight = False, boolean = False):


		meanScore, medianScore, numberOfAcceptedGrams = self.predictNgrams(cleanedText);
		allCapsBoost = self.predictAllCaps(allCapsRatio, mediaType);
		if allCapsBoost > 0:
			# return meanScore,medianScore+self.nGramMedianWeights["allCaps"],numberOfAcceptedGrams;
			return meanScore,medianScore+self.nGramMedianWeights["allCaps"],numberOfAcceptedGrams;
		else:
			return meanScore,medianScore,numberOfAcceptedGrams;


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


'''
10 or 15 minutes
'''