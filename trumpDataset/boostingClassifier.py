



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

class BoostingClassifier():
	def __init__(self, furthestDeviationWords = [100,100], meaningfulClassifiersFromTopBottomPercentages = [100,100], \
				 filterOutliers = -1):
		'''

		:param furthestDeviationWords: position i+1 represents how many i-grams that deviate furthest from mean
		:param meaningfulClassifiersFromTopBottomPercentages: position i+1 represents the number of most/least
		popular i-grams

		... will have an attached classifier
		:param filterOutliers: if set to -1, no outliers will be filtered. Otherwise, it is the number of SD's away
		from the median that will be excluded

		'''

		self.furthestDeviationWords = furthestDeviationWords;
		self.meaningfulClassifiersFromTopBottomPercentages = meaningfulClassifiersFromTopBottomPercentages;
		self.filterOutliers = filterOutliers;

		self.sourceNgrams();

	def sourceNgrams(self):

		for nAsInGram,numberOfNGrams in enumerate(self.furthestDeviationWords):
			getNMostPopularWordsMMostPopularBigramsWithCounts



	def train(self):

