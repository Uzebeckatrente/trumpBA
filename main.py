from colorama import Fore, Style

# import trumpBA.trumpDataset;
try:
	from trumpDataset import *
except:
	from trumpBA.trumpDataset import *;
# import trumpBA.trumpDataset
import time
from sklearn.model_selection import train_test_split
# analyzeAllCapsPercentage();
# exit();
# graphPopularityByTweetsPerDay();

# graphFavCount();
#
# graphFavCountAndDerivative()
# analyzeAllCapsPercentage()
# analyzeShortTweetPercentage()

# graphFavCountLogRound()
# tweet = getTweetById(1194648339071016961)
# print(tweet)
# insertIntoDB(processTrumpTwitterArchiveFile());
# populateCleanedTextColumn()
# exit()
# exit()
# analyzeAllCapsPercentage();
# exit();
# populateDeletedColumn()
# fixIsRt(processTrumpTwitterArchiveFile())
# processTrumpTwitterArchiveFile()
#
# graphFavsAndRtsByRatio()
# graphFavsByApprRating()
# analyzeSkewOfKeywords(False)
# updateFavCounts()
# graphFavsAndRtsByRatio()
# graphPearsonsForApprFavOffset()
# favouriteOverTimeTrends()
# analyzeTopAndBottomPercentiles()
# graphFavCount();
# graphFollowerCountByFavCount()
# exit()
# compareVocabOverTime(18)
# visualizeFavCountDistributionForKeyword("great",True);
# analyzeAllCapsPercentage();
# analyzeOverUnderMeanSkewOfKeywords();
# graphMediaByFavCount();
# favouriteOverTimeTrends()
# tweetCountOverTime()
#
# analyzeOverUnderMeanSkewOfKeywords(loadFromStorage=True,numberInChart=10)
# exit()
# favouriteOverTimeTrends()
# exit()
# populateAllCapsPercentageColumn()

# print(computePercentageAllCaps("CHINA HAS ALREADY BEGUN AGRICULTURAL PURCHASES FROM OUR GREAT PATRIOT FARMERS &amp; RANCHERS!"))
#
# print(computePercentageAllCaps("#ThanksForDelivering @UPS! https://t.co/4Sis7Tme17"))
# print(computePercentageAllCaps("I agree 100%! https://t.co/lFUfnLefxT"))
# print(computePercentageAllCaps("I AM MAKING SOME LOUD NOISES @nose #NOISE"))
# analyzeAllCapsPercentage()
# exit()
# analyzeTopAndBottomPercentiles()
# exit()
# analyzeTopAndBottomPercentiles()
# exit()
# historgramFavCounts([favC[0] for favC in getTweetsFromDB(n=-1,purePres=True,returnParams=["favCount"])])
# exit()
# # graphPearsonsForApprFavOffset()
#
# # favouriteVsLengthTrends()
# # favouriteOverTimeTrends()
# # graphPearsonsForApprFavOffset()
# # populateCleanedTextColumn()
# # tweetCountOverTime()
# # graphFavCount()
# # graphFavCountLogRound()
#
# #
# # analyzeTopAndBottomPercentiles()
# # compareVocabOverTime(18)
# # populateNewsHeadlinesTable()
#
# # for epochCount in range(3,101,3):
# # 	compareVocabOverTime(epochCount)
# # # compareVocabOverTime(5)
# # exit()
#
#
# # t1 = getTweetById(986318102735572992);
# # t2 = getTweetById(1157352001593892864);
#
# # tEmpty = getTweetsFromDB(purePres=True)
# # print(len(tEmpty))
# # exit()
#
# # graphPopularityOfTweetsInSpurts()
# # # dataVisMain();
# # exit();
np.random.seed(69)
# compareVocabOverTime(4,minCount=250)
allPresTweetsFavCountAndCleanedText = getTweetsFromDB(purePres=True,conditions=["isReply = 0"],returnParams=["favCount","cleanedText, allCapsRatio, mediaType, publishTime","allCapsWords","tweetText"], orderBy= "publishTime asc")
# for numYears in [1,4]:
# 	graphPopularityByWeekday([t[:5] for t in allPresTweetsFavCountAndCleanedText],years=numYears);
# exit()
# # analyzeOverUnderMeanSkewOfKeywords(loadFromStorage=False,numberInChart=10,minCount=12,ns=[1,2])
# analyzeSentimentSkewsVader(allPresTweetsFavCountAndCleanedText);
#
# exit()
normalizedTweets = normalizeTweetsFavCountsSlidingWindowStrategy(allPresTweetsFavCountAndCleanedText,30, determinator = "mean");
normalizedTweetsFixedWindow = normalizeTweetsFavCountsFixedWindowStrategy(allPresTweetsFavCountAndCleanedText,30, determinator = "mean");

endorseTweets = getTweetsFromDB(conditions = ["cleanedText like \"%endorse%\""],purePres = True,returnParams = ["favCount","cleanedText"])
hillaryTweets = getTweetsFromDB(conditions = ["cleanedText like \"%hillary%\""],purePres = True,returnParams = ["favCount","cleanedText"])
allTweets = hillaryTweets + endorseTweets;

# hunna = getTweetsFromDB(purePres=True,n=100,returnParams = ["favCount","cleanedText"])
# docs = [h[1] for h in hunna]



# allPresTweetsFavCountAndCleanedText.sort(key=lambda t: t[0]);
# for i in [0.1,0.2,0.3]:
# 	print(Fore.MAGENTA,"i: ",i,Fore.RESET)
# 	bottomTenthAndTopTenth =allPresTweetsFavCountAndCleanedText[:int(len(allPresTweetsFavCountAndCleanedText)*i)]+allPresTweetsFavCountAndCleanedText[int(len(allPresTweetsFavCountAndCleanedText)*(1-i)):]
# 	mainTrumpLDA(bottomTenthAndTopTenth)
# exit();

# graphFavCount(allPresTweetsFavCountAndCleanedText,title="unnormalized");
# graphFavCount(normalizedTweets,title="normalizedSlidingWindow");
# graphFavCount(normalizedTweetsFixedWindow,title="normalizedFixedWindow");



# graphTwoDataSetsTogether([t[0] for t in normalizedTweets],"ansatz Sliding",[t[0] for t in normalizedTweetsFixedWindow]," ansatz Fixed");



allPresTweetsFavCountAndCleanedTextByYear5 = splitTweetsByYear(allPresTweetsFavCountAndCleanedText,5);
allPresTweetsFavCountAndCleanedTextByYear5Normalized = splitTweetsByYear(normalizedTweets,5);
allPresTweetsFavCountAndCleanedTextByYear3 = splitTweetsByYear(allPresTweetsFavCountAndCleanedText,4);


# graphPopularityOfTweetsInSpurts();
# graphPopularityByTweetsPerDay(allPresTweetsFavCountAndCleanedText);
# graphPopularityByWeekday(allPresTweetsFavCountAndCleanedText)
# graphPopularityByDayTime(allPresTweetsFavCountAndCleanedText)
#
# exit();

# populateNewsRatioColumn()#;(allPresTweetsFavCountAndCleanedText);


trainingLastYear, testLastYear, foldsLastYear = splitTrainingTest(allPresTweetsFavCountAndCleanedTextByYear5[-1])
trainingLastYearNorm, testLastYearNorm, foldsByYearNorm = splitTrainingTest(allPresTweetsFavCountAndCleanedTextByYear5Normalized[-1])
trainingLastYearNorm.sort(key = lambda x: x[0]);
testLastYear.sort(key = lambda x: x[0]);
trainingLastYear3, testLastYear3, _ = splitTrainingTest(allPresTweetsFavCountAndCleanedTextByYear3[-1])
training, test, folds = splitTrainingTest(allPresTweetsFavCountAndCleanedText);
trainingNorm, testNorm, foldsNorm = splitTrainingTest(allPresTweetsFavCountAndCleanedText);
allTweets = training+test;


print(Fore.LIGHTRED_EX,str(len(training))," training tweets; ",str(len(test))," test tweets",Style.RESET_ALL);






# allCapsClassifier = AllCapsClassifier()
newPoissonClassifier = RegressionModel("mlpClassifier");
newPoissonClassifier2 = RegressionModel("mlPoisson");
poissonRegressor = RegressionModel("poisson");
olsClassifier = RegressionModel("ols")
mlpRegressor = RegressionModel("mlpRegressor")

# wordEmbeddingMatrix = newPoissonClassifier.computeWordEmbeddingMatrix(trainingLastYear);
# print(wordEmbeddingMatrix);

# newPoissonClassifier.train(trainingLastYear,percentileSplit=0.75,percentileNegativePositive=-1,hiddenLayers=(3,),wordEmbeddingMatrix=True);
# newPoissonClassifier.test(trainingLastYear,title="training",crossVal=False);
# scoreNew = newPoissonClassifier.test(testLastYear, title="test",crossVal=False);
#
# newPoissonClassifier.train(trainingLastYear,percentileSplit=0.75,percentileNegativePositive=-1,wordEmbeddingMatrix=False);
# newPoissonClassifier.test(trainingLastYear,title="training",crossVal=False);
# scoreNew = newPoissonClassifier.test(testLastYear, title="test",crossVal=False);


# print("XVAL on neg/pos ratio for last year")
# newPoissonClassifier.crossValAllCapsBoostTargetMatrix(allPresTweetsFavCountAndCleanedTextByYear5[-1]);
# print("XVAL neg/pos ratio for whole time span")
# exit();
# mlpClassifier = MLPPopularity(0.15);
# mlpClassifier.train(trainingLastYear);
# mlpClassifier.test(trainingLastYear,display = True, title="training");
# scoreOld = mlpClassifier.test(testLastYear,display = True,title="test");
#
# newPoissonClassifier.crossValPercentageOfGramsToTake(trainingLastYearNorm)#,alpha=0.1,percentageNGrams=0,wordEmbeddingMatrix=True);

# mlpClassifier.crossValRegularisatzia(allPresTweetsFavCountAndCleanedTextByYear5[-1])


# newPoissonClassifier2.train(trainingLastYear,percentageNGrams=0.5,wordEmbeddingMatrix=True,alpha=5);

'''
EWIGEN
'''
# olsClassifier.train(trainingLastYear,percentageNGrams=0.35,includeWordEmbeddingMatrix=True,alpha=10)
# scoreNew = olsClassifier.test(testLastYear, title="Ridge Regression On Final Year Test Data",crossVal=False);
#
# olsClassifier.train(trainingLastYear,percentageNGrams=0.35,includeWordEmbeddingMatrix=False,alpha=10)
# scoreNew = olsClassifier.test(testLastYear, title="Ridge Regression On Final Year Test Data",crossVal=False);
# olsClassifier.test(trainingLastYear, title="Ridge Regression On Final Year Train Data",crossVal=False);

# olsClassifier.crossValPercentageOfGramsToTake(allPresTweetsFavCountAndCleanedTextByYear5[-1])



# mlpRegressor.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=True,alpha=100000,numIterationsMax=500);
# scoreNew2 = mlpRegressor.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);
#
# mlpRegressor.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=False,alpha=100000,numIterationsMax=500);
# scoreNew2 = mlpRegressor.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);
# mlpRegressor.test(trainingLastYear,title="training wordEmbeddingMatrix=False",crossVal=False);
# #
# newPoissonClassifierReg.crossValPercentageOfGramsToTake(allPresTweetsFavCountAndCleanedTextByYear5[-1],alphasPoss=[500,1000,10000,100000])

# newPoissonClassifierReg.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=True,alpha=1000);
# scoreNew2 = newPoissonClassifierReg.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);
# newPoissonClassifierReg.test(trainingLastYear,title="training wordEmbeddingMatrix=False",crossVal=False);

# newPoissonClassifier2.train(trainingLastYear,percentageNGrams=0.5,includeWordEmbeddingMatrix=True,alpha=100);
# scoreNew2 = newPoissonClassifier2.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);
# newPoissonClassifier2.test(trainingLastYear,title="training wordEmbeddingMatrix=False",crossVal=False);

fourModelComparisonRegression(trainingLastYear,testLastYear,allPresTweetsFavCountAndCleanedTextByYear5)
exit()
'''
End Ewigen
'''


# newPoissonClassifierReg.crossValPercentageOfGramsToTake(allPresTweetsFavCountAndCleanedTextByYear5[-1])

#

# mlpClassifier.crossValNumHiddenLayers(allPresTweetsFavCountAndCleanedTextByYear5[-1])
# mlpRegressor.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=True)
# scoreNew = mlpRegressor.test(testLastYear, title="Ridge Regression On Final Year Test Data",crossVal=False);
# mlpRegressor.test(trainingLastYear, title="Ridge Regression On Final Year Train Data",crossVal=False);

# scoreNew = newPoissonClassifier2.test(testLastYear, title="test wordEmbeddingMatrix=True",crossVal=False);
# newPoissonClassifier2.test(trainingLastYear,title="training wordEmbeddingMatrix=True",crossVal=False);

# newPoissonClassifier2.crossValRegularisatzia(allPresTweetsFavCountAndCleanedTextByYear5[-1])

# exit();
#
# newPoissonClassifierReg.crossValEverything(allPresTweetsFavCountAndCleanedTextByYear5[-1])

# newPoissonClassifier2.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=True,alpha=100);
# newPoissonClassifier2.test(trainingLastYear,title="training wordEmbeddingMatrix=False",crossVal=False);
# scoreNew2 = newPoissonClassifier2.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);
#


#
# newPoissonClassifier2.train(trainingLastYearNorm,percentageNGrams=0.5,wordEmbeddingMatrix=True);
# newPoissonClassifier2.test(trainingLastYearNorm,title="training wordEmbeddingMatrix=True",crossVal=False);
# scoreNew = newPoissonClassifier2.test(testLastYearNorm, title="test wordEmbeddingMatrix=True",crossVal=False);
#
# newPoissonClassifier2.train(trainingLastYearNorm,percentageNGrams=0.5,wordEmbeddingMatrix=False);
# newPoissonClassifier2.test(trainingLastYearNorm,title="training wordEmbeddingMatrix=False",crossVal=False);
# scoreNew = newPoissonClassifier2.test(testLastYearNorm, title="test wordEmbeddingMatrix=False",crossVal=False);

# newPoissonClassifier.crossValNumHiddenLayers(allPresTweetsFavCountAndCleanedTextByYear5[-1]);
# plt.plot([1,2],[1,2])
# plt.show();
# newPoissonClassifier2.crossValNumHiddenLayers(allPresTweetsFavCountAndCleanedTextByYear5[-1]);
#
# exit();



# newPoissonClassifier.train(training);
# newPoissonClassifier.test(training,title="training",crossVal=True);
# scoreNew = newPoissonClassifier.test(test, title="test",crossVal=True);
#
# newPoissonClassifier.train(trainingNorm);
# newPoissonClassifier.test(trainingNorm,title="training",crossVal=True);
# scoreNew = newPoissonClassifier.test(testNorm, title="test",crossVal=True);
#
# newPoissonClassifier.train(trainingLastYear,allCapsBoost=0);
# newPoissonClassifier.test(trainingLastYear,title="training",crossVal=False);
# scoreNew = newPoissonClassifier.test(testLastYear, title="test",crossVal=False);
#
# newPoissonClassifier.train(trainingLastYear,allCapsBoost=0.5);
# newPoissonClassifier.test(trainingLastYear,title="training",crossVal=False);
# scoreNew = newPoissonClassifier.test(testLastYear, title="test",crossVal=False);

# newPoissonClassifier.train(trainingLastYearNorm);
# newPoissonClassifier.test(trainingLastYearNorm,title="training",crossVal=True);
# scoreNew = newPoissonClassifier.test(testLastYearNorm, title="test",crossVal=True);

scores = {};

# favouriteOverTimeTrends();

if True:
	scores["allNormalized"]= [];
	for i in range(len(foldsNorm)):
		trainFlat, holdOut = flattenFolds(foldsNorm, i)
		newPoissonClassifier2.train(trainFlat,extraParamNum=0)
		# poissonClassifier.train(training,reload=False)
		newPoissonClassifier2.test(trainFlat,"training",crossVal=True);
		score=newPoissonClassifier2.test(holdOut,"test",crossVal=True);
		scores["allNormalized"].append(score);

	scores["allRegular"]= [];
	for i in range(len(foldsNorm)):
		trainFlat, holdOut = flattenFolds(foldsNorm, i)
		newPoissonClassifier2.train(trainFlat,extraParamNum=0)
		# poissonClassifier.train(training,reload=False)
		newPoissonClassifier2.test(trainFlat,"training",crossVal=True);
		score=newPoissonClassifier2.test(holdOut,"test",crossVal=True);
		scores["allRegular"].append(score);
	# 
	# 
	# scores["fifthNormalized"]= [];
	# for i in range(len(foldsByYearNorm)):
	# 	trainFlat, holdOut = flattenFolds(foldsByYearNorm, i)
	# 	newPoissonClassifier.train(trainFlat,extraParamNum=0)
	# 	# poissonClassifier.train(training,reload=False)
	# 	newPoissonClassifier.test(trainFlat,"training",crossVal=True);
	# 	score=newPoissonClassifier.test(holdOut,"test",crossVal=True);
	# 	scores["fifthNormalized"].append(score);

for s in scores.keys():
	scores[s] = np.mean(scores[s]);
print(scores);

exit()

scores["fifthWithEmbedding"]= [];
for i in range(len(foldsLastYear)):
	if i == 4:
		print(i);
	trainFlat, holdOut = flattenFolds(foldsLastYear, i)
	newPoissonClassifier.train(trainFlat,extraParamNum=0,wordEmbeddingMatrix=True)
	# poissonClassifier.train(training,reload=False)
	newPoissonClassifier.test(trainFlat,"training",crossVal=True);
	score=newPoissonClassifier.test(holdOut,"test",crossVal=True);
	scores["fifthWithEmbedding"].append(score);



scores["fifthSansEmbedding"]= [];
for i in range(len(foldsLastYear)):
	trainFlat, holdOut = flattenFolds(foldsLastYear, i)
	newPoissonClassifier.train(trainFlat,extraParamNum=0,wordEmbeddingMatrix=False)
	# poissonClassifier.train(training,reload=False)
	newPoissonClassifier.test(trainFlat,"training",crossVal=True);
	score=newPoissonClassifier.test(holdOut,"test",crossVal=True);
	scores["fifthSansEmbedding"].append(score);







# booster = MixedBoostingBayesClassifierPercentile(percentile=0.5)
# booster = MixedBoostingBayesClassifier();

# mlpClassifier =

booster.train(training,testData = test,retabulate = False)

print(Fore.CYAN,"test error onlyLen True: ")
misClassifiedsBooster = booster.test(test,title="testing data mixed")
booster.test(training,title="training data mixed")

naiveClassifier = PureBayesClassifier([1,2,3,4],0.25)
naiveClassifier.train(training);

misClassifiedNaive=naiveClassifier.test(test,title="test data naive");
naiveClassifier.test(training,title="training data naive");

allMisClassifieds = []
for l in misClassifiedNaive:
	allMisClassifieds.extend(l)
for l in misClassifiedsBooster:
	allMisClassifieds.extend(l)
allMisClassifieds.extend(misClassifiedsAllCaps);


print("\n\n\n")
print("misclassified total: ")
misClassifiedCounts = {}
for tweet in set(allMisClassifieds):
	try:
		misClassifiedCounts[allMisClassifieds.count(tweet)] += 1;
	except:
		misClassifiedCounts[allMisClassifieds.count(tweet)] = 1;
print(misClassifiedCounts);
exit()



print("\n\n")
print(Fore.LIGHTRED_EX,"train error: ")
booster.test(training,title="training error")




exit()

analyzeOverUnderMeanSkewOfKeywords(loadFromStorage=True,tweetsHash="purePres")
exit()
# groupTweetsByMonthAndApprRating()
# sourceApprovalRatings()
# graphFavsAndRtsByRatio()
# graphFavsByApprRating()
# exit()

# mycursor.execute("select * from tta where media = \"checkme!\" and isRt = 0")
# tweets = mycursor.fetchall()
# for tweet in tweets:
# 	print(tweet[0],"\n",extractMediaFromTweet(tweet[0]))
# 	newMedia = str(input(""));
# 	query = "update tta set media = \"" + newMedia + "\" where id = " + str(tweet[-3]);
# 	print(query)
# 	mycursor.execute(query)
# 	mydb.commit()
# populateMediaColumn()
# mediaTweets = getPresidentTweetsWithMedia()
# for t in mediaTweets:
# 	media = extractMediaFromTweet(t[0])
# graphFavsByApprRating()
# graphFavsAndRtsByRatio()
# computeMostCommonWordsAndBigrams()
# computeMostCommonWordsAndBigrams()
# writeCleanedTotxtFile()
# computeWordEmbeddings()
# computeMostCommonWordsAndBigrams()
# populateCleanedTextColumn()


# graphFavsByApprRating()
# exit()



# computeMostCommonWordsAndBigrams()

exit()



mycursor.execute("select cleanedText from tta2 "+relevancyDateThreshhold)
presidentialTweets = mycursor.fetchall();
presidentialTweets = [t[0] for t in presidentialTweets]

vectorLen = 100

recomputeWordCounts = False
recomputeWordEmbeddingVectors = False
if recomputeWordCounts:
	computeMostCommonWordsAndBigrams()



if recomputeWordEmbeddingVectors:
	wordEmbeddings = computeWordEmbeddings()

	tweetsToVecsDict = {}
	timeStart = time.time()
	nForTopNWordEmbeddings = 5

	wordEmbeddingsMarix = np.ndarray((vectorLen, nForTopNWordEmbeddings*len(presidentialTweets)))

	for index,t in enumerate(presidentialTweets):
		topNVectors = getTopNVectorsForTweet(wordEmbeddings, t, nForTopNWordEmbeddings, vectorLen)
		for i in range(nForTopNWordEmbeddings):
			try:
				wordEmbeddingsMarix[:,index+i] = topNVectors[i]
			except:
				continue;
		if index%100 == 0:
			print(index/len(presidentialTweets))

	np.save("trumpDataset/wordEmbeddingMatrix.npy", wordEmbeddingsMarix)  # save
else:
	wordEmbeddingsMarix=np.load("trumpDataset/wordEmbeddingMatrix.npy")






means = []
percentagesCovered = []
ns = [i*5+5 for i in range(25)]
# ns = [100,200]
for numWords in ns:
	nMostPopularWordsWithCounts, mMostPopularBigramsWithCounts = getMostPopularWordsOverCountNMostPopularBigramsOverCount(numWords, -1);
	nMostPopularWordsJustWords = [w[0] for w in nMostPopularWordsWithCounts]
	mMostPopulaBigramsJustBigrams = [bg[0] for bg in mMostPopularBigramsWithCounts]
	print(Fore.GREEN,"n: ",numWords,Style.RESET_ALL);
	intersections, wordIntersections, _, numEmpty = analyzeCoverageOfNMostPopularWordsAndMMostPopularBigrams(nMostPopularWordsJustWords, mMostPopulaBigramsJustBigrams, presidentialTweets)
	print("intersections total: ",intersections,"\nthat means ",(len(presidentialTweets)-intersections[0])/len(presidentialTweets)*100,"% tweets covered")
	print("word intersections: ",wordIntersections,"\nthat means ",(len(presidentialTweets)-wordIntersections[0])/len(presidentialTweets)*100,"% tweets covered by words")
	# print("bigram intersections: ",bigramIntersections,"\nthat means ",(len(presidentialTweets)-bigramIntersections[0])/len(presidentialTweets)*100,"% tweets covered by bigrams")
	print("all of which is to say, there are ",numEmpty," non-coverable tweets, which means there are ",intersections[0]-numEmpty, " tweets we could cover and failed to")
	print("\n\n")

	percentagesCovered.append((len(presidentialTweets)-intersections[0])/len(presidentialTweets)*100)

	means.append(sum(list(map(lambda x: x[0]*x[1], intersections.items()))))

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
plt.subplot(2, 1, 1)
plt.plot(ns,means,color=color1,label="means")
plt.subplot(2,1,2)
plt.plot(ns,percentagesCovered,color=color2,label="percentages")
plt.legend()
plt.show()


# print(time.time()-timeStart)
'''
some rt's media is incomplete

Main observation: easier to learn lower percentiles.
Probably: the viral tweets use lots of unusual words, whereas there are typical "boring" triggers in his low-tweets


Observation: tweet popularity changes over time, it makes most sense to view tweets in a local time window. What makes tweets fluctuate by time?


'''