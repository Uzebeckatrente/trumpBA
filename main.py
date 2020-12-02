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
# compareVocabOverTime(epochCount=4,maxNumberOfNGrams=10,minCount=200)
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
# graphPopularityOfTweetsInSpurts()
# favouriteOverTimeTrends()
# graphFavCount()
# # graphFavCountAndDerivative()
# tweetCountOverTime()
# justGreat()
# exit()
# # # dataVisMain();
# analyzeOverUnderMeanSkewOfKeywords(numberInChart = 10)

# analyzeOverUnderMeanSkewOfKeywords(loadFromStorage=False,numberInChart=10,minCount=10,ns=[1,2])
# analyzeShortTweetPercentage()
# exit();
np.random.seed(69)


# skewNew(4,minCount=10, maxNumberOfNGrams=10)
# relativeAbundanceNew(4,minCount=10, maxNumberOfNGrams=20)

allPresTweetsFavCountAndCleanedText = getTweetsFromDB(purePres=True,conditions=["isReply = 0"],returnParams=["favCount","cleanedText, allCapsRatio, mediaType, publishTime","allCapsWords","tweetText"], orderBy= "publishTime asc")



# analyzeHashtagsAndAtTags(allPresTweetsFavCountAndCleanedText);
#
#
# exit()

analyzeSentimentSkewsVader(allPresTweetsFavCountAndCleanedText);
# favs = [t[0] for t in allPresTweetsFavCountAndCleanedText];
# histogramFavCounts(favs);
# exit()
# favs.sort()
# plt.plot(favs);
# for i in range(10):
# 	print(favs[int(len(favs)*i/10)])
# plt.show()
# exit()
# for numYears in [1,4]:
# 	graphPopularityByWeekday([t[:5] for t in allPresTweetsFavCountAndCleanedText],years=numYears,paramColor = "#ff4452");
# exit()


#
# exit()
normalizedTweets = normalizeTweetsFavCountsSlidingWindowStrategy(allPresTweetsFavCountAndCleanedText,30, determinator = "mean");
normalizedTweetsFixedWindow = normalizeTweetsFavCountsFixedWindowStrategy(allPresTweetsFavCountAndCleanedText,30, determinator = "mean");

endorseTweets = getTweetsFromDB(conditions = ["cleanedText like \"%endorse%\""],purePres = True,returnParams = ["favCount","cleanedText"])
hillaryTweets = getTweetsFromDB(conditions = ["cleanedText like \"%hillary%\""],purePres = True,returnParams = ["favCount","cleanedText"])
allTweets = hillaryTweets + endorseTweets;

# hunna = getTweetsFromDB(purePres=True,n=100,returnParams = ["favCount","cleanedText"])
# docs = [h[1] for h in hunna]





# graphFavCount(allPresTweetsFavCountAndCleanedText,title="unnormalized");
# graphFavCount(normalizedTweets,title="normalizedSlidingWindow");
# graphFavCount(normalizedTweetsFixedWindow,title="normalizedFixedWindow");



# graphTwoDataSetsTogether([t[0] for t in normalizedTweets],"ansatz Sliding",[t[0] for t in normalizedTweetsFixedWindow]," ansatz Fixed");



allPresTweetsFavCountAndCleanedTextByYear5 = splitTweetsByYear(allPresTweetsFavCountAndCleanedText,5);
allPresTweetsFavCountAndCleanedTextByYear5Normalized = splitTweetsByYear(normalizedTweets,5);
allPresTweetsFavCountAndCleanedTextByYear3 = splitTweetsByYear(allPresTweetsFavCountAndCleanedText,4);

lastYearTweets = allPresTweetsFavCountAndCleanedTextByYear5[-1]
lastYearTweets.sort(key=lambda t: t[0]);
allPresTweetsFavCountAndCleanedText.sort(key = lambda t: t[0]);
meanFavCountAllTweets = np.mean([t[0] for t in allPresTweetsFavCountAndCleanedText])
medianFavCountAllTweets = np.median([t[0] for t in allPresTweetsFavCountAndCleanedText])
meanFavCountLastYear = np.mean([t[0] for t in lastYearTweets])
medianFavCountLastYear = np.median([t[0] for t in lastYearTweets])
lastYear = True
doLDA = False;
if doLDA:
	for percentile in [0.1,0.2,0.3,0.4,0.5]:
		print(Fore.MAGENTA,"i: ", percentile, Fore.RESET)
		if lastYear:
			bottomTenthAndTopTenth = lastYearTweets[:int(len(lastYearTweets) * percentile)] + lastYearTweets[int(len(lastYearTweets) * (1 - percentile)):]
			medianFavCount = medianFavCountLastYear
			meanFavCount = meanFavCountLastYear
		else:
			meanFavCount = meanFavCountAllTweets
			medianFavCount = medianFavCountAllTweets
			bottomTenthAndTopTenth = allPresTweetsFavCountAndCleanedText[:int(len(allPresTweetsFavCountAndCleanedText) * percentile)] + allPresTweetsFavCountAndCleanedText[int(len(allPresTweetsFavCountAndCleanedText) * (1 - percentile)):]
		mainTrumpLDA(bottomTenthAndTopTenth, percentile=percentile, meanForDataset=meanFavCount, medianForDataset=medianFavCount)
# exit();

# graphPopularityOfTweetsInSpurts();
# graphPopularityByTweetsPerDay(allPresTweetsFavCountAndCleanedText,numYears=1);
# graphPopularityByWeekday(allPresTweetsFavCountAndCleanedText)
np.random.seed(428)
# graphPopularityByWeekday(allPresTweetsFavCountAndCleanedText,years=4)
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

trainingLastYear3070 = trainingLastYear.copy()
trainingLastYear3070.sort(key = lambda tup: tup[0]);
trainingLastYear3070 = trainingLastYear3070[:int(len(trainingLastYear3070)*0.3)]+trainingLastYear3070[int(len(trainingLastYear3070)*0.7):]

testLastYear3070 = testLastYear.copy()
testLastYear3070.sort(key = lambda tup: tup[0]);
testLastYear3070 = testLastYear3070[:int(len(testLastYear3070)*0.3)]+testLastYear3070[int(len(testLastYear3070)*0.7):]

training3070 = training.copy()
training3070.sort(key = lambda tup: tup[0]);
training3070 = training3070[:int(len(training3070)*0.3)]+training3070[int(len(training3070)*0.7):]

test3070 = test.copy()
test3070.sort(key = lambda tup: tup[0]);
test3070 = test3070[:int(len(test3070)*0.3)]+test3070[int(len(test3070)*0.7):]

trainingLastYear1090 = trainingLastYear.copy()
trainingLastYear1090.sort(key = lambda tup: tup[0]);
trainingLastYear1090 = trainingLastYear1090[:int(len(trainingLastYear1090)*0.1)]+trainingLastYear1090[int(len(trainingLastYear1090)*0.9):]

testLastYear1090 = testLastYear.copy()
testLastYear1090.sort(key = lambda tup: tup[0]);
testLastYear1090 = testLastYear1090[:int(len(testLastYear1090)*0.1)]+testLastYear1090[int(len(testLastYear1090)*0.9):]

training1090 = training.copy()
training1090.sort(key = lambda tup: tup[0]);
training1090 = training1090[:int(len(training1090)*0.1)]+training1090[int(len(training1090)*0.9):]

test1090 = test.copy()
test1090.sort(key = lambda tup: tup[0]);
test1090 = test1090[:int(len(test1090)*0.1)]+test1090[int(len(test1090)*0.9):]


print(Fore.LIGHTRED_EX,str(len(training))," training tweets; ",str(len(test))," test tweets",Style.RESET_ALL);






# allCapsClassifier = AllCapsClassifier()
mlpClassifier = RegressionModel("mlpClassifier");

newPoissonClassifier2 = RegressionModel("mlPoisson");
mlPoissonClassifier = RegressionModel("mlPoissonClassifier");
poissonRegressor = RegressionModel("poisson");
poissonClassifier = RegressionModel("poissonClassifier");
olsRegressor = RegressionModel("ols")
svmClassifier = RegressionModel("svm")
mlpRegressor = RegressionModel("mlpRegressor")
olsClassifier = RegressionModel("olsClassifier")

# mlpClassifier.crossValPercentageOfGramsToTake(trainingLastYear3070+testLastYear3070)


# wordEmbeddingMatrix = newPoissonClassifier.computeWordEmbeddingMatrix(trainingLastYear);
# print(wordEmbeddingMatrix);

# newPoissonClassifier.train(trainingLastYear,percentileSplit=0.75,percentileNegativePositive=-1,hiddenLayers=(3,),wordEmbeddingMatrix=True);
# newPoissonClassifier.test(trainingLastYear,title="training",crossVal=False);
# scoreNew = newPoissonClassifier.test(testLastYear, title="test",crossVal=False);
#
# newPoissonClassifier.train(trainingLastYear,percentileSplit=0.75,percentileNegativePositive=-1,wordEmbeddingMatrix=False);
# newPoissonClassifier.test(trainingLastYear,title="training",crossVal=False);
# scoreNew = newPoissonClassifier.test(testLastYear, title="test",crossVal=False);

# olsClassifier.train(trainingLastYear,percentageNGrams=0.35,includeWordEmbeddingMatrix=True,alpha=10,transformation="log",numBoxes=5)
# scoreNew = olsClassifier.test(testLastYear, title="Ridge Regression On Final Year Test Data",crossVal=False);
# exit()
# precisionAttenuatesWithBoxSize([svmClassifier,mlpClassifier],["SVM","MLP"],[2,3,4,5,8,11,14,17,20,40],trainingLastYear,testLastYear)
#
# # precisionAttenuatesWithBoxWidth([poissonClassifier,mlPoissonClassifier],["poisson","MLPoisson"],[400000,200000,100000,75000,50000,20000,10000],trainingLastYear,testLastYear)
# mlPoissonClassifier.train(allPresTweetsFavCountAndCleanedText,percentageNGrams=0.5,includeWordEmbeddingMatrix=False,alpha=0.0001,transformation="poisson",boxSize=100000);
# mlPoissonClassifier.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);
#
# poissonClassifier.train(trainingLastYear,percentageNGrams=0.25,includeWordEmbeddingMatrix=True,alpha=0.001,transformation="poisson",boxSize=100000);
# scoreNew2 = poissonClassifier.test(testLastYear, title="test wordEmbeddingMatrix=False",crossVal=False);

# mlpClassifier.train(trainingLastYear, percentageNGrams=0.35, includeWordEmbeddingMatrix=False, alpha=1, transformation="log", numIterationsMax=210, numBoxes=5)
#
# scoreNew = mlpClassifier.test(testLastYear, title="MLP Classifier All Years Test Data",crossVal=False,targetNames=("box "+str(i) for i in range(5)));
# scoreNew = mlpClassifier.test(trainingLastYear, title="All Years Training Data",crossVal=False,targetNames=("box "+str(i) for i in range(5)));


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


# mlpClassifier.train(training, percentageNGrams=0.35, includeWordEmbeddingMatrix=False, alpha=1, numIterationsMax=200, percentileSplits=(0.1,0.9,));
# scoreNew2 = mlpClassifier.test(test, title="Test Data Set Final Year",crossVal=False,targetNames=("0-10","90-100"));
# scoreNew2 = mlpClassifier.test(test, title="test wordEmbeddingMatrix=False",crossVal=False,targetNames=("0-10","90-100"));

# lastYear = False
# scoresSVM = []
# scoresMLP = []
# for percentile in [0.1]:#,0.2,0.3,0.4,0.5]:
# 	print(Fore.MAGENTA,"i: ", percentile, Fore.RESET)
# 	if lastYear:
# 		# bottomTenthAndTopTenth = lastYearTweets[:int(len(lastYearTweets) * percentile)] + lastYearTweets[int(len(lastYearTweets) * (1 - percentile)):]
# 		bottomTenthAndTopTenth = lastYearTweets[int(len(lastYearTweets) * 0.0):int(len(lastYearTweets) * 0.1)] + lastYearTweets[int(len(lastYearTweets) * 0.5):int(len(lastYearTweets) * 0.6)]
#
# 	else:
# 		# bottomTenthAndTopTenth = allPresTweetsFavCountAndCleanedText[:int(len(allPresTweetsFavCountAndCleanedText) * percentile)] + allPresTweetsFavCountAndCleanedText[int(len(allPresTweetsFavCountAndCleanedText) * (1 - percentile)):]
# 		bottomTenthAndTopTenth = allPresTweetsFavCountAndCleanedText[int(len(allPresTweetsFavCountAndCleanedText) * 0.0):int(len(allPresTweetsFavCountAndCleanedText) * 0.1)] + allPresTweetsFavCountAndCleanedText[int(len(allPresTweetsFavCountAndCleanedText) * 0.5):int(len(allPresTweetsFavCountAndCleanedText) * 0.6)]
#
# 	trainingBottomAndTopTenth, testBottomAndTopTenth, _ = splitTrainingTest(bottomTenthAndTopTenth)
# 	for alpha in [0.1,0.5,1,5,10,15,20,30,50]:
# 		for percentageNGrams in [0.1,0.15,0.2,0.3,0.4,0.5]:
# 			svmClassifier.train(trainingBottomAndTopTenth, percentageNGrams=percentageNGrams, includeWordEmbeddingMatrix=False, alpha=alpha)
# 			#
# 			scoreSVM = svmClassifier.test(testBottomAndTopTenth, title="All Years Test Data", crossVal=True);
# 			mlpClassifier.train(trainingBottomAndTopTenth,percentageNGrams = percentageNGrams, includeWordEmbeddingMatrix = False, alpha = alpha, numIterationsMax = 200)
# 			scoreMLP = mlpClassifier.test(testBottomAndTopTenth, title="All Years Training Data", crossVal=True);
# 			# if scoreNew > 0.85:
# 			# 	mlpClassifier.test(testBottomAndTopTenth, title="All Years Training Data", crossVal=False);
# 			print("scoreNew: ", scoreSVM);
#
# 			scoresSVM.append((percentile, alpha, percentageNGrams, scoreSVM));
# 			scoresMLP.append((percentile,alpha,percentageNGrams,scoreMLP));
# scoresSVM.sort(key = lambda tup: tup[3], reverse=True)
# scoresMLP.sort(key = lambda tup: tup[3],reverse=True)
# print("mlp scores all years: ",scoresMLP);
# print("svm scores all years: ", scoresSVM);
# exit()
#
# # exit();
#
# svmClassifier.train(training3070,percentageNGrams=0.3,includeWordEmbeddingMatrix=False,alpha=20)
# scoreSVM = svmClassifier.test(test3070, title="All Years Test Data", crossVal=False);
# scoreSVM = svmClassifier.test(training3070, title="All Years Training Data", crossVal=False);

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

# training
fourModelComparisonRegression(training,test,allPresTweetsFavCountAndCleanedText)
exit()
'''
End EWIGEN
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

scoresSVM = {};

# favouriteOverTimeTrends();

if True:
	scoresSVM["allNormalized"]= [];
	for percentile in range(len(foldsNorm)):
		trainFlat, holdOut = flattenFolds(foldsNorm, percentile)
		newPoissonClassifier2.train(trainFlat,extraParamNum=0)
		# poissonClassifier.train(training,reload=False)
		newPoissonClassifier2.test(trainFlat,"training",crossVal=True);
		score=newPoissonClassifier2.test(holdOut,"test",crossVal=True);
		scoresSVM["allNormalized"].append(score);

	scoresSVM["allRegular"]= [];
	for percentile in range(len(foldsNorm)):
		trainFlat, holdOut = flattenFolds(foldsNorm, percentile)
		newPoissonClassifier2.train(trainFlat,extraParamNum=0)
		# poissonClassifier.train(training,reload=False)
		newPoissonClassifier2.test(trainFlat,"training",crossVal=True);
		score=newPoissonClassifier2.test(holdOut,"test",crossVal=True);
		scoresSVM["allRegular"].append(score);
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

for s in scoresSVM.keys():
	scoresSVM[s] = np.mean(scoresSVM[s]);
print(scoresSVM);

exit()

scoresSVM["fifthWithEmbedding"]= [];
for percentile in range(len(foldsLastYear)):
	if percentile == 4:
		print(percentile);
	trainFlat, holdOut = flattenFolds(foldsLastYear, percentile)
	newPoissonClassifier.train(trainFlat,extraParamNum=0,wordEmbeddingMatrix=True)
	# poissonClassifier.train(training,reload=False)
	newPoissonClassifier.test(trainFlat,"training",crossVal=True);
	score=newPoissonClassifier.test(holdOut,"test",crossVal=True);
	scoresSVM["fifthWithEmbedding"].append(score);



scoresSVM["fifthSansEmbedding"]= [];
for percentile in range(len(foldsLastYear)):
	trainFlat, holdOut = flattenFolds(foldsLastYear, percentile)
	newPoissonClassifier.train(trainFlat,extraParamNum=0,wordEmbeddingMatrix=False)
	# poissonClassifier.train(training,reload=False)
	newPoissonClassifier.test(trainFlat,"training",crossVal=True);
	score=newPoissonClassifier.test(holdOut,"test",crossVal=True);
	scoresSVM["fifthSansEmbedding"].append(score);







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
		for percentile in range(nForTopNWordEmbeddings):
			try:
				wordEmbeddingsMarix[:, index + percentile] = topNVectors[percentile]
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