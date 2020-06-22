from colorama import Fore, Style

# import trumpBA.trumpDataset;
try:
	from trumpDataset import *
except:
	from trumpBA.trumpDataset import *;
# import trumpBA.trumpDataset
import time

# graphFavCountLogRound()
# tweet = getTweetById(1194648339071016961)
# print(tweet)
# insertIntoDB(processTrumpTwitterArchiveFile());
# populateCleanedTextColumn()
# exit()
# exit()

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
# graphPearsonsForApprFavOffset()

# favouriteVsLengthTrends()
# favouriteOverTimeTrends()
# graphPearsonsForApprFavOffset()
# populateCleanedTextColumn()

#
# analyzeTopAndBottomPercentiles()
# compareVocabOverTime(18)
# for epochCount in range(3,101,3):
# 	compareVocabOverTime(epochCount)
# # compareVocabOverTime(5)
# exit()


np.random.seed(69)
allPresTweetsFavCountAndCleanedText = getTweetsFromDB(purePres=True,conditions=["isReply = 0"],returnParams=["favCount","cleanedText, allCapsRatio, mediaType"], orderBy= "publishTime desc")

indices = np.random.permutation(len(allPresTweetsFavCountAndCleanedText))
numFolds = 5;
foldsIndices = [indices[int(len(allPresTweetsFavCountAndCleanedText)*(0.2*i)):int(len(allPresTweetsFavCountAndCleanedText)*(0.2*i+1))] for i in range(numFolds)]
training_idx, test_idx = indices[:int(len(allPresTweetsFavCountAndCleanedText)*0.8)], indices[int(len(allPresTweetsFavCountAndCleanedText)*0.8):]
training = []
test = []
folds = []
for indices in foldsIndices:
	fold = []
	for i in indices:
		fold.append(allPresTweetsFavCountAndCleanedText[i]);
	folds.append(fold);

for i in range(len(allPresTweetsFavCountAndCleanedText)):
	if len(allPresTweetsFavCountAndCleanedText[i][1])>0:
		if i in training_idx:
			training.append(allPresTweetsFavCountAndCleanedText[i])
		else:
			test.append(allPresTweetsFavCountAndCleanedText[i])
print(Fore.LIGHTRED_EX,str(len(training))," training tweets; ",str(len(test))," test tweets",Style.RESET_ALL);

allCapsClassifier = AllCapsClassifier()
mlpClassifier = MLPPopularity();
poissonClassifier = LogRegressionSlashPoisson();
olsClassifier = OlsTry()
mlpClassifier.train(training)
# poissonClassifier.train(training,reload=False)
mlpClassifier.test(training);
mlpClassifier.test(test);
exit()


# allCapsClassifier.test(test,title="test data");
#
# allCapsClassifier.train(training);
# misClassifiedsAllCaps =allCapsClassifier.test(test,training=False,title="tist data")[0];
# misClassifiedsAllCaps.sort(key=lambda tup: tup[0]);
# #
# naiveBayesClassifier = PureBayesClassifier()
# naiveBayesClassifier.train(training,testData = test,retabulate = False)
# naiveBayesClassifier.predict("peaceofficersmemorialday")
# print(Fore.CYAN,"test error: ")
# misClassifiedNaive = naiveBayesClassifier.test(test, title="naive bayes baybayy")
# print(Fore.LIGHTRED_EX,"train error: ")
# naiveBayesClassifier.test(training, title="training naive")
# #
#
#


booster = MixedBoostingBayesClassifier()

booster.train(training,testData = test,retabulate = False)



print(Fore.CYAN,"test error onlyLen True: ")
misClassifiedsBooster = booster.test(test,title="post")

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
'''