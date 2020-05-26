from colorama import Fore, Style

import trumpBA
import time
# tweet = getTweetById(1194648339071016961)
# print(tweet)
# insertIntoDB(processTrumpTwitterArchiveFile());
# populateCleanedTextColumn()
# exit()
# appendToDB(processTrumpTwitterArchiveFile(fileName = "addition.csv"))
# exit()

# populateDeletedColumn()
# fixIsRt(processTrumpTwitterArchiveFile())
# processTrumpTwitterArchiveFile()
#
# graphFavsAndRtsByRatio()
# graphFavsByApprRating()
# analyzeSkewOfKeywords(False)

# favouriteOverTimeTrends()
# exit()
crawlFollowerCounts()
exit()
analyzeTopAndBottomPercentiles()
exit()
historgramFavCounts([favC[0] for favC in getTweetsFromDB(n=-1,conditions="purePres",returnParams=["favCount"])])
exit()
# graphPearsonsForApprFavOffset()
exit()
favouriteVsLengthTrends()
favouriteOverTimeTrends()
graphPearsonsForApprFavOffset()
# exit()
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
analyzeOverUnderMeanSkewOfKeywords(False)
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