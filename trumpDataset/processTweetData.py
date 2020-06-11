from .basisFuncs import *



from .favs import getMeanFavCountPresTweets
from .media import extractMediaFromTweet

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
PYTHONHASHSEED = 0


from .stats import reduceDimensPCA
from .twitterApiConnection import *
from .part3funcs import *
import matplotlib.pyplot as plt

from .visualization import graphTwoDataSetsTogether, boxAndWhiskerForKeyWordsFavs





def analyzeCoverageOfNMostPopularWordsAndMMostPopularBigrams(words, bigrams, tweets):

	'''
	depricated! Replace plz
	:param words:
	:param bigrams:
	:param tweets:
	:return:
	'''
	intersections = {}
	wordIntersections = {}
	bigramIntersections = {}
	empties = 0
	for index, tweet in enumerate(tweets):
		tweetWords = tweet.split(" ")
		if len(tweetWords) == 0 or (len(tweetWords)==1 and len(tweetWords[0])==0):
			empties += 1
			wordIntersectionCount = 0
			bigramIntersectionCount=0
			intersectionCount=0
		else:
			bigramsTweet = computeBigrams(tweetWords)
			wordIntersectionCount = len(np.intersect1d(tweetWords,words))
			bigramIntersectionCount = len(np.intersect1d(bigramsTweet,bigrams))
			intersectionCount = wordIntersectionCount+bigramIntersectionCount
		try:
			intersections[intersectionCount] += 1
		except:
			intersections[intersectionCount] = 1
		try:
			wordIntersections[wordIntersectionCount] += 1
		except:
			wordIntersections[wordIntersectionCount] = 1



		try:
			bigramIntersections[bigramIntersectionCount] += 1
		except:
			bigramIntersections[bigramIntersectionCount] = 1

		# if index % 10 == 0:
		# 	print(index/len(tweets), intersectionCount)
	return intersections,wordIntersections,bigramIntersections,empties





def analyzeTopAndBottomPercentiles(nGrams = [1,2,3,4],tweets = None):
	tweets = getTweetsFromDB(purePres=True,orderBy="favCount asc", returnParams=["favCount","cleanedText"])
	topNGramsByPercentage = {}
	bottomNGramsByPercentage = {}

	for i in range(19,20,2):
		topNGramsForThisPercent = {n: [] for n in nGrams}
		bottomNGramsForThisPercent = {n: [] for n in nGrams}
		topNPercent, bottomNPercent = getTopBottomiPercentOfPurePresTweets(i,tweets)



		dataFrameDict = {}
		for n in nGrams:
			print("i: ",i, "n: ",n);
			computeMostCommonnGrams(topNPercent, n)
			computeMostCommonnGrams(bottomNPercent,n)

			tweetHashes = [hashTweets(topNPercent), hashTweets(bottomNPercent)]
			nMostPopularnGramsWithCountsTop, nMostPopularnGramsWithCountsBottom = getMMostLeastPopularnGramsWithCounts(
				25, n, tweetHashes =tweetHashes, percentage = True);
			topNGramsForThisPercent[n].append(nMostPopularnGramsWithCountsTop)
			bottomNGramsForThisPercent[n].append(nMostPopularnGramsWithCountsBottom)


			dataFrameDict[str(n)+"-grams for top "+str(i)+"%:"] = nMostPopularnGramsWithCountsTop
			dataFrameDict[str(n) + "-grams for bottom " + str(i) + "%:"] = nMostPopularnGramsWithCountsBottom
		minLen = min(min(len(l) for l in dataFrameDict.values()),50)
		for k in dataFrameDict.keys():
			dataFrameDict[k] = dataFrameDict[k][:minLen]
		topNGramsByPercentage[i] = topNGramsForThisPercent
		bottomNGramsByPercentage[i] = bottomNGramsForThisPercent
		df = pd.DataFrame(dataFrameDict)
		print(df)






'''
should I use CBOW or Skip Gram to compute vectors? I understand the difference but it doesn't seem obvoius to me which 
is better suited to this task
'''

