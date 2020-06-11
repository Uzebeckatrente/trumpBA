
from .basisFuncs import *
from .part3funcs import computePercentageAllCaps;
from .visualization import graphTwoDataSetsTogether;
from .stats import pearsonCorrelationCoefficient;
from .favs import getMedianFavCountPresTweets,getMeanFavCountPresTweets;

def populateDeletedColumn():
	mycursor.execute("select id from "+mainTable)
	updateFormula = "UPDATE "+mainTable+" SET deleted = %s WHERE id = %s";

	ids = [int(id[0]) for id in mycursor.fetchall()]
	tuples = [];
	print("iterations: ",int(len(ids)/100)+1)
	for i in range(int(len(ids)/100)+1):
		idPartition = ids[i*100:(i+1)*100];
		# idPartition = [ids[0],869766994899468288,ids[1]]
		responses = getTweetsById(idPartition);
		idIntsFromTwitter = [int(resp.id_str) for resp in responses]
		deletedInts = [int(id in idIntsFromTwitter) for id in idPartition];
		tuples.extend([(deletedInts[j],idPartition[j]) for j in range(len(deletedInts))])
		print("finished iteration ",i);
		print(list(zip(responses,deletedInts)))

	mycursor.executemany(updateFormula,tuples);
	print("executed")
	mydb.commit()

def populateIsReplyColumn():
	'''
	self: 2
	rando: 1
	None: 0
	TODO: fav-ness between in-chain tweets?
	:return:
	'''
	updateFormula = "UPDATE " + mainTable + " SET isReply = %s WHERE id = %s";
	tweets = [tweet[0] for tweet in getTweetsFromDB(purePres=False,conditions=["isReply = -1"],returnParams=["id"])]
	print("total of ", len(tweets), " tweets to update")
	tuples = []
	i=0;
	while True:
		tIds = tweets[i*100:(i+1)*100]
		if len(tIds) == 0:
			break;
		tObjs = getTweetsById(tIds);
		for t in tObjs:
			if t.in_reply_to_screen_name != None:
				if t.in_reply_to_screen_name.lower() == "realdonaldtrump":
					tuples.append((2,t.id))
				else:
					tuples.append((1,t.id))
			else:
				tuples.append((0,t.id));
		print(i/(len(tweets)/100))
		i+= 1
	mycursor.executemany(updateFormula,tuples);
	mydb.commit()


def getAllCapsSkewForAboveThreshold(threshhold = 0.7,trainingTweets = None):
	if trainingTweets == None:
		favs = [fav[0] for fav in getTweetsFromDB(purePres=True,conditions=["allCapsRatio > " + str(threshhold)], returnParams=["favCount"])];
	else:
		favs = [tweet[0] for tweet in trainingTweets if tweet[2] > threshhold];
	favs.sort();
	return favs[int(len(favs) / 2)];

def populateAllCapsPercentageColumn():

	tweets = getTweetsFromDB(conditions=["allCapsRatio = -1"],returnParams=["id","tweetText"])
	mycursor.execute("select id, tweetText from "+mainTable)
	updateFormula = "UPDATE "+mainTable+" SET allCapsRatio = %s WHERE id = %s";
	print("total of ",len(tweets)," tweets to update")
	tuples = []
	for index,tweet in enumerate(tweets):
		text = tweet[1];
		percentageCaps = computePercentageAllCaps(text);
		tuples.append((str(round(percentageCaps,2)),tweet[0]))
		if index%100==0:
			print(index/len(tweets))
	mycursor.executemany(updateFormula,tuples);
	mydb.commit()


	print("executed")
	mydb.commit()


def analyzeAllCapsPercentage():
	tweets = getTweetsFromDB(purePres=True,conditions=["allCapsRatio >= 0"],returnParams=["favCount, publishTime, allCapsRatio"]);
	# tweets = [tweet for tweet in tweets if tweet[2] > 0.5]
	tweets.sort(key=lambda tuple: tuple[2], reverse=False);
	overallMedian= getMedianFavCountPresTweets(tweets)
	overallMean = getMeanFavCountPresTweets(tweets)
	pearsons = []

	for numSlices in range(1,8):
		meanFavCountsByAllCapsRatioInterval = []
		medianFavCountsByAllCapsRatioInterval = []
		labels = []
		myIs = [i/numSlices for i in range(numSlices+1)]
		for i in range(len(myIs)-1):
			tweetsInRange = getTweetsFromDB(purePres=True, conditions=["allCapsRatio >= "+str(myIs[i]), "allCapsRatio <= "+str(myIs[i+1])],
									 returnParams=["favCount"]);

			labels.append("["+str(round(myIs[i],2))+ ","+str(round(myIs[i+1],2))+"]")
			mean = getMeanFavCountPresTweets(tweetsInRange);
			median = getMedianFavCountPresTweets(tweetsInRange);
			meanFavCountsByAllCapsRatioInterval.append(mean)
			medianFavCountsByAllCapsRatioInterval.append(median)


		x = myIs[:-1]

		# plt.bar(x, height=medianFavCountsByAllCapsRatioInterval)
		# plt.bar(x, height=medianFavCountsByAllCapsRatioInterval)
		fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		plt.xticks(x, labels)

		plt.scatter(x,medianFavCountsByAllCapsRatioInterval,label="medianFavCountsByAllCapsRatioInterval");
		plt.scatter(x, meanFavCountsByAllCapsRatioInterval, label="meanFavCountsByAllCapsRatioInterval");
		plt.plot([x[0],x[-1]],[overallMean,overallMean],label="mean")
		plt.plot([x[0], x[-1]], [overallMedian, overallMedian], label="median")

		pearsonMean = pearsonCorrelationCoefficient(x, meanFavCountsByAllCapsRatioInterval)
		pearsonMedian = pearsonCorrelationCoefficient(x, medianFavCountsByAllCapsRatioInterval)
		pearsons.append((pearsonMean,pearsonMedian));
		plt.legend()
		plt.title("pearson mean: "+str(pearsonMean)+"; pearson median: "+str(pearsonMedian))
		plt.show()
	for p in pearsons:
		print(p);
	exit()
	favCounts = [tweet[0] for tweet in tweets]
	publishTimes = [dateTimeToInt(tweet[1]) for tweet in tweets]
	allCapsRatio = [tweet[2] for tweet in tweets]

	plt.scatter(allCapsRatio, favCounts);
	pearson = pearsonCorrelationCoefficient(favCounts, allCapsRatio)
	plt.title(pearson)


	plt.show()

