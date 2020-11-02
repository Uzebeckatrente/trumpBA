
from .basisFuncs import *
from .part3funcs import computePercentageAllCapsAndUppersList;
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
	tweets = [tweet[0] for tweet in getTweetsFromDB(purePres=False,conditions=["isReply = 2"],returnParams=["id"])]
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
					tuples.append((t.in_reply_to_status_id,t.id))
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
		percentageCaps = computePercentageAllCapsAndUppersList(text);
		tuples.append((str(round(percentageCaps,2)),tweet[0]))
		if index%100==0:
			print(index/len(tweets))
	mycursor.executemany(updateFormula,tuples);
	mydb.commit()


	print("executed")
	mydb.commit()


def populateAllCapsWords():

	tweets = getTweetsFromDB(returnParams=["id","tweetText"])
	mycursor.execute("select id, tweetText from "+mainTable)
	updateFormula = "UPDATE "+mainTable+" SET allCapsWords = %s WHERE id = %s";
	print("total of ",len(tweets)," tweets to update")
	tuples = []
	for index,tweet in enumerate(tweets):
		text = tweet[1];
		_, uppers = computePercentageAllCapsAndUppersList(text);
		tuples.append((uppers,tweet[0]))
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

	for numSlices in [5]:
		meanFavCountsByAllCapsRatioInterval = []
		medianFavCountsByAllCapsRatioInterval = []
		boxSizesMean = []
		boxSizesMedian = []
		labels = []
		myIs = [0.0]+[i/numSlices for i in range(numSlices+1)]
		# myIs = [i / numSlices for i in range(numSlices + 1)]
		for i in range(len(myIs)-1):
			if i == 1:
				tweetsInRange = getTweetsFromDB(purePres=True, conditions=["allCapsRatio > "+str(myIs[i]), "allCapsRatio <= "+str(myIs[i+1])],
										 returnParams=["favCount"]);
				labels.append("(" + str(round(myIs[i], 2)) + "," + str(round(myIs[i + 1], 2)) + "]")
			else:
				tweetsInRange = getTweetsFromDB(purePres=True, conditions=["allCapsRatio >= "+str(myIs[i]), "allCapsRatio <= "+str(myIs[i+1])],
										 returnParams=["favCount"]);
				labels.append("[" + str(round(myIs[i], 2)) + "," + str(round(myIs[i + 1], 2)) + "]")


			mean = getMeanFavCountPresTweets(tweetsInRange);
			median = getMedianFavCountPresTweets(tweetsInRange);
			meanFavCountsByAllCapsRatioInterval.append(mean)
			medianFavCountsByAllCapsRatioInterval.append(median)
			boxSizesMean.append(len(tweetsInRange))
			boxSizesMedian.append(len(tweetsInRange));


		x = [i/numSlices for i in range(numSlices+1)]
		# x = myIs[:-1]


		# plt.bar(x, height=medianFavCountsByAllCapsRatioInterval)
		# plt.bar(x, height=medianFavCountsByAllCapsRatioInterval)
		# fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		# fig.subplots_adjust(left=0.06, right=0.94)
		plt.xticks(x, labels)

		plt.scatter(x,medianFavCountsByAllCapsRatioInterval,label="Median Fav Count for Tweets in All-caps Interval",sizes=boxSizesMedian,color="#FF0000");
		plt.scatter(x, meanFavCountsByAllCapsRatioInterval, label="Mean Fav Count for Tweets in All-caps Interval",sizes=boxSizesMean,color="#0000FF");
		plt.plot([x[0],x[-1]],[overallMean,overallMean],label="Mean Favourite Count for All Tweets",color="#00FF00")
		plt.plot([x[0], x[-1]], [overallMedian, overallMedian], label="Median Favourite Count for All Tweets",color="#000000")

		pearsonMean = pearsonCorrelationCoefficient(x, meanFavCountsByAllCapsRatioInterval)
		pearsonMedian = pearsonCorrelationCoefficient(x, medianFavCountsByAllCapsRatioInterval)

		fitLineCoefficientsMedian = np.polyfit([i for i in range(len(medianFavCountsByAllCapsRatioInterval))], medianFavCountsByAllCapsRatioInterval, deg=1, full=False)
		slope = fitLineCoefficientsMedian[0];
		intercept = fitLineCoefficientsMedian[1]
		plt.plot([x[0], x[-1]],[slope * i + intercept for i in [0,len(medianFavCountsByAllCapsRatioInterval)]],color="#FF0000",linestyle='dashed')

		fitLineCoefficientsMean = np.polyfit([i for i in range(len(meanFavCountsByAllCapsRatioInterval))], meanFavCountsByAllCapsRatioInterval, deg=1, full=False)
		slope = fitLineCoefficientsMean[0];
		intercept = fitLineCoefficientsMean[1]
		plt.plot([x[0], x[-1]], [slope * i + intercept for i in [0, len(meanFavCountsByAllCapsRatioInterval)]], color="#0000FF",linestyle='dashed')


		pearsons.append((pearsonMean,pearsonMedian));
		thaLegend = plt.legend(fontsize=15)
		for handle in thaLegend.legendHandles:
			handle._sizes = [100]
		plt.title("Tweets With High Percentage of Capital Letters Receive More Favourites", fontsize=25)
		plt.xlabel("Percentage of Words Capitalized in Tweet",fontsize=20)
		plt.ylabel("Average Favourite Count",fontsize=20)
		plt.xticks(fontsize=18)
		plt.yticks(fontsize=18)
		print("boxSizesMean: ",boxSizesMean," boxSizesMedian: ",boxSizesMedian)
		print(boxSizesMean[0]/np.sum(boxSizesMean))
		print(boxSizesMedian[0]/np.sum(boxSizesMean))
		print((boxSizesMedian[0]+boxSizesMedian[1])/np.sum(boxSizesMean))
		print((boxSizesMean[0]+boxSizesMean[1])/np.sum(boxSizesMean))
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

def analyzeShortTweetPercentage():
	tweets = getTweetsFromDB(purePres=True,returnParams=["favCount, publishTime, tweetText"]);
	# tweets = [tweet for tweet in tweets if tweet[2] > 0.5]
	empties = [t for t in tweets if t[2] == ""]
	rest = [t for t in tweets if t[2] != ""]
	favCounts = [t[0] for t in tweets]
	rest.sort(key=lambda tuple: len(tuple[2].split(" ")), reverse=False);
	print(len(empties),len(rest))
	tweets = empties + rest;
	overallMedian= getMedianFavCountPresTweets(tweets)
	overallMean = getMeanFavCountPresTweets(tweets)
	pearsons = []

	for numSlices in [5]:
		meanFavCountsByAllCapsRatioInterval = []
		medianFavCountsByAllCapsRatioInterval = []
		boxSizesMean = []
		boxSizesMedian = []
		labels = []
		myIs = list(range((np.min([len(tup[2].split(" ")) for tup in tweets])),np.max([len(tup[2].split(" ")) for tup in tweets])))
		myIsComp = [0,3,5,8,14,np.max([len(tup[2].split(" ")) for tup in tweets])]
		# myIs = [i / numSlices for i in range(numSlices + 1)]
		for i in range(len(myIsComp[:-1])):
		# for i in myIs:
			tweetsInRange = [t for t in tweets if myIsComp[i] <= len(t[2].split(" ")) < myIsComp[i+1]]
			labels.append("[" + str(round(max(myIsComp[i],0), 2)) + "," + str(round(myIsComp[i + 1], 2)) + ")")


			mean = getMeanFavCountPresTweets(tweetsInRange);
			median = getMedianFavCountPresTweets(tweetsInRange);
			meanFavCountsByAllCapsRatioInterval.append(mean)
			medianFavCountsByAllCapsRatioInterval.append(median)
			boxSizesMean.append(len(tweetsInRange))
			boxSizesMedian.append(len(tweetsInRange));


		x = [i for i in range(len(labels))]
		# x = myIs[:-1]


		# plt.bar(x, height=medianFavCountsByAllCapsRatioInterval)
		# plt.bar(x, height=medianFavCountsByAllCapsRatioInterval)
		# fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		# fig.subplots_adjust(left=0.06, right=0.94)
		plt.xticks(x, labels)

		plt.scatter(x,medianFavCountsByAllCapsRatioInterval,label="Median Favourite Count for Tweets in Length Interval",sizes=boxSizesMedian,color="#FF0000");
		plt.scatter(x, meanFavCountsByAllCapsRatioInterval, label="Mean Favourite Count for Tweets in Length Interval",sizes=boxSizesMean,color="#0000FF");
		plt.plot([x[0],x[-1]],[overallMean,overallMean],label="Mean Favourite Count for All Tweets",color="#00FF00")
		plt.plot([x[0], x[-1]], [overallMedian, overallMedian], label="Median Favourite Count for All Tweets",color="#000000")

		pearsonMean = pearsonCorrelationCoefficient(x, meanFavCountsByAllCapsRatioInterval)
		pearsonMedian = pearsonCorrelationCoefficient(x, medianFavCountsByAllCapsRatioInterval)

		fitLineCoefficientsMedian = np.polyfit([i for i in range(len(medianFavCountsByAllCapsRatioInterval))], medianFavCountsByAllCapsRatioInterval, deg=1, full=False)
		slope = fitLineCoefficientsMedian[0];
		intercept = fitLineCoefficientsMedian[1]
		plt.plot([x[0], x[-1]],[slope * i + intercept for i in [x[0],x[-1]]],color="#FF0000",linestyle='dashed',)

		fitLineCoefficientsMean = np.polyfit([i for i in range(len(meanFavCountsByAllCapsRatioInterval))], meanFavCountsByAllCapsRatioInterval, deg=1, full=False)
		slope = fitLineCoefficientsMean[0];
		intercept = fitLineCoefficientsMean[1]
		plt.plot([x[0], x[-1]], [slope * i + intercept for i in [x[0],x[-1]]], color="#0000FF",linestyle='dashed',)


		pearsons.append((pearsonMean,pearsonMedian));
		thaLegend = plt.legend(fontsize=18)
		for handle in thaLegend.legendHandles:
			handle._sizes = [100]
		plt.title("Short Tweets are More Popular",fontsize=25)
		plt.ylabel("Average Favourite Count",fontsize=20)
		plt.xlabel("Tweets of Length", fontsize=20)
		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)
		print("boxSizesMean: ",boxSizesMean," boxSizesMedian: ",boxSizesMedian)
		print(boxSizesMean[0]/np.sum(boxSizesMean))
		print(boxSizesMedian[0]/np.sum(boxSizesMean))
		print(boxSizesMean[-1] / np.sum(boxSizesMean))
		print(boxSizesMedian[-1] / np.sum(boxSizesMean))
		print((boxSizesMedian[0]+boxSizesMedian[1])/np.sum(boxSizesMean))
		print((boxSizesMean[0]+boxSizesMean[1])/np.sum(boxSizesMean))
		plt.ylim(int(min(meanFavCountsByAllCapsRatioInterval[-1],meanFavCountsByAllCapsRatioInterval[-1])-20000),int(max(meanFavCountsByAllCapsRatioInterval[0],meanFavCountsByAllCapsRatioInterval[0])+5000))
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

