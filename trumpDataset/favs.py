

from .basisFuncs import *
from .part3funcs import getMostPopularWordsOverCountNMostPopularBigramsOverCount, computeMostCommonnGrams, \
	getnGramsWithOverMOccurences,extractNGramsFromCleanedText
from .stats import pearsonCorrelationCoefficient
from .visualization import boxAndWhiskerForKeyWordsFavs, computeXandYForPlotting,graphTwoDataSetsTogether;


def getMedianFavCountPresTweets(favs = None):
	if favs == None:
		favs = getTweetsFromDB(purePres=True,returnParams=["favCount"],orderBy="favCount")
	favs.sort(key=lambda tup: tup[0])
	median = favs[int(len(favs)/2)][0];
	return median


def getMeanFavCountPresTweets(favs = None):
	if favs == None:
		query = "SELECT avg(favCount) FROM trump.tta2 "+relevancyDateThreshhold+" and isRt = 0"
		mycursor.execute(query)
		fvc = mycursor.fetchone()[0]
		return int(fvc)
	else:
		favs = [fav[0] for fav in favs]
		return np.mean(favs);



def loadnGramsWithFavsMinusMeanMedian(ns, tweetsHash):
	'''

	:param ns:
	:param tweetsHash:
	:return: [(n-gram count, skew, stdev, favs),...]
	'''
	wordsAndBigramsWithFavsMinusMeanFile = open(
		"trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsMinusMean" + tweetsHash + ".p", "rb")
	nGramsWithFavsMinusMean = pickle.load(wordsAndBigramsWithFavsMinusMeanFile);
	wordsAndBigramsWithFavsMinusMeanFile.close()

	wordsAndBigramsWithFavsMinusMedianFile = open(
		"trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsMinusMedian" + tweetsHash + ".p", "rb")
	nGramsWithFavsMinusMedian = pickle.load(wordsAndBigramsWithFavsMinusMedianFile);
	wordsAndBigramsWithFavsMinusMedianFile.close()

	return nGramsWithFavsMinusMean,nGramsWithFavsMinusMedian

def loadnGramsWithFavsProbabilities(ns,numBins, tweetsHash):
	'''

	:param ns:
	:param tweetsHash:
	:return: [(n-gram count, skew, stdev, favs),...]
	'''
	wordsAndBigramsWithFavsProbabilitiesFile = open(
		"trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsProbabilities" +str(numBins)+ tweetsHash + ".p", "rb")
	nGramsWithFavsProbabilities = pickle.load(wordsAndBigramsWithFavsProbabilitiesFile);
	wordsAndBigramsWithFavsProbabilitiesFile.close()


	return nGramsWithFavsProbabilities


def calculateAndStorenGramsFavsProbabilities(ns,tweets,numBins=2, retabulate = False):
	'''
	calculates, for each ngram, P(bin_i | nGram)
	:param ns:
	:param tweets:
	:param bins:
	:param retabulate:
	:return:
	'''
	tweetsHash = hashTweets(tweets);
	if numBins != 2:
		raise Exception("not yet implemented lol")

	if not retabulate:
		try:
			f=open("trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsProbabilities" +str(numBins)+ tweetsHash + ".p",'rb');
			f.close()
			print("not retabulating bin probabilities; run again with retabulate = True to retabulate")
			return tweetsHash
		except:
			pass
	print(Fore.MAGENTA,"retabulating probabilities",Style.RESET_ALL)


	startingTime = time.time()
	fvcMedian = getMedianFavCountPresTweets(tweets)

	nGrams = []

	nGramsWithBinProbs = {}
	totalNGramsProbs = {bin: 0 for bin in range(numBins)}
	for n in ns:
		computeMostCommonnGrams(tweets, n);
		myNGrams = getnGramsWithOverMOccurences(n, 5, tweetsHash)
		nGrams.extend(myNGrams)
		for nGram in nGrams:
			nGramsWithBinProbs[nGram[0]] = [0]*numBins


	positives = 0;
	totals = 0;
	for index, tweet in enumerate(tweets):
		cleanedText = tweet[1]
		realFavCount = tweet[0]
		if numBins == 2:
			bin = int(realFavCount>fvcMedian);
			positives += bin;
			totals += 1;
		else:
			raise Exception("not implemented yet lol")
		nGrams = extractNGramsFromCleanedText(cleanedText, ns);
		for nGram in nGrams:
			if nGram in nGramsWithBinProbs:
				nGramsWithBinProbs[nGram][bin] += 1
				totalNGramsProbs[bin] += 1;
		if index % 100 == 0:
			print(Fore.RED, index / len(tweets), Style.RESET_ALL)

	print("ratio positive: ",positives/totals)
	for bin in range(numBins):
		for nGram in nGramsWithBinProbs.keys():
			nGramsWithBinProbs[nGram][bin] /= totalNGramsProbs[bin]


	with open("trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsProbabilities" +str(numBins)+ tweetsHash + ".p",
			  'wb') as wordsAndBigramsWithFavsMinusMeanFile:
		pickle.dump(nGramsWithBinProbs, wordsAndBigramsWithFavsMinusMeanFile)
		wordsAndBigramsWithFavsMinusMeanFile.close()
	print(Fore.MAGENTA, time.time() - startingTime, Style.RESET_ALL);
	return tweetsHash


def updateFavCounts():
	updateFormula = "UPDATE "+mainTable+" SET favCount = %s WHERE id = %s";

	tweets = getTweetsFromDB(n=-1,purePres=False,conditions=[],returnParams=["id","favCount"]);

	counter = 0;
	diffs=[]
	oldDiffs = []
	tuples = []
	while True:
		print(counter*100/len(tweets))
		hundredTweets = tweets[counter*100:(counter+1)*100]#[tweet for tweet in corpus[counter * 100:(counter + 1) * 100]];
		hundredTweetDict = {}
		for t in hundredTweets:
			hundredTweetDict[t[0]] = t[1]
		if len(hundredTweets) == 0:
			break;
		hundredTweetIds = [t[0] for t in hundredTweets]
		# hundredTweetIds = [1247641140507049986]#,1257656121898283010, 1257736426206031874,1257753237790158848, 1257714716492722176, 1194615176823218176]
		retTweets = getTweetsById(hundredTweetIds)
		returnedIds = [tweet.id for tweet in retTweets]


		for retTweet in retTweets:
			id = retTweet.id;
			newFavCount = retTweet.favorite_count;
			prevFavCount = hundredTweetDict[id];
			oldDiffs.append(newFavCount-prevFavCount);
			diffs.append((str(id),str(newFavCount),"diff: ",newFavCount-prevFavCount));
			tuples.append((str(id),str(newFavCount)));
		counter += 1;

	for diff in diffs:
		print(diff);
	plt.scatter(list(range(len(oldDiffs))),oldDiffs)
	plt.show()

	toLog = str(input("should I log this tho bro me"))
	if toLog == "yes":
		mycursor.executemany(updateFormula,tuples);
		mydb.commit()




def calculateAndStorenGramsWithFavsMinusMeanMedian(ns,tweets, retabulate = False):
	tweetsHash = hashTweets(tweets);

	if not retabulate:
		try:
			f=open("trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsMinusMean" + tweetsHash + ".p",'rb');
			f.close()
			f=open("trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsMinusMean" + tweetsHash + ".p", 'rb');
			f.close()
			print("not retabulating mean/median skews; run again with retabulate = True to retabulate")
			return tweetsHash
		except:
			pass
	print(Fore.MAGENTA,"retabulating mean/median skews",Style.RESET_ALL)


	startingTime = time.time()
	fvcMean = getMeanFavCountPresTweets(tweets)
	fvcMedian = getMedianFavCountPresTweets(tweets)

	nGrams = []
	for n in ns:
		computeMostCommonnGrams(tweets, n);
		myNGrams = getnGramsWithOverMOccurences(n, 5, tweetsHash)
		nGrams.extend(myNGrams)
	favsList = []
	keywords = []

	nGramsWithFavsMinusMean = []
	nGramsWithFavsMinusMedian = []

	for index, nGram in enumerate(nGrams):
		favs = getFavsByNGram(nGram[0], rts=False);
		medianFavs = favs[int(len(favs) / 2)]
		meanFavs = np.mean(favs)
		###not the same length because the source words contain rts and also possiblty multiple counts per tweet
		favsList.append(favs);
		keywords.append(nGram[0]);
		nGramsWithFavsMinusMean.append((nGram[0], nGram[1], np.std(favs),meanFavs - fvcMean, favs));
		# nGramsWithFavsMinusMeanDict[nGram[0]] = (str(nGram[1]), meanFavs - fvcMean,favs);
		nGramsWithFavsMinusMedian.append((nGram[0], nGram[1], np.std(favs), medianFavs - fvcMedian, favs));
		# nGramsWithFavsMinusMedianDict[nGram[0]] = (str(nGram[1]), medianFavs - fvcMedian, favs);
		if index % 100 == 0:
			print(Fore.RED, index / len(nGrams), Style.RESET_ALL)

	print("collected: ", len(nGramsWithFavsMinusMean), " grams!")

	print("percent skews above median: ", len([x for x in nGramsWithFavsMinusMedian if x[nGramStorageIndicesSkew["skew"]] >= 0]) / len(nGramsWithFavsMinusMedian))

	nGramsWithFavsMinusMean.sort(key=lambda tup: np.fabs(tup[nGramStorageIndicesSkew["skew"]]), reverse=True)
	nGramsWithFavsMinusMedian.sort(key=lambda tup: np.fabs(tup[nGramStorageIndicesSkew["skew"]]), reverse=True)

	with open("trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsMinusMean" + tweetsHash + ".p",
			  'wb') as wordsAndBigramsWithFavsMinusMeanFile:
		pickle.dump(nGramsWithFavsMinusMean, wordsAndBigramsWithFavsMinusMeanFile)
		wordsAndBigramsWithFavsMinusMeanFile.close()
	with open("trumpBA/trumpDataset/npStores/" + str(ns) + "gramsWithFavsMinusMedian" + tweetsHash + ".p",
			  'wb') as wordsAndBigramsWithFavsMinusMedianFile:
		pickle.dump(nGramsWithFavsMinusMedian, wordsAndBigramsWithFavsMinusMedianFile)
		wordsAndBigramsWithFavsMinusMedianFile.close()
	print(Fore.MAGENTA, time.time() - startingTime, Style.RESET_ALL);
	return tweetsHash

def analyzeOverUnderMeanSkewOfKeywords(tweets = "purePres", loadFromStorage = True,ns = [1,2,3,4]):
	fvcMean = getMeanFavCountPresTweets()
	fvcMedian = getMedianFavCountPresTweets()

	if tweets == "purePres":
		tweets = getTweetsFromDB(purePres=True,returnParams=["favCount","cleanedText"])
	tweetsHash = hashTweets(tweets)


	if loadFromStorage:
		try:
			nGramsWithFavsMinusMean, nGramsWithFavsMinusMedian = loadnGramsWithFavsMinusMeanMedian(ns, tweetsHash)

		except:
			print("extracting from storage failed; retabulating.")
			loadFromStorage = False

	if not loadFromStorage:
		tweetsHash = calculateAndStorenGramsWithFavsMinusMeanMedian(ns, tweets,retabulate=True);
		nGramsWithFavsMinusMean, nGramsWithFavsMinusMedian = loadnGramsWithFavsMinusMeanMedian(ns, tweetsHash)



	# nGramsWithFavsMinusMean = nGramsWithFavsMinusMean[-20:]
	# nGramsWithFavsMinusMedian = nGramsWithFavsMinusMedian[-20:]

	dataFrameDict = {}

	dataFrameDict["n-grams furthest from mean from ds mean :"] = [nGram[nGramStorageIndicesSkew["nGram"]] + "; count:" + str(nGram[nGramStorageIndicesSkew["count"]]) + "; skew: " + str(nGram[nGramStorageIndicesSkew["skew"]]) for nGram in nGramsWithFavsMinusMean[:40]]
	dataFrameDict["n-grams furthest median from ds median :"] = [nGram[nGramStorageIndicesSkew["nGram"]] + "; " + str(nGram[nGramStorageIndicesSkew["count"]]) for nGram in nGramsWithFavsMinusMedian[:40]]

	df = pd.DataFrame(dataFrameDict)
	print(df)
	meanXs = [tup[nGramStorageIndicesSkew["nGram"]].replace(" ", "\n") + "\n" + str(tup[nGramStorageIndicesSkew["count"]]) for tup in nGramsWithFavsMinusMean[:20]]
	medianXs = [tup[nGramStorageIndicesSkew["nGram"]].replace(" ", "\n") + "\n" + str(tup[nGramStorageIndicesSkew["count"]]) for tup in nGramsWithFavsMinusMedian[:20]]
	meanYs = [tup[nGramStorageIndicesSkew["favs"]] for tup in nGramsWithFavsMinusMean[:20]]
	medianYs = [tup[nGramStorageIndicesSkew["favs"]] for tup in nGramsWithFavsMinusMedian[:20]];
	boxAndWhiskerForKeyWordsFavs(meanXs,meanYs , fvcMean, title="meansDifference")
	boxAndWhiskerForKeyWordsFavs(medianXs, medianYs, fvcMedian, title="mediansDifference")



def favouriteVsLengthTrends():
	tweets = getTweetsFromDB(orderBy="publishTime asc", conditions=["isRt = 0", "president", "deleted = 0", "length(tweetText) > 1"], returnParams=["tweetText","favCount"])
	tweetFavCounts = [int(t[1]) for t in tweets];
	tweetLengths = [len(t[0]) for t in tweets]
	tweetFavCounts = np.array(tweetFavCounts)
	tweetLengths = np.array(tweetLengths)

	indices = np.argsort(tweetLengths)
	tweetFavCounts = tweetFavCounts[indices]
	tweetLengths = tweetLengths[indices]
	pearson = pearsonCorrelationCoefficient(tweetFavCounts, tweetLengths)
	plt.plot(tweetLengths,tweetFavCounts);
	plt.title(pearson)
	plt.show()

def graphMediaByFavCount():
	photoTweets = [t[0] for t in getTweetsFromDB(orderBy="favCount asc", purePres=True, returnParams=["favCount"],conditions=["mediaType = \"photo\""])]
	articleTweets = [t[0] for t in getTweetsFromDB(orderBy="favCount asc", purePres=True, returnParams=["favCount"],conditions=["mediaType = \"articleLink\""])]
	videoTweets = [t[0] for t in getTweetsFromDB(orderBy="favCount asc", purePres=True, returnParams=["favCount"], conditions=["mediaType like \"video#%\""])]
	noMediaTweets = [t[0] for t in getTweetsFromDB(orderBy="favCount asc", purePres=True, returnParams=["favCount"], conditions=["mediaType = \"none\""])]
	interLen = 0;
	colors = ["r","g","b","y"]
	plt.scatter([i +interLen for i in range(len(photoTweets))], photoTweets,label="photo tweets",c=colors[0])
	plt.plot([interLen,interLen+len(photoTweets)],[np.median(photoTweets),np.median(photoTweets)],c=colors[0])
	interLen += len(photoTweets);
	plt.scatter([i +interLen for i in range(len(articleTweets))], articleTweets,label="article tweets",c=colors[1])
	plt.plot([interLen, interLen + len(articleTweets)], [np.median(articleTweets), np.median(articleTweets)],c=colors[1])
	interLen += len(articleTweets);
	plt.scatter([i+interLen for i in range(len(videoTweets))], videoTweets,label="video tweets",c=colors[2])
	plt.plot([interLen, interLen + len(videoTweets)], [np.median(videoTweets), np.median(videoTweets)],c=colors[2])
	interLen += len(videoTweets);
	plt.scatter([i +interLen for i in range(len(noMediaTweets))], noMediaTweets,label="no media tweets",c=colors[3])
	plt.plot([interLen, interLen + len(noMediaTweets)], [np.median(noMediaTweets), np.median(noMediaTweets)],c=colors[3])
	plt.title("Favourite Counts Based on Media contained in Tweets")
	plt.legend();
	plt.show();

	bestPearson = -1
	bestDaysPerMonthCount = -1;
	presidentDate = None;
	electionDate = None;
	for daysPerMonthCount in [7]:  # ,2,5,7,14,21,31,62]:#int(np.exp2(i)) for i in range(10)]:
		plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(
			tweets, daysPerMonth=daysPerMonthCount);

		fitLineCoefficients = np.polyfit([i for i in range(len(months))], avgFavCounts, 1, full=False)
		slope = fitLineCoefficients[0];
		intercept = fitLineCoefficients[1]
		pearson = pearsonCorrelationCoefficient([slope * i + intercept for i in range(len(months))], avgFavCounts)
		if pearson > bestPearson:
			bestPearson = pearson
			bestDaysPerMonthCount = daysPerMonthCount
	# plt.plot([i for i in range(len(months))], avgFavCounts, label="favCounts")
	# plt.plot([i for i in range(len(months))], [slope * i + intercept for i in range(len(months))], label="ols")
	# plt.title("fav counts grouped by "+str(bestDaysPerMonthCount)+" days; pearson correlation coefficient: "+str(pearson))
	# plt.legend()
	# plt.show()

	for index, month in enumerate(months):
		if month > datetime.datetime(day=20, month=2, year=2017):
			presidentDate = index;
			break;
	for index, month in enumerate(months):
		if month > datetime.datetime(day=8, month=1, year=2016):
			electionDate = index;
			break;
	# plt.yscale("log")
	plt.plot([presidentDate, presidentDate], [0, max(avgFavCounts)], label="Inauguration");
	plt.plot([electionDate, electionDate], [0, max(avgFavCounts)], label="Election");
	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(tweets, daysPerMonth=bestDaysPerMonthCount);

	showEveryNthDate = 2;
	newMonths = months[1:]
	newMonths[0] = str(newMonths[0])[:10]
	newDates = months.copy()

	while len(newDates) - newDates.count("") > 15:
		newDates = [str(date - datetime.timedelta(days=1))[:10] if index % showEveryNthDate == showEveryNthDate - 1 else "" for index, date in enumerate(newMonths[1:])];
		showEveryNthDate += 1;
		print("n00b")
	newMonths[1:] = newDates
	plt.xticks([i for i in range(len(newMonths))], newMonths);
	firstPublishTime = tweets[0][1]

	fitLineCoefficients = np.polyfit([i for i in range(len(months))], avgFavCounts, 1, full=False)
	slope = fitLineCoefficients[0];
	intercept = fitLineCoefficients[1]
	plt.plot([i for i in range(len(months))], avgFavCounts, label="favCounts")
	plt.plot([i for i in range(len(months))], [slope * i + intercept for i in range(len(months))], label="ols")
	plt.title("fav counts grouped by " + str(bestDaysPerMonthCount) + " days; pearson correlation coefficient: " + str(round(bestPearson, 2)))
	plt.legend()
	plt.show()


def favouriteOverTimeTrends():
	tweets = getTweetsFromDB(orderBy="publishTime asc",purePres=False,conditions=["deleted = 0","isRt = 0"])
	bestPearson = -1
	bestDaysPerMonthCount = -1;
	presidentDate = None;
	electionDate = None;
	for daysPerMonthCount in [7]:#,2,5,7,14,21,31,62]:#int(np.exp2(i)) for i in range(10)]:
		plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
		avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(
			tweets, daysPerMonth=daysPerMonthCount);

		fitLineCoefficients = np.polyfit([i for i in range(len(months))], avgFavCounts, 1, full=False)
		slope = fitLineCoefficients[0];
		intercept = fitLineCoefficients[1]
		pearson = pearsonCorrelationCoefficient([slope * i + intercept for i in range(len(months))], avgFavCounts)
		if pearson > bestPearson:
			bestPearson = pearson
			bestDaysPerMonthCount = daysPerMonthCount
		# plt.plot([i for i in range(len(months))], avgFavCounts, label="favCounts")
		# plt.plot([i for i in range(len(months))], [slope * i + intercept for i in range(len(months))], label="ols")
		# plt.title("fav counts grouped by "+str(bestDaysPerMonthCount)+" days; pearson correlation coefficient: "+str(pearson))
		# plt.legend()
		# plt.show()

	for index, month in enumerate(months):
		if month > datetime.datetime(day=20,month=2,year=2017):
			presidentDate = index;
			break;
	for index, month in enumerate(months):
		if month > datetime.datetime(day=8,month=1,year=2016):
			electionDate = index;
			break;
	# plt.yscale("log")
	plt.plot([presidentDate,presidentDate],[0,max(avgFavCounts)],label="Inauguration");
	plt.plot([electionDate, electionDate], [0, max(avgFavCounts)], label="Election");
	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(tweets, daysPerMonth=bestDaysPerMonthCount);

	showEveryNthDate = 2;
	newMonths = months[1:]
	newMonths[0] = str(newMonths[0])[:10]
	newDates = months.copy()

	while len(newDates) - newDates.count("") > 15:
		newDates = [str(date-datetime.timedelta(days=1))[:10] if index % showEveryNthDate == showEveryNthDate - 1 else "" for index, date in enumerate(newMonths[1:])];
		showEveryNthDate += 1;
		print("n00b")
	newMonths[1:] = newDates
	plt.xticks([i for i in range(len(newMonths))], newMonths);
	firstPublishTime = tweets[0][1]


	fitLineCoefficients = np.polyfit([i for i in range(len(months))], avgFavCounts, 1, full=False)
	slope = fitLineCoefficients[0];
	intercept = fitLineCoefficients[1]
	plt.plot([i for i in range(len(months))],avgFavCounts,label="favCounts")
	plt.plot([i for i in range(len(months))],[slope*i+intercept for i in range(len(months))],label="ols")
	plt.title("fav counts grouped by "+str(bestDaysPerMonthCount)+" days; pearson correlation coefficient: "+str(round(bestPearson,2)))
	plt.legend()
	plt.show()

def graphPearsonsForApprFavOffset():
	ratings = getApprovalRatings()
	ratings = ratings[100:]
	pearsons = []
	dateRange = range(-100,100)
	for i in dateRange:
		tweets = getTweetsFromDB(orderBy="publishTime asc",
								 conditions=["isRt = 0","deleted = 0", "publishTime >= \"" + str(ratings[0][0]+datetime.timedelta(days=i)) + "\""])

		avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(
			tweets, daysPerMonth=1);
		# print("m0: ", months[0], " r0: ", ratings[0][0])
		print(str(ratings[0][0] + datetime.timedelta(days=i)))
		minLen = min(len(months), len(ratings))



		favsStandardized = avgFavCounts[:minLen];
		ratingsStandardized = [rat[1] for rat in ratings[:minLen]]
		pearson = pearsonCorrelationCoefficient(favsStandardized, ratingsStandardized);
		pearsons.append(pearson);

	maxOffsetIndex = np.argmax(pearsons)

	maxOffSetDay = ratings[0][0]+datetime.timedelta(days=dateRange[maxOffsetIndex])
	plt.plot([i for i in dateRange],pearsons);
	plt.scatter(dateRange[maxOffsetIndex],pearsons[maxOffsetIndex], color = "r", marker="x")
	plt.title("correlation coefficients for approval rating and favCount, offset by variable number of days")
	plt.show()

def graphFavCount():
	tweets = getTweetsFromDB(purePres=True,returnParams=["favCount"],orderBy="favCount asc");
	plt.plot([i for i in range(len(tweets))], [t[0] for t in tweets])
	plt.title("FavCount for all presidential tweets")
	plt.show();


def graphFavCountLogRound():

	favCounts = getTweetsFromDB(purePres=True, returnParams=["favCount", "mediaType", "isReply"], orderBy="favCount asc");

	favCountsPure = [np.log(fc[0]) for fc in favCounts if (fc[1] == "none" and fc[2] == 0)];
	favCounts = [np.log(fc[0]) for fc in favCounts]

	std = np.std(favCounts);
	mean = np.mean(favCounts);
	meanPure = np.mean(favCountsPure);
	stdPure = np.std(favCountsPure);
	# favCounts = [c[0] for c in favCounts]
	# favCounts = [np.log(c[0]) for c in favCounts]

	zScores = [(fc - mean)/std for fc in favCounts]
	zScoresPure = [(fc - meanPure) / stdPure for fc in favCountsPure]
	zScoresAboveNeg2 = [z for z in zScores if z>=-2];
	zScoresAboveNeg2Pure = [z for z in zScoresPure if z >= -2];
	print("percentage z scores above -2: ",len(zScoresAboveNeg2)/len(zScores))
	print("percentage pure z scores above -2: ", len(zScoresAboveNeg2Pure) / len(zScoresPure))

	plt.plot([i for i in range(len(favCounts))], favCounts)
	plt.title("FavCount for all presidential tweets")
	plt.show();

	plt.plot([i for i in range(len(favCountsPure))], favCountsPure)
	plt.title("FavCount for all presidential tweets pure")
	plt.show();


def graphFavsByApprRating():
	ratings = getApprovalRatings()

	firstRating = ratings[0]

	tweets = getTweetsFromDB(orderBy="publishTime asc", conditions=["isRt = 0", "publishTime >= \"" + str(firstRating[0]) + "\""])

	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(tweets,daysPerMonth=1);
	# print(len(avgFavCounts),len(ratings));
	print("m0: ",months[0]," r0: ",ratings[0][0])
	minLen = min(len(months),len(ratings))

	fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)

	host = fig.add_subplot(111)

	par1 = host.twinx()

	# host.set_xlim(0, 2)
	# host.set_ylim(0, 2)
	# par1.set_ylim(0, 4)

	host.set_xlabel("months")
	host.set_ylabel("favourite count")
	par1.set_ylabel("approval rating")

	color1 = plt.cm.viridis(0)
	color2 = plt.cm.viridis(0.5)
	# color3 = plt.cm.viridis(1.0)
	xes = [i for i in range(minLen)]

	favsStandardized = avgFavCounts[:minLen];
	ratingsStandardized = [rat[1] for rat in ratings[:minLen]]
	pearson = pearsonCorrelationCoefficient(favsStandardized, ratingsStandardized);

	pFavs, = host.plot(xes, favsStandardized, color=color1, label="favCount")
	pApproval, = par1.plot(xes, ratingsStandardized, color=color2, label="approval rating")

	print("pearson : ",pearson)
	lns = [pFavs, pApproval]
	host.legend(handles=lns, loc='best')


	# right, left, top, bottom
	# no x-ticks
	# Sometimes handy, same for xaxis
	# par2.yaxis.set_ticks_position('right')
	plt.xticks(fifteenIndices, ["\'" + m[2:] for m in monthsCorrespondingToFifteenIndices], size='small')
	plt.title("pearson correlation " + str(pearson));



	# host.yaxis.label.set_color(pFavs.get_color())
	par1.yaxis.label.set_color(pApproval.get_color())
	plt.show()




def graphFavsAndRtsByRatio():


	daysPerMonth = 1;
	tweets = getTweetsFromDB(purePres=True,conditions = ["isRt = 0"],returnParams=["tweetText","publishTime", "favCount", "rtCount"],orderBy="publishTime asc")

	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(tweets, daysPerMonth)
	favsRtsRatio = [ avgFavCounts[i] / avgRtCounts[i] if avgRtCounts[i] > 0 else 0 for i in range(len(months))]



	pearson = pearsonCorrelationCoefficient(avgFavCounts,avgRtCounts)
	graphTwoDataSetsTogether(avgFavCounts, "fav counts", avgRtCounts ,"rt counts",title="rt counts"+ "favourite count and rt count; correlation coefficient: " + str(round(pearson,2)))
	exit()


	fig=plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06,right=0.94)

	host = fig.add_subplot(111)

	par1 = host.twinx()


	host.set_xlabel("months")
	host.set_ylabel("counts")
	par1.set_ylabel("favs/rts ratio")

	color1 = plt.cm.viridis(0)
	color2 = plt.cm.viridis(0.5)
	color3 = plt.cm.viridis(1.0)
	xes = [i for i in range(len(months))]

	pFavs, = host.plot(xes, avgFavCounts, color=color1, label="favCount")
	pRts, = host.plot(xes, avgRtCounts, color=color2, label="rtCount")
	# pRatio, = par1.plot(xes, favsRtsRatio, color=color3, label="ratiooo")


	# lns = [pFavs,pRts,pRatio]
	lns = [pFavs, pRts]
	host.legend(handles=lns, loc='best')

	# right, left, top, bottom
	# no x-ticks
	# Sometimes handy, same for xaxis
	# par2.yaxis.set_ticks_position('right')
	plt.xticks(fifteenIndices, ["\'"+m[2:] for m in monthsCorrespondingToFifteenIndices], size='small')


	# host.yaxis.label.set_color(pFavs.get_color())
	# par1.yaxis.label.set_color(pRatio.get_color())
	# plt.title();
	plt.show()