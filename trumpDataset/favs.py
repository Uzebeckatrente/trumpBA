

from trumpDataset.basisFuncs import *
from trumpDataset.part3funcs import getMostPopularWordsOverCountNMostPopularBigramsOverCount
from trumpDataset.stats import pearsonCorrelationCoefficient
from trumpDataset.visualization import boxAndWhiskerForKeyWordsFavs, computeXandYForPlotting


def getAverageFavCountPresTweets():
	query = "SELECT avg(favCount) FROM trump.tta2 "+relevancyDateThreshhold+" and isRt = 0"
	mycursor.execute(query)
	fvc = mycursor.fetchone()[0]
	return int(fvc)


def analyzeOverUnderMeanSkewOfKeywords(loadFromStorage = True):


	if loadFromStorage:
		wordsAndBigramsWithFavsMinusMeanFile = open("trumpDataset/wordsAndBigramsWithFavsMinusMean.p", "rb")
		wordsAndBigramsWithFavsMinusMean = pickle.load(wordsAndBigramsWithFavsMinusMeanFile);
		wordsAndBigramsWithFavsMinusMeanFile.close()

	else:
		startingTime = time.time()
		words, bigrams = getMostPopularWordsOverCountNMostPopularBigramsOverCount(25,10)
		favsList = []
		keywords = []


		wordsAndBigramsWithFavsMinusMean = []
		fvc = getAverageFavCountPresTweets()
		for index,word in enumerate(words):
			favs = getFavsByKeyword(word[0],rts = False);
			meanFavs = np.median(favs)
			###not the same length because the source words contain rts and also possiblty multiple counts per tweet
			favsList.append(favs);
			keywords.append(word[0]);
			wordsAndBigramsWithFavsMinusMean.append((word[0]+"\n"+str(word[1]), meanFavs - fvc,favs));
			if index%100 == 0:
				print(Fore.RED,index/len(words),Style.RESET_ALL)
		print("did thee words")
		# for index,bigram in enumerate(bigrams):
		# 	favs = getFavsByKeyword(bigram[0], rts=False);
		# 	meanFavs = np.median(favs)
		# 	favsList.append(favs);
		# 	keywords.append(bigram[0].replace(" ","\n"));
		# 	wordsAndBigramsWithFavsMinusMean.append((bigram[0].replace(" ","\n")+"\n"+str(bigram[1]), meanFavs - fvc,favs));
		# 	if index%100 == 0:
		# 		print(Fore.RED,index/len(bigrams),Style.RESET_ALL)
		wordsAndBigramsWithFavsMinusMean.sort(key=lambda tup: np.fabs(tup[1]), reverse=True)

		with open('trumpDataset/wordsAndBigramsWithFavsMinusMean.p', 'wb') as wordsAndBigramsWithFavsMinusMeanFile:
			pickle.dump(wordsAndBigramsWithFavsMinusMean, wordsAndBigramsWithFavsMinusMeanFile)
			wordsAndBigramsWithFavsMinusMeanFile.close()
		print(Fore.MAGENTA,time.time()-startingTime,Style.RESET_ALL);

	for w in wordsAndBigramsWithFavsMinusMean[:20]:
		print(w[0],w[1])
	boxAndWhiskerForKeyWordsFavs([tup[0] for tup in wordsAndBigramsWithFavsMinusMean[:20]],[tup[2] for tup in wordsAndBigramsWithFavsMinusMean[:20]],fvc)



def favouriteVsLengthTrends():
	tweets = getTweetsFromDB(orderBy="order by publishTime asc", conditions=["isRt = 0", "president", "deleted = 0", "length(tweetText) > 1"], returnParams=["tweetText","favCount"])
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



def favouriteOverTimeTrends():
	tweets = getTweetsFromDB(orderBy="publishTime asc",conditions="purePres")
	bestPearson = -1
	bestDaysPerMonthCount = -1;
	for daysPerMonthCount in [int(np.exp2(i)) for i in range(10)]:
		avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(
			tweets, daysPerMonth=daysPerMonthCount);

		fitLineCoefficients = np.polyfit([i for i in range(len(months))], avgFavCounts, 1, full=False)
		slope = fitLineCoefficients[0];
		intercept = fitLineCoefficients[1]
		pearson = pearsonCorrelationCoefficient([slope * i + intercept for i in range(len(months))], avgFavCounts)
		if pearson > bestPearson:
			bestPearson = pearson
			bestDaysPerMonthCount = daysPerMonthCount
		plt.plot([i for i in range(len(months))], avgFavCounts, label="favCounts")
		plt.plot([i for i in range(len(months))], [slope * i + intercept for i in range(len(months))], label="ols")
		plt.title(str(bestDaysPerMonthCount) + " days per month; pearson: " + str(bestPearson))
		plt.legend()
		plt.show()
	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(tweets, daysPerMonth=bestDaysPerMonthCount);



	fitLineCoefficients = np.polyfit([i for i in range(len(months))], avgFavCounts, 1, full=False)
	slope = fitLineCoefficients[0];
	intercept = fitLineCoefficients[1]
	plt.plot([i for i in range(len(months))],avgFavCounts,label="favCounts")
	plt.plot([i for i in range(len(months))],[slope*i+intercept for i in range(len(months))],label="ols")
	plt.title(str(bestDaysPerMonthCount)+" days per month; pearson: "+str(bestPearson))
	plt.legend()
	plt.show()

def graphPearsonsForApprFavOffset():
	ratings = getApprovalRatings()
	ratings = ratings[100:]
	pearsons = []
	dateRange = range(-100,100)
	for i in dateRange:
		tweets = getTweetsFromDB(orderBy="order by publishTime asc",
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
	plt.title(maxOffSetDay)
	plt.show()


def graphFavsByApprRating():
	ratings = getApprovalRatings()

	firstRating = ratings[0]

	tweets = getTweetsFromDB(orderBy="order by publishTime asc", conditions=["isRt = 0", "publishTime >= \"" + str(firstRating[0]) + "\""])

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


	daysPerMonth = 10;
	mycursor.execute("select tweetText, publishTime, favCount, rtCount from "+mainTable+" where isRt = 0 order by publishTime asc");
	tweets = mycursor.fetchall()
	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, _ = computeXandYForPlotting(tweets, daysPerMonth)
	favsRtsRatio = [ avgFavCounts[i] / avgRtCounts[i] if avgRtCounts[i] > 0 else 0 for i in range(len(months))]



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
	pRatio, = par1.plot(xes, favsRtsRatio, color=color3, label="ratiooo")


	lns = [pFavs,pRts,pRatio]
	host.legend(handles=lns, loc='best')

	# right, left, top, bottom
	# no x-ticks
	# Sometimes handy, same for xaxis
	# par2.yaxis.set_ticks_position('right')
	plt.xticks(fifteenIndices, ["\'"+m[2:] for m in monthsCorrespondingToFifteenIndices], size='small')


	# host.yaxis.label.set_color(pFavs.get_color())
	par1.yaxis.label.set_color(pRatio.get_color())
	plt.show()