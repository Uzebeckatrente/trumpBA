from matplotlib import patches

from .basisFuncs import *
from .part3funcs import computeMostCommonnGrams,getnGramsWithOverMOccurences;
from .favs import loadnGramsWithFavsMinusMeanMedian,calculateAndStorenGramsWithFavsMinusMeanMedian;
from .stats import pearsonCorrelationCoefficient;
import networkx as nx



def compareTwoEpochsByNTramTopo(epoch1, epoch2,allNGrams):
	runningSimilarity = 0;
	for nGram in allNGrams:
		if nGram in epoch1:
			count1, skew1, topo1 = epoch1[nGram]
		else:
			count1, skew1, topo1 = 0,0,0;

		if nGram in epoch2:
			count2, skew2, topo2 = epoch2[nGram]
		else:
			count2, skew2, topo2 = 0,0,0;
		runningSimilarity += np.fabs(topo1-topo2);
	return runningSimilarity;


# def topNWordsOfEpochChangeOverTime(epochCount = 10):
# 	tweets = getTweetsFromDB(purePres=True, returnParams=["favCount", "cleanedText, publishTime"], orderBy="publishTime asc")

def epochSimilarityVisualization(dataFrameDict, epochCount):
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)
	iterColor = 5592405;
	for epoch in dataFrameDict:
		iterColor += 1001;
		iterColorString = hex(iterColor)[2:]
		c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
		args = np.argsort(dataFrameDict[epoch])
		plt.plot([e for e in range(epochCount)], dataFrameDict[epoch],color=c);
		plt.scatter([e for e in range(epochCount)], dataFrameDict[epoch], label=str(epoch),color=c);
		plt.scatter(args[1], dataFrameDict[epoch][args[1]], c="red")
		plt.scatter(args[0], dataFrameDict[epoch][args[0]], c="orange")
		for i in range(len(dataFrameDict[epoch])):
			if i == args[0]:
				dataFrameDict[epoch][i] = Fore.RED + str(dataFrameDict[epoch][i]) + Style.RESET_ALL
			elif i == args[1]:
				dataFrameDict[epoch][i] = Fore.BLUE + str(dataFrameDict[epoch][i]) + Style.RESET_ALL
			else:
				dataFrameDict[epoch][i] = Fore.BLACK + str(dataFrameDict[epoch][i]) + Style.RESET_ALL
	plt.xticks([e for e in range(epochCount)]);
	plt.yticks([e for e in range(epochCount)]);
	plt.legend()
	df = pd.DataFrame(data=dataFrameDict, index=[epoch for epoch in range(epochCount)]);

	print(df);
	plt.show()

def computenGramsWithBiggestSkewDifference(nGramsForAllEpochsRegularizedSkews):
	skewsForAllNgrams = {}

	nGramsInAllSets = None
	for epoch in nGramsForAllEpochsRegularizedSkews:
		print(epoch[0])
		nGramsInThisEpoch = set([epoch[i][0] for i in range(len(epoch))])
		if nGramsInAllSets == None:
			nGramsInAllSets = nGramsInThisEpoch;
		else:
			nGramsInAllSets = set.intersection(nGramsInAllSets,nGramsInThisEpoch)



	for epoch in nGramsForAllEpochsRegularizedSkews:
		for nGramTuple in epoch:
			nGram = nGramTuple[0];
			if nGram not in nGramsInAllSets: continue;
			skew = nGramTuple[1]
			if nGram in skewsForAllNgrams:
				skewsForAllNgrams[nGram].append(skew)
			else:
				skewsForAllNgrams[nGram] = [skew]
	allNGrams = []
	skewsDifferences = []
	for nGram in skewsForAllNgrams.keys():
		allNGrams.append(nGram);
		skews = skewsForAllNgrams[nGram];
		difference = max(skews)-min(skews);
		skewsDifferences.append(difference);
	idx = np.argsort(skewsDifferences)[::-1];
	allNGramsInDifferenceOrder = [allNGrams[i] for i in idx]
	skewsDifferences = [skewsDifferences[i] for i in idx]
	print("allNGramsInDifferenceOrder: ",allNGramsInDifferenceOrder, " diffs: ",skewsDifferences)
	return allNGramsInDifferenceOrder


def relativeAbundanceNew(epochCount = 4, globalRegularization = False, minCount = 10, maxNumberOfNGrams = np.infty):

	tweets = getTweetsFromDB(purePres=True, returnParams=["favCount","cleanedText, publishTime"],orderBy="publishTime asc")
	firstPublishTime = tweets[0][2]
	lastPublishTime = tweets[-1][2]
	totalSeconds = (lastPublishTime - firstPublishTime).total_seconds()
	epochTime = totalSeconds / epochCount;
	print(epochTime);

	print("total epoch time in days: ", epochTime / (3600 * 24))
	ns = [1, 2, 3, 4]

	# assign tweets to epochs
	epochs = []
	startingIndex = 0;
	endingTime = firstPublishTime + datetime.timedelta(seconds=epochTime);
	for index, tweet in enumerate(tweets):
		if tweet[2] > endingTime:
			epochs.append(tweets[startingIndex:index]);
			startingIndex = index;
			endingTime = endingTime + datetime.timedelta(seconds=epochTime)
			if len(epochs) == epochCount - 1:
				break;
	epochs.append(tweets[startingIndex:])

	nGramsAndCountsForEachEpoch = []
	nGramsAndCountsForEachEpochRegularized = []
	allNGrams = set()


	for index, epochTweets in enumerate(epochs):
		epochCleanTextAndFavs = [tweet[0:2] for tweet in epochTweets]
		cleanTextsThisEpoch = [t[1] for t in epochCleanTextAndFavs];
		tweetsHash = hashTweets(epochCleanTextAndFavs)
		nGramsAndCountsForEpoch = []
		for n in ns:
			computeMostCommonnGrams(epochTweets, n);
			myNGrams = getnGramsWithOverMOccurences(n, minCount, tweetsHash)

			nGramsAndCountsForEpoch.extend(myNGrams);

		maxOccurencesAcrossEpoch = max([tup[1] for tup in nGramsAndCountsForEpoch])
		nGramsAndCountsForEpochRegularized = [(tup[0],tup[1]/maxOccurencesAcrossEpoch) for tup in nGramsAndCountsForEpoch];
		nGramsAndCountsForEachEpoch.append(nGramsAndCountsForEpoch);
		nGramsAndCountsForEachEpochRegularized.append(nGramsAndCountsForEpochRegularized);
		allNGrams.update([tup[0] for tup in nGramsAndCountsForEpoch])
	allNGrams = list(allNGrams)
	allNGrams.sort()
	for epoch in range(len(nGramsAndCountsForEachEpoch)):
		nGramsForEpoch = [tup[0] for tup in nGramsAndCountsForEachEpoch[epoch]];
		for nGram in allNGrams:
			if nGram not in nGramsForEpoch:
				nGramsAndCountsForEachEpoch[epoch].append((nGram,0));
				nGramsAndCountsForEachEpochRegularized[epoch].append((nGram, 0));
		nGramsAndCountsForEachEpoch[epoch].sort(key = lambda tup: tup[0])
		nGramsAndCountsForEachEpochRegularized[epoch].sort(key=lambda tup: tup[0])

	volatilityPerNgram = []
	for nGramIndex in range(len(allNGrams)):
		volatility = 0;
		for epoch in range(epochCount-1):
			volatility += np.fabs(nGramsAndCountsForEachEpochRegularized[epoch][nGramIndex][1]-nGramsAndCountsForEachEpochRegularized[epoch+1][nGramIndex][1])
		if allNGrams[nGramIndex] == "great" or allNGrams[nGramIndex] == "coronavirus":
			volatility = np.infty
		volatilityPerNgram.append(volatility);
	volatilityIDX = list(np.argsort(volatilityPerNgram))[::-1]
	for epoch in range(epochCount):
		nGramsAndCountsForEachEpochRegularized[epoch] = [nGramsAndCountsForEachEpochRegularized[epoch][i] for i in volatilityIDX][:maxNumberOfNGrams]
		nGramsAndCountsForEachEpoch[epoch] = [nGramsAndCountsForEachEpoch[epoch][i] for i in volatilityIDX][:maxNumberOfNGrams]
	allNGrams = [allNGrams[i] for i in volatilityIDX][:maxNumberOfNGrams]




	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.08, right=0.94, bottom=0.21, top=0.92)
	fig.tight_layout(pad=3.0)
	axCount = fig.add_subplot(111);
	iterColor = 11;


	ind = np.arange(len(nGramsAndCountsForEachEpochRegularized[0]))
	width = 1 / (epochCount + 1)
	rects = []
	np.random.seed(423)
	for epoch, epochCounts in enumerate(nGramsAndCountsForEachEpochRegularized):
		iterColorString = hex(iterColor)[2:]
		c = randomHexColor()
		rects1 = axCount.bar(ind + width * epoch, [tup[1] for tup in epochCounts], width, color=c)
		rects.append(rects1)

		iterColor += int(16777215 /epochCount)

	for nGramIndex in range(len(allNGrams)):
		acceptableRect = patches.Rectangle((nGramIndex-width*0.5, 0), width*epochCount, 1, linewidth=0.5, alpha=0.5, color=("#aabbcc" if nGramIndex % 2 == 0 else "#bbccaa"))
		plt.gca().add_patch(acceptableRect)



	axCount.set_title("Relative Frequency for each Year of Presidency for a Selection of n-grams",fontsize=25)
	axCount.set_ylabel('Relative Frequency of n-gram Usage',fontsize=20)
	# axCount.set_xlabel('Relative Frequency of n-gram Usage')
	axCount.set_xticks(ind + width)
	axCount.set_xticklabels(["\n"+ nGram.replace(" ","\n") for nGramIndex, nGram in enumerate(allNGrams[:maxNumberOfNGrams])]);
	axCount.tick_params(axis='both', which='major', labelsize=20)
	axCount.xaxis.grid(False)
	for tick in axCount.xaxis.get_major_ticks():
		# tick.label.set_fontsize(14)
		# specify integer or one of preset strings, e.g.
		# tick.label.set_fontsize('x-small')
		tick.label.set_rotation('vertical')
	axCount.legend((rects[i][0] for i in range(len(rects))), ("year " + str(i) for i in range(len(rects))),fontsize=20)
	plt.show()




def skewNew(epochCount = 4, globalRegularization = False, minCount = 10, maxNumberOfNGrams = np.infty):

	tweets = getTweetsFromDB(purePres=True, returnParams=["favCount","cleanedText, publishTime"],orderBy="publishTime asc")
	firstPublishTime = tweets[0][2]
	lastPublishTime = tweets[-1][2]
	totalSeconds = (lastPublishTime - firstPublishTime).total_seconds()
	epochTime = totalSeconds / epochCount;
	print(epochTime);

	print("total epoch time in days: ", epochTime / (3600 * 24))
	ns = [1, 2, 3, 4]

	# assign tweets to epochs
	epochs = []
	startingIndex = 0;
	endingTime = firstPublishTime + datetime.timedelta(seconds=epochTime);
	for index, tweet in enumerate(tweets):
		if tweet[2] > endingTime:
			epochs.append(tweets[startingIndex:index]);
			startingIndex = index;
			endingTime = endingTime + datetime.timedelta(seconds=epochTime)
			if len(epochs) == epochCount - 1:
				break;
	epochs.append(tweets[startingIndex:])

	nGramsAndSkewsForEachEpoch = []
	nGramsAndSkewsForEachEpochRegularized = []
	allNGrams = set()


	for index, epochTweets in enumerate(epochs):
		epochCleanTextAndFavs = [tweet[0:2] for tweet in epochTweets]
		epochMeanFavCount = np.mean([t[0] for t in epochTweets])
		cleanTextsThisEpoch = [t[1] for t in epochCleanTextAndFavs];
		tweetsHash = hashTweets(epochCleanTextAndFavs)
		nGramsAndSkewsForEpoch = []

		for n in ns:
			computeMostCommonnGrams(epochTweets, n);
			myNGrams = getnGramsWithOverMOccurences(n, minCount, tweetsHash)

			for nGram in myNGrams:
				tweetsWithThisNGramInThem = [t for t in epochTweets if nGram[0] in t[1]];
				favsForTweetsWithNGramInThem = [t[0] for t in tweetsWithThisNGramInThem];
				meanFavForTweetsWithNGramInThem = np.mean(favsForTweetsWithNGramInThem)
				skew = meanFavForTweetsWithNGramInThem-epochMeanFavCount
				nGramsAndSkewsForEpoch.append((nGram[0],skew));
		skewsForEpoch = [tup[1] for tup in nGramsAndSkewsForEpoch];
		minSkew = np.min(skewsForEpoch);
		maxSkew = np.max(skewsForEpoch);
		skewsForEpoch = [skew-minSkew for skew in skewsForEpoch];
		skewsForEpoch = [skew/(maxSkew-minSkew) for skew in skewsForEpoch];
		skewsForEpoch = [2*skew-1 for skew in skewsForEpoch];

		nGramsAndSkewsForEpochRegularized = [(tup[0],(2*(tup[1]-minSkew)/(maxSkew-minSkew))-1) for tup in nGramsAndSkewsForEpoch];
		nGramsAndSkewsForEpochRegularized.sort(key = lambda tup: tup[1]);
		nGramsAndSkewsForEachEpoch.append(nGramsAndSkewsForEpoch);
		nGramsAndSkewsForEachEpochRegularized.append(nGramsAndSkewsForEpochRegularized);
		allNGrams.update([tup[0] for tup in nGramsAndSkewsForEpoch])
		print("shwa",list(tup[0] for tup in nGramsAndSkewsForEpoch))
		print("zigadumbe: ",nGramsAndSkewsForEpoch);
	allNGrams = list(allNGrams)
	allNGrams.sort()
	for epoch in range(len(nGramsAndSkewsForEachEpoch)):
		nGramsForEpoch = [tup[0] for tup in nGramsAndSkewsForEachEpoch[epoch]];
		for nGram in allNGrams:
			if nGram not in nGramsForEpoch:
				nGramsAndSkewsForEachEpoch[epoch].append((nGram,0));
				nGramsAndSkewsForEachEpochRegularized[epoch].append((nGram, 0));
		nGramsAndSkewsForEachEpoch[epoch].sort(key = lambda tup: tup[0])
		nGramsAndSkewsForEachEpochRegularized[epoch].sort(key=lambda tup: tup[0])

	volatilityPerNgram = []
	for nGramIndex in range(len(allNGrams)):
		volatility = 0;
		for epoch in range(epochCount-1):
			if nGramsAndSkewsForEachEpochRegularized[epoch][nGramIndex][1] != 0:
				volatility += np.fabs(nGramsAndSkewsForEachEpochRegularized[epoch][nGramIndex][1]-nGramsAndSkewsForEachEpochRegularized[epoch+1][nGramIndex][1])
		print(allNGrams[nGramIndex]," has volatility: ",volatility)
		if allNGrams[nGramIndex] == "invisible" and False:
			volatility = np.infty
		if "prime" in allNGrams[nGramIndex]:
			volatility = 0;
		volatilityPerNgram.append(volatility);
	volatilityIDX = list(np.argsort(volatilityPerNgram))[::-1]
	for epoch in range(epochCount):
		nGramsAndSkewsForEachEpoch[epoch] = [nGramsAndSkewsForEachEpoch[epoch][i] for i in volatilityIDX][:maxNumberOfNGrams]
		nGramsAndSkewsForEachEpochRegularized[epoch] = [nGramsAndSkewsForEachEpochRegularized[epoch][i] for i in volatilityIDX][:maxNumberOfNGrams]
	allNGrams = [allNGrams[i] for i in volatilityIDX][:maxNumberOfNGrams]
	print("allNGrams: ",allNGrams)


	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.08, right=0.94, bottom=0.21, top=0.92)
	fig.tight_layout(pad=3.0)
	axCount = fig.add_subplot(111);
	iterColor = 11;


	ind = np.arange(len(nGramsAndSkewsForEachEpochRegularized[0]))
	width = 1 / (epochCount + 1)
	rects = []
	np.random.seed(423)
	for epoch, epochSkews in enumerate(nGramsAndSkewsForEachEpochRegularized):
		print("epochSkews: ",epochSkews)
		iterColorString = hex(iterColor)[2:]
		c = randomHexColor()
		print("thingoes: ",[tup[1] for tup in epochSkews]);
		rects1 = axCount.bar(ind + width * epoch, [tup[1] for tup in epochSkews], width, color=c)
		rects.append(rects1)

		iterColor += int(16777215 /epochCount)

	for nGramIndex in range(len(allNGrams)):
		acceptableRect = patches.Rectangle((nGramIndex-width*0.5, 0), width*epochCount, 1, linewidth=0.5, alpha=0.5, color=("#aabbcc" if nGramIndex % 2 == 0 else "#bbccaa"))
		plt.gca().add_patch(acceptableRect)

	axCount.set_title("Skew of Top N-Grams for each Year of Presidency",fontsize=25)
	axCount.set_ylabel('Skew for each n-gram (Regularized)',fontsize=20)
	# axCount.set_xlabel('Relative Frequency of n-gram Usage')
	axCount.set_xticks(ind + width)
	axCount.set_xticklabels(["\n"+ nGram.replace(" ","\n") for nGramIndex, nGram in enumerate(allNGrams[:maxNumberOfNGrams])]);
	axCount.tick_params(axis='both', which='major', labelsize=20)
	axCount.xaxis.grid(False)
	for tick in axCount.xaxis.get_major_ticks():
		# tick.label.set_fontsize(14)
		# specify integer or one of preset strings, e.g.
		# tick.label.set_fontsize('x-small')
		tick.label.set_rotation('vertical')
	axCount.legend((rects[i][0] for i in range(len(rects))), ("year " + str(i) for i in range(len(rects))),fontsize=20)
	plt.show()


	###old
	iterColor = 11;
	for nGramIndex in range(maxNumberOfNGrams):
		nGram = allNGrams[nGramIndex];
		skews = [];
		for epochIndex in range(len(nGramsAndSkewsForEachEpochRegularized)):
			skew = nGramsAndSkewsForEachEpochRegularized[epochIndex][nGramIndex];
			skews.append(skew[1]);
		# c = randomHexColor()
		iterColorString = hex(iterColor)[2:]
		c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
		print(c)
		print("skews: ",skews)



		plt.plot(list(range(len(skews))), skews, color=c, label=nGram);
		plt.scatter(list(range(len(skews))), skews, color=c);

		iterColor += int(16777215 / maxNumberOfNGrams)


	plt.gca().set_title("Skew for a Selection of Highly Skews n-grams of Each Year of Trump's Presidency")
	plt.gca().set_xticks(list(range(0, epochCount)))
	plt.gca().set_xticklabels(["Year " + str(i + 1) for i in range(0, epochCount)])
	plt.gca().set_yticks([-0.8, 0, 0.8]);
	plt.gca().set_yticklabels(["Least\nPopular", "Neutral", "Most\nPopular"])
	plt.gca().legend(loc="lower middle")  # custom_lines, ['Cold', 'Medium', 'Hot'],
	# axCount.set_ylabel("Relative Abundance of Most Frequent n-grams")

	plt.show()

def compareVocabOverTime(epochCount=5,globalRegularization = False,minCount = 10, maxNumberOfNGrams = np.infty, allGraphsInOnePlot = False):
	# epochCount = 50;
	tweets = getTweetsFromDB(purePres=True, returnParams=["favCount","cleanedText, publishTime"],orderBy="publishTime asc")
	quickieGrams = ["coronavirus", "great", "fake news"]
	quickie = False






	firstPublishTime = tweets[0][2]
	lastPublishTime = tweets[-1][2]
	totalSeconds = (lastPublishTime-firstPublishTime).total_seconds()
	epochTime = totalSeconds/epochCount;
	print(epochTime);

	print("total epoch time in days: ",epochTime/(3600*24))
	ns = [1, 2, 3, 4]



	#assign tweets to epochs
	epochs = []
	startingIndex = 0;
	endingTime = firstPublishTime + datetime.timedelta(seconds = epochTime);
	for index,tweet in enumerate(tweets):
		if tweet[2] > endingTime:
			epochs.append(tweets[startingIndex:index]);
			startingIndex = index;
			endingTime = endingTime + datetime.timedelta(seconds = epochTime)
			if len(epochs) == epochCount-1:
				break;

	epochs.append(tweets[startingIndex:])
	nGramsForAllEpochs = []


	for index, epochTweets in enumerate(epochs):

		epochCleanTextAndFavs = [tweet[0:2] for tweet in epochTweets]
		tweetsHash = hashTweets(epochCleanTextAndFavs)


		nGramsForEpoch = []
		for n in ns:
			computeMostCommonnGrams(epochTweets, n);
			myNGrams = getnGramsWithOverMOccurences(n, minCount, tweetsHash)
			nGramsForEpoch.extend(myNGrams)
		nGramsForEpoch.sort(key=lambda tuple: tuple[1], reverse=True)

		calculateAndStorenGramsWithFavsMinusMeanMedian(ns, epochCleanTextAndFavs, False)
		_,nGramsWithFavsMinusMedian = loadnGramsWithFavsMinusMeanMedian(ns, tweetsHash)
		# if index == len(epochs)-1:
		# 	plt.scatter([i for i in range(len(nGramsWithFavsMinusMedian))],[tup[nGramStorageIndicesSkew["skew"]] for tup in nGramsWithFavsMinusMedian])
		# 	plt.plot([0, len(nGramsWithFavsMinusMedian)],[0,0]);
		# 	plt.show()
		nGramsWithFavsMinusMeanMedianDict = {}
		for nGram in nGramsWithFavsMinusMedian:
			nGramsWithFavsMinusMeanMedianDict[nGram[0]] = nGram
		for i in range(len(nGramsForEpoch)):
			e1 = nGramsForEpoch[i][0]
			e2 = nGramsForEpoch[i][1]
			medianDictEntry = nGramsWithFavsMinusMeanMedianDict[nGramsForEpoch[i][0]]
			e3 = medianDictEntry[nGramStorageIndicesSkew["skew"]]
			if e3 > 0:
				print("n00b")
			nGramsForEpoch[i] = (e1,e2,e3)
		nGramsForAllEpochs.append(nGramsForEpoch);



	print(Fore.RED,"\n\n\nnow sorted by skew\n\n\n",Style.RESET_ALL)

	for epoch in range(len(nGramsForAllEpochs)):
		nGramsForAllEpochs[epoch].sort(key = lambda tup: tup[2], reverse = True)

	minimumCountThreshholds = [5]#[5, 10, 50, 100, 200, 400, 800];
	for minimumCountThreshhold in minimumCountThreshholds:
		print(Fore.MAGENTA, "minimumCountThreshhold: ", minimumCountThreshhold, Style.RESET_ALL, "\n\n")
		for epoch in range(epochCount - 1):
			if epoch == epochCount-2:
				print("n00b:");
			thisEpochsGrams = [tup[0] for tup in nGramsForAllEpochs[epoch]][:minimumCountThreshhold]
			thisEpochsGramsFull = nGramsForAllEpochs[epoch][:minimumCountThreshhold]
			nextEpochsGrams = [tup[0] for tup in nGramsForAllEpochs[epoch + 1]][:minimumCountThreshhold]
			nextEpochsGramsFull = nGramsForAllEpochs[epoch + 1][:minimumCountThreshhold]
			intersectionSet = np.intersect1d(thisEpochsGrams, nextEpochsGrams);


			print("epoch: ", epoch, " has ", len(nGramsForAllEpochs[epoch]), " ngrams of count at least",minCount,"; its most popular minimumCountThreshhold bigrams were shared to ", len(intersectionSet) / len(thisEpochsGrams), " extent with epoch ", epoch + 1);
			for i in range(minimumCountThreshhold):
				try:
					print(thisEpochsGramsFull[i], nextEpochsGramsFull[i])
				except:
					break;
			print()

		intersectionSet = np.intersect1d([tup[0] for tup in nGramsForAllEpochs[0]][:minimumCountThreshhold], [tup[0] for tup in nGramsForAllEpochs[-1]][:minimumCountThreshhold]);
		print("epoch: ", 0, " has ", len(nGramsForAllEpochs[0]), " ngrams of count at least ",minCount,"; its most popular minimumCountThreshhold bigrams were shared to ", len(intersectionSet) / len([tup[0] for tup in nGramsForAllEpochs[0]][:minimumCountThreshhold]), " extent with epoch ", len(epochs) - 1);
	# print(nGramsForAllEpochs);

	'''
	format of nGramsForAllEpochs: [[(nGram, count, skew),...],...]
	'''

	maxCounts = []
	maxAbsSkews = []

	for epochGrams in nGramsForAllEpochs:
		maxCount = max([tup[1] for tup in epochGrams]);
		maxAbsSkew = max([np.fabs(tup[2]) for tup in epochGrams]);
		maxCounts.append(maxCount);
		maxAbsSkews.append(maxAbsSkew)
	maxCount = max(maxCounts);
	maxAbsSkew = max(maxAbsSkews);
	print(maxCount);
	print(maxAbsSkew);




	#TODO: regularize count and skew so that each is between [0,1] and [-1,1] respectively
	nGramsForAllEpochsRegularized = []
	nGramsForAllEpochsRegularizedTopos = []
	nGramsForAllEpochsRegularizedCounts = []
	nGramsForAllEpochsRegularizedSkews = []
	allNgrams = set()
	nGramsInAllEpochs = set([tup[0] for tup in nGramsForAllEpochs[0]]);
	for i in range(len(nGramsForAllEpochs)):
		if i == len(nGramsForAllEpochs)-1:
			print("n00b")
		allNgrams.update([tup[0] for tup in nGramsForAllEpochs[i]])
		nGramsInAllEpochs.intersection_update([tup[0] for tup in nGramsForAllEpochs[i]])

		if globalRegularization:
			nGramsReg = [(tup[0], tup[1] / maxCount, tup[2] / maxAbsSkew, (tup[1] / maxCount) * (tup[2] / maxAbsSkew)) for tup in nGramsForAllEpochs[i]];
		else:
			nGramsReg = [(tup[0], tup[1] / maxCounts[i], tup[2] / maxAbsSkews[i], (tup[1] / maxCounts[i]) * (tup[2] / maxAbsSkews[i])) for tup in nGramsForAllEpochs[i]];

		nGramsForAllEpochsRegularized.append( {tup[0]: tup[1:] for tup in nGramsReg})


		nGramsRegTopos = [(tup[0],np.fabs(tup[3])) for tup in nGramsReg]
		nGramsRegTopos.sort(key = lambda tup: tup[1], reverse=True);
		nGramsForAllEpochsRegularizedTopos.append(nGramsRegTopos)

		nGramsRegCounts = [(tup[0], tup[1]) for tup in nGramsReg]
		nGramsRegCounts.sort(key=lambda tup: tup[1], reverse=True);
		nGramsForAllEpochsRegularizedCounts.append(nGramsRegCounts)

		nGramsRegSkews = [(tup[0], np.fabs(tup[2])) for tup in nGramsReg]
		nGramsRegSkews.sort(key=lambda tup: tup[1], reverse=True);
		nGramsForAllEpochsRegularizedSkews.append(nGramsRegSkews)


	nGramsWithBiggestSkewDifference = computenGramsWithBiggestSkewDifference(nGramsForAllEpochsRegularizedSkews)


	topnFromEachEpoch = 10;
	nGramTopoDict = {nGram:[] for nGram in nGramsInAllEpochs}
	nGramTopoDictAll = {nGram : [] for nGram in allNgrams}
	nGramSkewDictAll = {nGram: [] for nGram in allNgrams}
	nGramCountDictAll = {nGram: [] for nGram in allNgrams}

	for nGram in allNgrams:
		for epoch in range(len(nGramsForAllEpochs)):
			if epoch == len(nGramsForAllEpochs)-1:
				print("n00b")
			topn_nGramsByTopo = [tup[0] for tup in nGramsForAllEpochsRegularizedTopos[epoch]][:topnFromEachEpoch]#
			topn_nGramsByCount = [tup[0] for tup in nGramsForAllEpochsRegularizedCounts[epoch]][:topnFromEachEpoch]#

			topn_nGramsBySkew = [tup[0] for tup in nGramsForAllEpochsRegularizedSkews[epoch]][:topnFromEachEpoch]#
			# topn_nGramsBySkew = nGramsWithBiggestSkewDifference[:topnFromEachEpoch]


			if nGram in nGramsInAllEpochs:
				nGramTopoDict[nGram].append(nGramsForAllEpochsRegularized[epoch][nGram][2])
			if nGram in topn_nGramsByTopo:
				nGramTopoDictAll[nGram].append(nGramsForAllEpochsRegularized[epoch][nGram][2])
			else:
				nGramTopoDictAll[nGram].append(0);

			if nGram in topn_nGramsBySkew:
				nGramSkewDictAll[nGram].append(nGramsForAllEpochsRegularized[epoch][nGram][1])
			else:
				nGramSkewDictAll[nGram].append(0);

			if nGram in topn_nGramsByCount or nGram in nGramsForAllEpochsRegularized[epoch]:
				nGramCountDictAll[nGram].append(nGramsForAllEpochsRegularized[epoch][nGram][0])
			else:
				nGramCountDictAll[nGram].append(0);

	dataFrameDict = {}
	for epoch1 in range(epochCount):
		dataFrameDict[epoch1] = []
		for epoch2 in range(epochCount):
			epoch1Grams = nGramsForAllEpochsRegularized[epoch1]
			epoch2Grams = nGramsForAllEpochsRegularized[epoch2]
			similarity = round(compareTwoEpochsByNTramTopo(epoch1Grams, epoch2Grams, allNgrams), 2);
			dataFrameDict[epoch1].append(similarity)

	if False:
		epochSimilarityVisualization(dataFrameDict,epochCount)



	if True:
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94, bottom=0.02, top = 0.97)
		fig.tight_layout(pad=3.0)


		if allGraphsInOnePlot:
			axInEveryEpoch = fig.add_subplot(411)
			axTopo = fig.add_subplot(412)
			axCount = fig.add_subplot(413)
			axSkew = fig.add_subplot(414)
			axes = [axInEveryEpoch, axTopo, axSkew, axCount]
		else:
			axCount = fig.add_subplot(111);
			# axInEveryEpoch = fig.add_subplot(111);

		# # iterColor = 11;
		# # for nGram,topos in nGramTopoDict.items():
		# # 	iterColorString = hex(iterColor)[2:]
		# # 	c = "#"+(6-len(iterColorString))*'0'+iterColorString;
		# # 	print(c)
		# #
		# # 	axInEveryEpoch.plot(list(range(len(topos))),topos,color=c,label = nGram);
		# # 	axInEveryEpoch.scatter(list(range(len(topos))), topos, color=c);
		# # 	iterColor += int(16777215 / len(nGramTopoDict))
		# #
		# # if not allGraphsInOnePlot:
		# # 	axInEveryEpoch.set_title("topo for ngrams appearing in every epoch")
		# # 	axInEveryEpoch.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axInEveryEpoch, nGramsForAllEpochsRegularized))  #
		# # 	plt.show()
		# # 	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		# # 	fig.subplots_adjust(left=0.06, right=0.94, bottom=0.02, top=0.97)
		# # 	fig.tight_layout(pad=3.0)
		# # 	axTopo = fig.add_subplot(111);
		# #
		# #
		# #
		# # iterColor = 11;
		# # for nGram, topos in nGramTopoDictAll.items():
		# # 	if not quickie or nGram in quickieGrams:
		# #
		# # 		# c = randomHexColor()
		# # 		iterColorString = hex(iterColor)[2:]
		# # 		c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
		# # 		print(c)
		# #
		# #
		# # 		axTopo.plot(list(range(len(topos))), topos, color=c, label=nGram);
		# # 		axTopo.scatter(list(range(len(topos))), topos, color=c);
		# # 		iterColor += int(16777215 / len(nGramTopoDictAll))
		# # if not allGraphsInOnePlot:
		# # 	axTopo.set_title("topo for top " + str(topnFromEachEpoch) + " ngrams of each epoch")
		# # 	axTopo.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axTopo, nGramsForAllEpochsRegularized))  #
		# # 	plt.show()
		# # 	# fig = plt.figure()
		# # 	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		# # 	fig.subplots_adjust(left=0.08, right=0.94, bottom=0.11, top=0.92)
		# # 	fig.tight_layout(pad=3.0)
		# # 	axCount = fig.add_subplot(111);
		#
		# fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		# fig.subplots_adjust(left=0.08, right=0.94, bottom=0.11, top=0.92)
		# fig.tight_layout(pad=3.0)
		# axCount = fig.add_subplot(111);
		# iterColor = 11;
		# nGramCountDictItemsList = list(nGramCountDictAll.items())[:maxNumberOfNGrams];
		# nGramCountDictNGrams = [tup[0] for tup in nGramCountDictItemsList]
		# nGramCountDictCounts = [tup[1] for tup in nGramCountDictItemsList]
		# nGramCountDictCountsPerEpoch = []
		# for epoch in range(len(nGramCountDictCounts[0])):
		# 	nGramCountDictCountsThisEpoch = [nGramCounts[epoch] for nGramCounts in nGramCountDictCounts]
		# 	nGramCountDictCountsPerEpoch.append(nGramCountDictCountsThisEpoch)
		# ind = np.arange(len(nGramCountDictNGrams))
		# width = 1/(len(nGramCountDictCountsPerEpoch)+1)
		# rects = []
		# for epoch, epochCounts in enumerate(nGramCountDictCountsPerEpoch):
		# 	rects1 = axCount.bar(ind + width*epoch, epochCounts, width, color='r')
		# 	rects.append(rects1)
		# 	axCount.set_ylabel('Scores')
		# axCount.set_xticks(ind + width)
		# axCount.set_xticklabels(nGramCountDictNGrams)
		# axCount.legend((rects[i][0] for i in range(len(rects))), ("epoch " + str(i) for i in range(len(rects))))
		#
		# if not allGraphsInOnePlot:
		# 	plt.show()
		# 	fig = plt.figure()#num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		# 	#fig.subplots_adjust(left=0.06, right=0.94, bottom=0.02, top=0.97)
		# 	#fig.tight_layout(pad=3.0)
		# 	axSkew = fig.add_subplot(111);
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.08, right=0.94, bottom=0.11, top=0.92)
	fig.tight_layout(pad=3.0)
	axSkew = fig.add_subplot(111);
	iterColor = 11;
	for nGram, skews in list(nGramSkewDictAll.items()):
		if not quickie or nGram in quickieGrams:
			# c = randomHexColor()
			iterColorString = hex(iterColor)[2:]
			c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
			print(c)

			if not np.any(skews):
				axSkew.plot(list(range(len(skews))), skews, color=c);
				axSkew.scatter(list(range(len(skews))), skews, color=c);
			else:
				axSkew.plot(list(range(len(skews))), skews, color=c, label=nGram);
				axSkew.scatter(list(range(len(skews))), skews, color=c);


			iterColor += int(16777215 / len(nGramCountDictAll))

	if not allGraphsInOnePlot:

		axSkew.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axSkew, nGramsForAllEpochsRegularized))  #
		if epochCount != 4:
			axSkew.set_title("Skew for Highest Skewed " + str(topnFromEachEpoch) + " ngrams of each Month")
			axSkew.set_xticks(list(range(0,epochCount,3)))
			axSkew.set_xticklabels(["Month "+str(i+1) for i in range(0,epochCount,3)])
		else:
			axSkew.set_title("Skew for Highest Skewed " + str(topnFromEachEpoch) + " ngrams of each Year")
			axSkew.set_xticks(list(range(0, epochCount)))
			axSkew.set_xticklabels(["Year " + str(i + 1) for i in range(0, epochCount)])
		axSkew.set_yticks([-0.8, 0,0.8]);
		axSkew.set_yticklabels(["Least\nPopular", "Neutral","Most\nPopular"])
		axSkew.legend(loc="lower middle")#custom_lines, ['Cold', 'Medium', 'Hot'],
		# axCount.set_ylabel("Relative Abundance of Most Frequent n-grams")

		plt.show()



	if allGraphsInOnePlot:
		for ax in axes:
			ax.plot([0, epochCount-1], [0, 0],'--', linewidth=5)
		axInEveryEpoch.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axInEveryEpoch,nGramsForAllEpochsRegularized))#
		axCount.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axCount,nGramsForAllEpochsRegularized))#
		axTopo.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axTopo,nGramsForAllEpochsRegularized))#
		axSkew.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axSkew,nGramsForAllEpochsRegularized))#

		axSkew.set_title("skews top " + str(topnFromEachEpoch) + " ngrams of each epoch")
		axCount.set_title("counts top " + str(topnFromEachEpoch) + " ngrams of each epoch")
		axTopo.set_title("topo for top " + str(topnFromEachEpoch) + " ngrams of each epoch")
		axInEveryEpoch.set_title("topo for ngrams appearing in every epoch")


		plt.show()




	print(nGramsInAllEpochs)


	#TODO: devise a measurement, topo, that combines count and skew to determine the most popular tweets in both count and skew
	#TODO: compare the topo between epochs
	#TODO: visualize the similarities between epochs somehow lol


	#goal: model changing vocabulary over time in order to have a more accurate idea of which terms are in trend.
	#bemerkung: there is great shifting of the tides, words shift in and out of grace
	#conclusion: for training, we should (use only bzw. highly prioritise) nearest training data/last relevant water mark
	#for example: he hasn't tweeted about crooked hillary with as much virility as in 2016 but everytime he does it has a strong skew
	#however: great used to have a great skew, now a shitty one
	#fazit: he is tweeting more and more lol





def tweetCountOverTime(daysPerMonth=7,globalRegularization = False):
	# epochCount = 50;
	tweets = getTweetsFromDB(purePres=True, conditions=["president"],returnParams=["publishTime"],orderBy="publishTime asc")



	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)


	firstPublishTime = tweets[0][0]
	lastPublishTime = tweets[-1][0]
	totalSeconds = (lastPublishTime-firstPublishTime).total_seconds()
	epochTime = daysPerMonth*60*60*24;
	print(epochTime);
	print("total epoch time in days: ",epochTime/(3600*24))
	ns = [1, 2, 3, 4]


	#assign tweets to epochs
	epochs = []
	startingIndex = 0;
	endingTime = firstPublishTime + datetime.timedelta(seconds = epochTime);
	dates = []

	for index,tweet in enumerate(tweets):
		if tweet[0] > endingTime:
			epochs.append(tweets[startingIndex:index]);
			startingIndex = index;
			endingTime = endingTime + datetime.timedelta(seconds = epochTime)
			dates.append(str(tweet[0])[:4])
		if index == len(tweets)-1:
			break;

	# epochs.append(tweets[startingIndex:])
	dates.append(str(tweets[-1][0])[:4])

	showEveryNthDate = 2;
	newDates = dates.copy()

	dates = [date if index in [5+int(len(dates)/3.3 * x) for x in list(range(4))] else "" for index, date in enumerate(dates[1:])];
	plt.gca().xaxis.grid(False)
	print("dizzle: ",dates)
	epochCount = len(epochs);

	x = list(range(epochCount));
	y = [len(epoch)/daysPerMonth for epoch in epochs];

	# lastYear = 2012
	# firstIndicesOfYear = []
	# ticksIndiciesOfYear = []
	# for index, month in enumerate(dates):
	# 	if month.year >= lastYear:
	# 		firstIndicesOfYear.append(index)
	# 		ticksIndiciesOfYear.append(str(lastYear));
	# 		lastYear += 1;



	plt.xticks([i for i in range(len(dates)) if len(dates[i]) > 1],[date for date in dates if len(date)>1],fontsize=15);
	plt.yticks(fontsize=15)
	fitLineCoefficients = np.polyfit(x, y, 1, full=False)
	slope = fitLineCoefficients[0];
	intercept = fitLineCoefficients[1]
	pearson = pearsonCorrelationCoefficient(y, [slope*i+intercept for i in x])
	plt.ylabel("Tweets per Day",fontsize=15)
	# plt.xlabel("Time")
	plt.plot(x,[slope*i+intercept for i in x],label="ols")
	plt.title("Tweets per Day Increases with Time (Correlation Coefficient "+str(round(pearson,2))+")",fontsize=20)
	plt.scatter(x,y);

	plt.show();
	exit();

