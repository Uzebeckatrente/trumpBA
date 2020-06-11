
from .basisFuncs import *
from .part3funcs import computeMostCommonnGrams,getnGramsWithOverMOccurences;
from .favs import loadnGramsWithFavsMinusMeanMedian,calculateAndStorenGramsWithFavsMinusMeanMedian;

def compareVocabOverTime(epochCount=50):
	# epochCount = 50;
	tweets = getTweetsFromDB(purePres=True, returnParams=["favCount","cleanedText, publishTime"],orderBy="publishTime asc")

	fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)
	axInEveryEpoch = fig.add_subplot(411)
	axTopo = fig.add_subplot(412)
	axCount = fig.add_subplot(413)
	axSkew = fig.add_subplot(414)
	axes = [axInEveryEpoch,axTopo,axSkew,axCount]

	firstPublishTime = tweets[0][2]
	lastPublishTime = tweets[-1][2]
	totalSeconds = (lastPublishTime-firstPublishTime).total_seconds()
	epochTime = totalSeconds/epochCount;
	print(epochTime);
	print("total epoch time in days: ",epochTime/(3600*24))
	ns = [1, 2, 3, 4]
	minCount = 60;


	#assign tweets to epochs
	epochs = []
	startingIndex = 0;
	endingTime = firstPublishTime + datetime.timedelta(seconds = epochTime);
	for index,tweet in enumerate(tweets):
		if tweet[2] > endingTime:
			epochs.append(tweets[startingIndex:index]);
			startingIndex = index;
			endingTime = endingTime + datetime.timedelta(seconds = epochTime)
			print("appendin")
			if len(epochs) == epochCount-1:
				break;

	epochs.append(tweets[startingIndex:])
	nGramsForAllEpochs = []


	for epochTweets in epochs:
		epochCleanTextAndFavs = [tweet[0:2] for tweet in epochTweets]
		tweetsHash = hashTweets(epochCleanTextAndFavs)


		nGramsForEpoch = []
		for n in ns:
			computeMostCommonnGrams(epochTweets, n);
			myNGrams = getnGramsWithOverMOccurences(n, minCount, tweetsHash)
			nGramsForEpoch.extend(myNGrams)
		nGramsForEpoch.sort(key=lambda tuple: tuple[1], reverse=True)

		calculateAndStorenGramsWithFavsMinusMeanMedian(ns, epochCleanTextAndFavs, False)
		nGramsWithFavsMinusMeanMedian, _ = loadnGramsWithFavsMinusMeanMedian(ns, tweetsHash)
		nGramsWithFavsMinusMeanMedianDict = {}
		for nGram in nGramsWithFavsMinusMeanMedian:
			nGramsWithFavsMinusMeanMedianDict[nGram[0]] = nGram
		for i in range(len(nGramsForEpoch)):
			e1 = nGramsForEpoch[i][0]
			e2 = nGramsForEpoch[i][1]
			medianDictEntry = nGramsWithFavsMinusMeanMedianDict[nGramsForEpoch[i][0]]
			e3 = medianDictEntry[nGramStorageIndicesSkew["skew"]]
			nGramsForEpoch[i] = (e1,e2,e3)
		nGramsForAllEpochs.append(nGramsForEpoch);



	print(Fore.RED,"\n\n\nnow sorted by skew\n\n\n",Style.RESET_ALL)

	for epoch in range(len(nGramsForAllEpochs)):
		nGramsForAllEpochs[epoch].sort(key = lambda tup: tup[2], reverse = True)

	minimumCountThreshholds = [10]#[5, 10, 50, 100, 200, 400, 800];
	for minimumCountThreshhold in minimumCountThreshholds:
		print(Fore.MAGENTA, "minimumCountThreshhold: ", minimumCountThreshhold, Style.RESET_ALL, "\n\n")
		for epoch in range(epochCount - 1):
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
		allNgrams.update([tup[0] for tup in nGramsForAllEpochs[i]])
		nGramsInAllEpochs.intersection_update([tup[0] for tup in nGramsForAllEpochs[i]])

		nGramsReg = [(tup[0], tup[1] / maxCount, tup[2] / maxAbsSkew, (tup[1] / maxCount) * (tup[2] / maxAbsSkew)) for tup in nGramsForAllEpochs[i]];

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




	topnFromEachEpoch = 20;
	nGramTopoDict = {nGram:[] for nGram in nGramsInAllEpochs}
	nGramTopoDictAll = {nGram : [] for nGram in allNgrams}
	nGramSkewDictAll = {nGram: [] for nGram in allNgrams}
	nGramCountDictAll = {nGram: [] for nGram in allNgrams}

	for nGram in allNgrams:
		for epoch in range(len(nGramsForAllEpochs)):
			topn_nGramsByTopo = [tup[0] for tup in nGramsForAllEpochsRegularizedTopos[epoch]][:topnFromEachEpoch]#
			topn_nGramsByCount = [tup[0] for tup in nGramsForAllEpochsRegularizedCounts[epoch]][:topnFromEachEpoch]#
			topn_nGramsBySkew = [tup[0] for tup in nGramsForAllEpochsRegularizedSkews[epoch]][:topnFromEachEpoch]#



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

			if nGram in topn_nGramsByCount:
				nGramCountDictAll[nGram].append(nGramsForAllEpochsRegularized[epoch][nGram][0])
			else:
				nGramCountDictAll[nGram].append(0);


	iterColor = 11;
	for nGram,topos in nGramTopoDict.items():
		# c = randomHexColor()
		iterColorString = hex(iterColor)[2:]
		c = "#"+(6-len(iterColorString))*'0'+iterColorString;
		print(c)
		axInEveryEpoch.set_title("topo for ngrams in every epoch")
		axInEveryEpoch.plot(list(range(len(topos))),topos,color=c,label = nGram);
		axInEveryEpoch.scatter(list(range(len(topos))), topos, color=c);
		iterColor += int(16777215 / len(nGramTopoDict))



	iterColor = 11;
	for nGram, topos in nGramTopoDictAll.items():
		# c = randomHexColor()
		iterColorString = hex(iterColor)[2:]
		c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
		print(c)

		axTopo.set_title("topo for top " + str(topnFromEachEpoch) + " ngrams of each epoch")
		axTopo.plot(list(range(len(topos))), topos, color=c, label=nGram);
		axTopo.scatter(list(range(len(topos))), topos, color=c);
		iterColor += int(16777215 / len(nGramTopoDictAll))

	iterColor = 11;
	for nGram, counts in nGramCountDictAll.items():
		# c = randomHexColor()
		iterColorString = hex(iterColor)[2:]
		c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
		print(c)
		axCount.set_title("counts for ngrams in every epoch")
		axCount.plot(list(range(len(counts))), counts, color=c, label=nGram);
		axCount.scatter(list(range(len(counts))), counts, color=c);

		iterColor += int(16777215 / len(nGramCountDictAll))


	iterColor = 11;
	for nGram, skews in nGramSkewDictAll.items():
		# c = randomHexColor()
		iterColorString = hex(iterColor)[2:]
		c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
		print(c)
		axSkew.set_title("skews for ngrams in every epoch")
		axSkew.plot(list(range(len(skews))), skews, color=c, label=nGram);
		axSkew.scatter(list(range(len(skews))), skews, color=c);
		iterColor += int(16777215 / len(nGramCountDictAll))



	for ax in axes:
		ax.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, ax))
		ax.plot([0, epochCount-1], [0, 0],'--', linewidth=5)
	#
	# axInEveryEpoch.legend()
	# axTopo.legend()
	# axSkew.legend()
	# axCount.legend()

	plt.show()


	print(nGramsInAllEpochs)


	#TODO: devise a measurement, topo, that combines count and skew to determine the most popular tweets in both count and skew
	#TODO: compare the topo between epochs
	#TODO: visualize the similarities between epochs somehow lol


	#fazit: he is tweeting more and more lol