
from .basisFuncs import *
from .part3funcs import computeMostCommonnGrams,getnGramsWithOverMOccurences;
from .favs import loadnGramsWithFavsMinusMeanMedian,calculateAndStorenGramsWithFavsMinusMeanMedian;
from .stats import pearsonCorrelationCoefficient;



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


def compareVocabOverTime(epochCount=50,globalRegularization = False):
	# epochCount = 50;
	tweets = getTweetsFromDB(purePres=True, returnParams=["favCount","cleanedText, publishTime"],orderBy="publishTime asc")
	quickieGrams = ["coronavirus", "great", "fake news"]
	quickie = False



	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)


	firstPublishTime = tweets[0][2]
	lastPublishTime = tweets[-1][2]
	totalSeconds = (lastPublishTime-firstPublishTime).total_seconds()
	epochTime = totalSeconds/epochCount;
	print(epochTime);

	print("total epoch time in days: ",epochTime/(3600*24))
	ns = [1, 2, 3, 4]
	minCount = 30;


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

	minimumCountThreshholds = [10]#[5, 10, 50, 100, 200, 400, 800];
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

	dataFrameDict = {}
	for epoch1 in range(epochCount):
		dataFrameDict[epoch1] = []
		for epoch2 in range(epochCount):
			epoch1Grams = nGramsForAllEpochsRegularized[epoch1]
			epoch2Grams = nGramsForAllEpochsRegularized[epoch2]
			similarity = round(compareTwoEpochsByNTramTopo(epoch1Grams, epoch2Grams, allNgrams), 2);
			dataFrameDict[epoch1].append(similarity)
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



	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.tight_layout(pad=3.0)
	fig.subplots_adjust(left=0.06, right=0.94, bottom=0.02, top = 0.97)
	fig.tight_layout(pad=3.0)



	axInEveryEpoch = fig.add_subplot(411)
	axTopo = fig.add_subplot(412)
	axCount = fig.add_subplot(413)
	axSkew = fig.add_subplot(414)
	axes = [axInEveryEpoch, axTopo, axSkew, axCount]
	iterColor = 11;
	for nGram,topos in nGramTopoDict.items():

		# c = randomHexColor()
		iterColorString = hex(iterColor)[2:]
		c = "#"+(6-len(iterColorString))*'0'+iterColorString;
		print(c)
		axInEveryEpoch.set_title("topo for ngrams appearing in every epoch")
		axInEveryEpoch.plot(list(range(len(topos))),topos,color=c,label = nGram);
		axInEveryEpoch.scatter(list(range(len(topos))), topos, color=c);
		iterColor += int(16777215 / len(nGramTopoDict))



	iterColor = 11;
	for nGram, topos in nGramTopoDictAll.items():
		if not quickie or nGram in quickieGrams:

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
		if not quickie or nGram in quickieGrams:

			# c = randomHexColor()
			iterColorString = hex(iterColor)[2:]
			c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
			print(c)
			axCount.set_title("counts top " + str(topnFromEachEpoch) + " ngrams of each epoch")
			axCount.plot(list(range(len(counts))), counts, color=c, label=nGram);
			axCount.scatter(list(range(len(counts))), counts, color=c);

			iterColor += int(16777215 / len(nGramCountDictAll))


	iterColor = 11;
	for nGram, skews in nGramSkewDictAll.items():
		if not quickie or nGram in quickieGrams:
			# c = randomHexColor()
			iterColorString = hex(iterColor)[2:]
			c = "#" + (6 - len(iterColorString)) * '0' + iterColorString;
			print(c)
			axSkew.set_title("skews top " + str(topnFromEachEpoch) + " ngrams of each epoch")
			axSkew.plot(list(range(len(skews))), skews, color=c, label=nGram);
			axSkew.scatter(list(range(len(skews))), skews, color=c);
			iterColor += int(16777215 / len(nGramCountDictAll))



	for ax in axes:
		ax.plot([0, epochCount-1], [0, 0],'--', linewidth=5)
	axInEveryEpoch.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axInEveryEpoch,nGramsForAllEpochsRegularized))#
	axCount.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axCount,nGramsForAllEpochsRegularized))#
	axTopo.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axTopo,nGramsForAllEpochsRegularized))#
	axSkew.figure.canvas.mpl_connect("motion_notify_event", lambda event: on_plot_hover(event, axSkew,nGramsForAllEpochsRegularized))#
	#


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





def tweetCountOverTime(daysPerMonth=31,globalRegularization = False):
	# epochCount = 50;
	tweets = getTweetsFromDB(purePres=True, returnParams=["publishTime"],orderBy="publishTime asc")



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
			dates.append(str(tweet[0])[:10])
		if index == len(tweets)-1:
			break;

	epochs.append(tweets[startingIndex:])
	dates.append(str(tweets[-1][0])[:10])

	showEveryNthDate = 2;
	newDates = dates.copy()
	while len(newDates)-newDates.count("") > 15:
		newDates = [date if index % showEveryNthDate == showEveryNthDate-1 else "" for index, date in enumerate(dates[1:])];
		showEveryNthDate += 1;
		print("n00b")
	dates[1:] = newDates
	epochCount = len(epochs);

	x = list(range(epochCount));
	y = [len(epoch) for epoch in epochs];
	plt.xticks([i for i in range(len(dates)+1)],dates);

	fitLineCoefficients = np.polyfit(x, y, 1, full=False)
	slope = fitLineCoefficients[0];
	intercept = fitLineCoefficients[1]
	pearson = pearsonCorrelationCoefficient(y, [slope*i+intercept for i in x])
	plt.plot(x,[slope*i+intercept for i in x],label="ols")
	plt.title("number of tweets over time; correlation coefficient :"+str(round(pearson,2)))
	plt.plot(x,y);
	plt.show();
	exit();

