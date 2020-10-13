from .basisFuncs import *
from .part3funcs import normalize_headline
from .visualization import graphTwoDataSetsTogether;
from .stats import ols,pearsonCorrelationCoefficient;
from matplotlib.lines import Line2D


def determineYearOfTweet(minTime,maxTime,years,t):
	timePerYear = (maxTime-minTime)/years;
	maxTime = minTime + timePerYear;
	for i in range(years):
		if minTime <= t <= maxTime:
			return i;
		minTime = maxTime;
		maxTime += timePerYear
	return -1;

def determineDayOfWeek(dt):
	return dt.weekday();

def determineSegmentOfDay(dt: datetime.datetime, numSegments = 12):
	segSize = 24/numSegments;
	segments = [segSize*i for i in range(numSegments+1)];
	for i in range(len(segments)):
		if segments[i] <= dt.hour <= segments[i+1]:
			return i;
		



def graphPopularityByWeekday(tweets, years = 4):
	tweets.sort(key = lambda t: t[-1]);
	tweets = list(map(lambda t: t[:4] + (t[4] + datetime.timedelta(hours=-5),), tweets))

	days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)

	axPop = plt.subplot(1, 1, 1)
	# axNum = plt.subplot(2, 1, 2)
	# axProd = plt.subplot(3, 1, 3)
	axes = [axPop]
	
	startingDate = tweets[0][-1]

	totalTime = tweets[-1][-1]-tweets[0][-1];
	if years == 4:
		timePerYear = datetime.timedelta(days=365)
	else:
		timePerYear = totalTime/years;
	endingDate = startingDate + timePerYear
	mediansByYear = []
	numTweetsByYear = []
	custom_lines = []
	for year in range(years):
		# fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		# fig.subplots_adjust(left=0.06, right=0.94)
		theseTweets = [t for t in tweets if startingDate <= t[-1] <= endingDate]
		daysFavCounts = [[] for _ in range(len(days))]

		for tweet in theseTweets:
			dt = tweet[-1];
			dayOfWeek = determineDayOfWeek(dt);
			daysFavCounts[dayOfWeek].append(tweet[0]);
		for d in daysFavCounts:
			print(len(d));

		numTweets = [len(favs) for favs in daysFavCounts]
		numTweetsByYear.append(numTweets)
		medians = [np.median(favs) for favs in daysFavCounts]

		mediansByYear.append(medians);
		idx = np.argsort(medians);
		idxDays = [days[i] for i in idx];
		medians = [medians[i] for i in idx];
		favsSum = np.sum([len(favs) for favs in daysFavCounts]);
		sizesPop = [2000*len(favs)/favsSum for favs in daysFavCounts]

		# favsSum = np.sum([len(favs) for favs in daysFavCounts]);
		# sizesPop = [2000 * len(favs) / favsSum for favs in daysFavCounts]


		myColor = randomHexColor()

		# 	axPop.plot([i for i in range(len(mediansByDay[i]))],mediansByDay[i],label=days[i], markerfacecolor='#'+str(i*1000));
		# 	axNum.plot([i for i in range(len(numTweetsByDay[i]))],numTweetsByDay[i],label=days[i], markerfacecolor='#'+str(i*1000))
		# 	axProd.plot([i for i in range(len(numTweetsByDay[i]))], mediansByDay[i]/numTweetsByDay[i], label=days[i], markerfacecolor='#' + str(i * 1000))

		axPop.scatter([i for i in range(len(medians))], medians, s=numTweets, c=myColor)
		axPop.plot([i for i in range(len(medians))], medians, c=myColor, linewidth=0.5);

		# axNum.scatter([i for i in range(len(numTweets))], numTweets, s=numTweets, c=myColor)
		# axNum.plot([i for i in range(len(numTweets))], numTweets, c=myColor, linewidth=0.5);

		# axProd.scatter([i for i in range(len(numTweets))],[medians[i]/numTweets[i] for i in range(len(numTweets))],c=myColor)
		# axProd.plot([i for i in range(len(numTweets))], [medians[i] / numTweets[i] for i in range(len(numTweets))], linewidth=0.5,c=myColor)

		custom_line = Line2D([0], [0], color=myColor, lw=4)
		custom_lines.append(custom_line)


		startingDate = endingDate;
		endingDate += timePerYear;
		print("order: ",[x for x in zip(idxDays, medians)]);


	mediansByDay = list(np.array(mediansByYear).T)
	numTweetsByDay = list(np.array(numTweetsByYear).T)





	# for i in range(len(mediansByDay)):
	# 	axPop.plot([i for i in range(len(mediansByDay[i]))],mediansByDay[i],label=days[i], markerfacecolor='#'+str(i*1000));
	# 	axNum.plot([i for i in range(len(numTweetsByDay[i]))],numTweetsByDay[i],label=days[i], markerfacecolor='#'+str(i*1000))
	# 	axProd.plot([i for i in range(len(numTweetsByDay[i]))], mediansByDay[i]/numTweetsByDay[i], label=days[i], markerfacecolor='#' + str(i * 1000))

	# for ax in axes:
	axPop.set_xticks([])

	axPop.set_xticks([i for i in range(len(days))])
	axPop.set_xticklabels(days)
	# axPop.set_xlabel("Days of the Week")
	axPop.set_ylabel('Median Favourite Count')
	axPop.set_title("Popularity of Tweets On Different Days of the Week")
	# axProd.set_title("Popularity of Tweets On Different Days of the Week")
	# axPop.set_title("Number of Tweets on Different Days of the Week")

	# plt.xticks([i for i in range(len(days))], days)
	# plt.xlabel("Days of the Week")
	# plt.ylabel('Median Favourite Count')
	if years > 1:
		axPop.legend(custom_lines, ["Year " + str(i + 1) for i in range(len(custom_lines))], loc="upper left");
	plt.tight_layout()
	plt.show();

def toTwoDigitsStr(n):
	n = int(n);
	if n < 10:
		return ("0"+str(n))
	return str(n)

def graphPopularityByDayTime(tweets, years=5,numSegments = 8):
	tweets.sort(key=lambda t: t[4]);
	tweets = list(map(lambda t: t[:4]+(t[4] + datetime.timedelta(hours=-5),),tweets))
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)

	segSize = 24 / numSegments;
	segments = [str(int(segSize * i))+":"+toTwoDigitsStr(int(segSize*i*100*(6/10))%60)+" til " + str(int(segSize * (i+1)))+":"+toTwoDigitsStr(int(segSize*(i+1)*100*(6/10))%60)+" til " for i in range(numSegments)];
	print(segments);


	startingDate = tweets[0][4]

	totalTime = tweets[-1][4] - tweets[0][4];


	if years == 4:
		timePerYear = datetime.timedelta(days=365)
	else:
		timePerYear = totalTime / years;
	endingDate = startingDate + timePerYear
	mediansByYear = []
	sizesByYear = []
	custom_lines = []

	for year in range(years):

		theseTweets = [t for t in tweets if startingDate <= t[4] <= endingDate]
		segmentsFavCounts = [[] for _ in range(len(segments))]

		for tweet in theseTweets:
			dt = tweet[4];
			segment = determineSegmentOfDay(dt,numSegments);
			segmentsFavCounts[segment].append(tweet[0]);
		for d in segmentsFavCounts:
			print(len(d));

		medians = [np.median(favs) for favs in segmentsFavCounts]
		favsSum = np.sum([len(favs) for favs in segmentsFavCounts]);
		sizes = [2000*len(favs)/favsSum for favs in segmentsFavCounts]

		mediansByYear.append(medians);
		idx = np.argsort(medians);
		sizesByYear.append(sizes)
		idxSegs = [segments[i] for i in idx];
		medians = [medians[i] for i in idx];

		myColor = randomHexColor()
		plt.scatter([i for i in range(len(medians))],medians,s=sizes,c=myColor)
		plt.plot([i for i in range(len(medians))], medians, c=myColor, linewidth=0.5);

		custom_line = Line2D([0], [0], color=myColor, lw=4)
		custom_lines.append(custom_line)

		# plt.boxplot(daysFavCounts, notch=False, showfliers=False)
		# plt.title("n00b")

		# plt.xticks([x+1 for x in range(len(daysFavCounts))],[days[i] +" " + str(np.median(daysFavCounts[i])) for i in range(len(daysFavCounts))])
		# plt.show()
		startingDate = endingDate;
		endingDate += timePerYear;
		print("order: ", [x for x in zip(idxSegs, medians)]);

	mediansBySegment = list(np.array(mediansByYear).T)
	sizesBySegment = list(np.array(sizesByYear).T)
	#
	# for i in range(len(mediansBySegment)):
	# 	myColor = randomHexColor()
	# 	plt.scatter([j for j in range(len(mediansBySegment[i]))], mediansBySegment[i],c=myColor,s=sizesBySegment[i]);
	# 	plt.plot([j for j in range(len(mediansBySegment[i]))], mediansBySegment[i],c=myColor,linewidth=0.5);
	#
	# 	custom_line = Line2D([0], [0], color=myColor, lw=4)
	# 	custom_lines.append(custom_line)

	plt.xticks([i for i in range(len(segments))],segments)
	plt.xlabel("Times of Tweet Sending (Eastern Time)")
	plt.ylabel('Median Favourite Count')
	if years > 1:
		plt.legend(custom_lines, ["Year " + str(i+1) for i in range(len(custom_lines))],loc="lower right");
	plt.title("Favourite Counts by Time of Day")
	plt.show();
	###transpose the x values


def graphPopularityByTweetsPerDay(tweets, numYears = 4):
	tweets.sort(key=lambda tup: tup[-1])
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)
	favsPerDayCount = {}
	
	startingDate = tweets[0][-1]
	totalTime = tweets[-1][-1] - tweets[0][-1];
	if numYears == 4:
		timePerYear = datetime.timedelta(days=365)
	else:
		timePerYear = totalTime / numYears;
	endingDate = startingDate + timePerYear
	for year in range(numYears):
		theseTweets = [t for t in tweets if startingDate <= t[-1] <= endingDate]
		startingTime = theseTweets[0][-1]
		currentTime = startingTime;
		totalDays = (theseTweets[-1][-1]-startingTime).days
		index = 0;
		while currentTime < datetime.datetime.now():
			tweetsForDay = [t for t in theseTweets if currentTime <= t[-1] <= currentTime + datetime.timedelta(days=1)];
			if len(tweetsForDay) == 0:
				mean = 0;
			else:
				mean = np.mean([t[0] for t in tweetsForDay])
			try:
				favsPerDayCount[len(tweetsForDay)].append(mean);
			except:
				favsPerDayCount[len(tweetsForDay)] = [mean];

			currentTime += datetime.timedelta(days=1)
			if index%20 == 0:
				print(index/totalDays);
			index += 1;

		favsPerDayCountList = [0]*(max(favsPerDayCount.keys())+1);
		lenFavsPerDayList = [0]*(max(favsPerDayCount.keys())+1);
		for count in favsPerDayCount.keys():
			if count == 0:
				continue;
			try:
				favsPerDayCountList[count] = np.median(favsPerDayCount[count]);
				# favsPerDayCountList[count] = np.mean(favsPerDayCount[count]);
				lenFavsPerDayList[count] = len(favsPerDayCount[count])*3;

			except:
				print(count, len(favsPerDayCountList));
		# plt.xticks([i for i in range(len(favsPerDayCountList))],[str(i) + "\n("+str(len(favsPerDayCount[i])) + ")" if i in favsPerDayCount else "0\n(0)" for i in range(len(favsPerDayCountList))]);
		myColor = randomHexColor()
		plt.scatter([i for i in range(len(favsPerDayCountList))],favsPerDayCountList,c=myColor,s=lenFavsPerDayList);
		plt.plot([i for i in range(len(favsPerDayCountList))], favsPerDayCountList, c=myColor,linewidth=0.5);
		plt.title("Popularity of Tweets Grouped by Tweets per Day")
		popularityForDay1Index = 1
		popularityForDay15Index = 0;
		for i in range(1,len(favsPerDayCountList)):
			if favsPerDayCountList[i] == 0:
				popularityForDay15Index = i
				break;
		if popularityForDay15Index == 0:
			popularityForDay15Index = len(favsPerDayCountList);
		m, c = ols([i for i in range(popularityForDay15Index-popularityForDay1Index)],favsPerDayCountList[popularityForDay1Index:popularityForDay15Index])

		pearson = pearsonCorrelationCoefficient([x*m+c for x in range(popularityForDay1Index,popularityForDay15Index)], favsPerDayCountList[popularityForDay1Index:popularityForDay15Index])
		if numYears != 4:
			plt.plot([1,15],[1*m+c,15*m+c],c=myColor,linewidth=2.5,label="Epoch " + str(year) +"; Regression Correlation Coefficient "+str(round(pearson,2))+" (Pearson)")
		else:
			plt.plot([1, 15], [1 * m + c, 15 * m + c], c=myColor, linewidth=2.5, label="Year " + str(year+1) + " of Presidency; Regression Correlation Coefficient " + str(round(pearson, 2)) + " (Pearson)")


		startingDate = endingDate;
		endingDate += timePerYear;
	plt.legend()
	plt.xlabel('Number of Tweets per Day')
	plt.ylabel('Median Favourite Count')
	plt.ylim(60000,120000)
	plt.show();



def graphPopularityOfTweetsInSpurts():
	'''
	self: 2
	rando: 1
	None: 0
	'''

	checkup = False
	selfReplies = getTweetsFromDB(purePres=True, conditions=["isReply > 1"],returnParams=["favCount", "cleanedText, isReply,id, publishTime"], orderBy="publishTime desc")

	allTweets = getTweetsFromDB(purePres=True, returnParams=["favCount", "cleanedText, isReply,id, publishTime"], orderBy="publishTime asc")


	groups = [];

	groupIds = []
	chainsOfTweets = [];
	
	for index, t1 in enumerate(selfReplies):
		added = True;
		group = {t1};
		while added:
			interGroup = group.copy()
			added = False
			for t2 in selfReplies:
				if t2 not in group:
					for t in group:
						if t[2]==t2[3] or t[3] == t2[2]:
							interGroup.add(t2)
							added = True;
			group = interGroup

		theseGroupIds = [g[3] for g in group]

		theseGroupIds.sort();
		if theseGroupIds not in groupIds:
			groups.append(group);
			groupIds.append(theseGroupIds);
		else:
			pass;
			# print("we've reached another recap!")
	if checkup:
		allIds = []
		seenIds = set();
		summy = 0;
		for g in groups:
			summy += len(g);
			ids = [t[3] for t in g];
			for id in ids:
				if id in seenIds:
					print("gadzuk!", id);
				seenIds.add(id);
			allIds.extend(ids);
		print(summy,len(selfReplies))
		exit()

	groupsAsLists = []
	for group in groups:
		tweetIds = [t[3] for t in group];
		groupAsList = []
		brokenRoot = False;
		for t in group:
			inReplyTo = t[2]
			if t[2] not in tweetIds:
				try:
					tweetFromAllTweets = [t for t in allTweets if t[3] == inReplyTo][0]
				except:
					brokenRoot = True;
					print("beforeRoot: ",t[3]," root: ",inReplyTo);
					break;
				group.add(tweetFromAllTweets)
				groupAsList.append(tweetFromAllTweets)
				currentTweet = tweetFromAllTweets;
				break;
		if brokenRoot:
			continue;
		else:
			while len(groupAsList) < len(group):
				nextTweet = [t for t in group if t[2] == currentTweet[3]][0] #tweet s.t. it is in reply to current tweet
				groupAsList.append(nextTweet)
				currentTweet = nextTweet;
			groupsAsLists.append(groupAsList);


	lens ={};
	for chain in groupsAsLists:
		try:
			lens[len(chain)] += 1;
		except:
			lens[len(chain)] = 1;



	len2chains = [chain for chain in groupsAsLists if len(chain) == 2];
	len3chains = [chain for chain in groupsAsLists if len(chain) == 3];
	len3chainsFirstThird = [chain for chain in groupsAsLists if len(chain) == 3];
	len3chainsFirstSecond = [chain for chain in groupsAsLists if len(chain) == 3];
	len3chainsSecondThird = [chain for chain in groupsAsLists if len(chain) == 3];

	print(lens);
	for lenny in lens.keys():
		print("len: ",lenny,lens[lenny]/len(groupsAsLists))


	lenNot2chains = [chain for chain in groupsAsLists if len(chain) != 2];
	len2chains.sort(key=lambda chain: chain[1][0] - chain[0][0])
	len3chainsFirstThird.sort(key=lambda chain: chain[2][0] - chain[0][0])
	len3chainsFirstSecond.sort(key=lambda chain: chain[1][0] - chain[0][0])
	len3chainsSecondThird.sort(key=lambda chain: chain[2][0] - chain[1][0])

	# print(len2chains[:2]);
	diffs = []
	percentDiffs = []
	for chain in len2chains:

		diff = chain[1][0] - chain[0][0];
		percentDiff = diff/(2*(chain[1][0]+chain[0][0]))
		diffs.append(diff);
		percentDiffs.append(percentDiff);

	percentDiffsFirstThird = []
	percentDiffsSecondThird = []
	percentDiffsFirstSecond = []
	diffsFirstThird = []
	diffsSecondThird = []
	diffsFirstSecond = []

	for chainIndex in range(len(len3chains)):
		chainFirstThird = len3chainsFirstThird[chainIndex]
		chainSecondThird = len3chainsSecondThird[chainIndex]
		chainFirstSecond = len3chainsFirstSecond[chainIndex]
		diffFirstThird = chainFirstThird[2][0] - chainFirstThird[0][0];
		print("chainFirstThird: ",[c[0] for c in chainFirstThird])
		diffSecondThird = chainSecondThird[2][0] - chainSecondThird[1][0];
		diffFirstSecond = chainFirstSecond[1][0] - chainFirstSecond[0][0];
		percentDiffFirstThird = diffFirstThird / (2 * (chainFirstThird[2][0] + chainFirstThird[0][0]))
		percentDiffSecondThird = diffSecondThird / (2 * (chainSecondThird[2][0] + chainSecondThird[1][0]))
		percentDiffFirstSecond = diffFirstSecond / (2 * (chainFirstSecond[1][0] + chainFirstSecond[0][0]))

		percentDiffsFirstThird.append(percentDiffFirstThird);
		percentDiffsFirstSecond.append(percentDiffFirstSecond);
		percentDiffsSecondThird.append(percentDiffSecondThird);

		diffsFirstThird.append(diffFirstThird);
		diffsSecondThird.append(diffSecondThird);
		diffsFirstSecond.append(diffFirstSecond);


	# diffs.sort();
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.suptitle("Attenuating Popularity of Tweets in 2-chains", fontsize=20)

	axdiff = plt.subplot(2,1,1)
	axpdiff = plt.subplot(2, 1, 2)
	axdiff.scatter([i for i in range(len(diffs))], diffs,label="Tweet 2 - Tweet 1");
	axdiff.plot([0,len(diffs)],[0,0])
	axpdiff.scatter([i for i in range(len(percentDiffs))], percentDiffs);
	axpdiff.plot([0, len(percentDiffs)], [0, 0],label="no difference")
	axdiff.plot([0,len(percentDiffs)], [np.mean(diffs), np.mean(diffs)], label="mean ("+str(round(np.mean(diffs),2))+")");

	axpdiff.plot([0,len(percentDiffs)],[np.mean(percentDiffs),np.mean(percentDiffs)],label="mean ("+str(round(np.mean(percentDiffs),2))+")")
	axdiff.legend(loc="middle right")
	axpdiff.legend(loc="middle right")
	print("positive diffs: ",len([diff for diff in percentDiffs if diff <= 0])/len(percentDiffs))
	axdiff.set_title("Net Popularity Difference")
	axpdiff.set_title("Percentage Popularity Difference")

	plt.show();
	plt.clf()

	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.suptitle("difference between items in a 3-chain", fontsize=20)

	colors = [randomHexColor(), randomHexColor(), randomHexColor()]
	print("colors: ",colors)
	axdiff = plt.subplot(2, 1, 1)
	axpdiff = plt.subplot(2, 1, 2)

	axdiff = plt.subplot(2, 1, 1)
	axpdiff = plt.subplot(2, 1, 2)
	axdiff.scatter([i for i in range(len(diffsFirstThird))], diffsFirstThird,label="third - first",c=colors[0]);
	axdiff.scatter([i for i in range(len(diffsSecondThird))], diffsSecondThird, label="third - second",c=colors[1]);
	axdiff.scatter([i for i in range(len(diffsFirstSecond))], diffsFirstSecond, label="second - first",c=colors[2]);
	axdiff.plot([0, len(diffsFirstThird)], [0, 0])

	axpdiff.scatter([i for i in range(len(percentDiffsFirstThird))], percentDiffsFirstThird, label="third - first",c=colors[0]);
	axpdiff.scatter([i for i in range(len(percentDiffsSecondThird))], percentDiffsSecondThird, label="third - second",c=colors[1]);
	axpdiff.scatter([i for i in range(len(percentDiffsFirstSecond))], percentDiffsFirstSecond, label="second - first",c=colors[2]);

	axpdiff.plot([0, len(percentDiffsFirstThird)], [0, 0], label="no difference")

	axpdiff.plot([0, len(percentDiffsFirstThird)], [np.mean(percentDiffsFirstThird), np.mean(percentDiffsFirstThird)], label="mean (" + str(round(np.mean(percentDiffsFirstThird),2)) + ") third - first",c=colors[0])
	axpdiff.plot([0, len(percentDiffsFirstThird)], [np.mean(percentDiffsSecondThird), np.mean(percentDiffsSecondThird)], label="mean (" + str(round(np.mean(percentDiffsSecondThird),2)) + ") third - second",c=colors[1])
	axpdiff.plot([0, len(percentDiffsFirstThird)], [np.mean(percentDiffsFirstSecond), np.mean(percentDiffsFirstSecond)], label="mean (" + str(round(np.mean(percentDiffsFirstSecond),2)) + ") second - first",c=colors[2])
	axpdiff.legend()
	axdiff.legend()
	print("positive diffs: ", len([diff for diff in percentDiffs if diff <= 0]) / len(percentDiffs))

	plt.show();
	exit();






