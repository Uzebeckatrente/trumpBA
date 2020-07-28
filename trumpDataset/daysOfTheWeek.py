from .basisFuncs import *
from .part3funcs import normalize_headline
from .visualization import graphTwoDataSetsTogether;
from .stats import ols;

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

def determineSegmentOfDay(dt: datetime.datetime, numSegments = 4):
	segSize = 24/numSegments;
	segments = [segSize*i for i in range(numSegments+1)];
	for i in range(len(segments)):
		if segments[i] <= dt.hour <= segments[i+1]:
			return i;
		



def graphPopularityByWeekday(tweets, years = 5):
	tweets.sort(key = lambda t: t[-1]);
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)
	days = ["mon", "tues", "wed", "thurs", "fri", "sat", "sun"]
	
	startingDate = tweets[0][-1]

	totalTime = tweets[-1][-1]-tweets[0][-1];
	timePerYear = totalTime/years;
	endingDate = startingDate + timePerYear
	mediansByYear = []
	numTweetsByYear = []

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

		# plt.boxplot(daysFavCounts, notch=False, showfliers=False)
		# plt.title("n00b")
		#
		# plt.xticks([x+1 for x in range(len(daysFavCounts))],[days[i] +" " + str(np.median(daysFavCounts[i])) for i in range(len(daysFavCounts))])
		# plt.show()
		startingDate = endingDate;
		endingDate += timePerYear;
		print("order: ",[x for x in zip(idxDays, medians)]);


	mediansByDay = list(np.array(mediansByYear).T)
	numTweetsByDay = list(np.array(numTweetsByYear).T)



	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)

	axPop = plt.subplot(3, 1,1)
	axNum = plt.subplot(3, 1, 2)
	axProd = plt.subplot(3, 1, 3)

	for i in range(len(mediansByDay)):
		axPop.plot([i for i in range(len(mediansByDay[i]))],mediansByDay[i],label=days[i], markerfacecolor='#'+str(i*1000));
		axNum.plot([i for i in range(len(numTweetsByDay[i]))],numTweetsByDay[i],label=days[i], markerfacecolor='#'+str(i*1000))
		axProd.plot([i for i in range(len(numTweetsByDay[i]))], mediansByDay[i]/numTweetsByDay[i], label=days[i], markerfacecolor='#' + str(i * 1000))


	axNum.legend();
	plt.show();


def graphPopularityByDayTime(tweets, years=5):
	tweets.sort(key=lambda t: t[-1]);
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)

	numSegments = 6;
	segSize = 24 / numSegments;
	segments = [str(int(segSize * i)) for i in range(numSegments)];
	print(segments);


	startingDate = tweets[0][-1]

	totalTime = tweets[-1][-1] - tweets[0][-1];
	timePerYear = totalTime / years;
	endingDate = startingDate + timePerYear
	mediansByYear = []

	for year in range(years):
		fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
		fig.subplots_adjust(left=0.06, right=0.94)
		theseTweets = [t for t in tweets if startingDate <= t[-1] <= endingDate]
		segmentsFavCounts = [[] for _ in range(len(segments))]

		for tweet in theseTweets:
			dt = tweet[-1];
			segment = determineSegmentOfDay(dt,numSegments);
			segmentsFavCounts[segment].append(tweet[0]);
		for d in segmentsFavCounts:
			print(len(d));

		medians = [np.median(favs) for favs in segmentsFavCounts]
		mediansByYear.append(medians);
		idx = np.argsort(medians);
		idxSegs = [segments[i] for i in idx];
		medians = [medians[i] for i in idx];

		# plt.boxplot(daysFavCounts, notch=False, showfliers=False)
		# plt.title("n00b")
		#
		# plt.xticks([x+1 for x in range(len(daysFavCounts))],[days[i] +" " + str(np.median(daysFavCounts[i])) for i in range(len(daysFavCounts))])
		# plt.show()
		startingDate = endingDate;
		endingDate += timePerYear;
		print("order: ", [x for x in zip(idxSegs, medians)]);

	mediansBySegment = list(np.array(mediansByYear).T)
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)
	for i in range(len(mediansBySegment)):
		plt.plot([i for i in range(len(mediansBySegment[i]))], mediansBySegment[i], label=segments[i]);
	plt.legend();
	plt.show();


def graphPopularityByTweetsPerDay(tweets, numYears = 8):
	tweets.sort(key=lambda tup: tup[-1])
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)
	favsPerDayCount = {}
	
	startingDate = tweets[0][-1]
	totalTime = tweets[-1][-1] - tweets[0][-1];
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
				lenFavsPerDayList[count] = len(favsPerDayCount[count])*3;

			except:
				print(count, len(favsPerDayCountList));
		# plt.xticks([i for i in range(len(favsPerDayCountList))],[str(i) + "\n("+str(len(favsPerDayCount[i])) + ")" if i in favsPerDayCount else "0\n(0)" for i in range(len(favsPerDayCountList))]);
		myColor = randomHexColor()
		plt.scatter([i for i in range(len(favsPerDayCountList))],favsPerDayCountList,label="year " + str(year),c=myColor,s=lenFavsPerDayList);
		plt.plot([i for i in range(len(favsPerDayCountList))], favsPerDayCountList, c=myColor,linewidth=0.5);

		popularityForDay1Index = 1
		popularityForDay15Index = 0;
		for i in range(1,len(favsPerDayCountList)):
			if favsPerDayCountList[i] == 0:
				popularityForDay15Index = i
				break;
		if popularityForDay15Index == 0:
			popularityForDay15Index = len(favsPerDayCountList);
		m, c = ols([i for i in range(popularityForDay15Index-popularityForDay1Index)],favsPerDayCountList[popularityForDay1Index:popularityForDay15Index])


		plt.plot([1,15],[1*m+c,15*m+c],c=myColor,linewidth=2.5)

		startingDate = endingDate;
		endingDate += timePerYear;
	plt.legend()
	plt.show();



def graphPopularityOfTweetsInSpurts():
	'''
	self: 2
	rando: 1
	None: 0
	'''
	fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.suptitle("difference between second and first item in a 2-chain", fontsize=20)
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
			print("we've reached another recap!")
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



	len2chains = [chain for chain in groupsAsLists if len(chain) == 2];
	lenNot2chains = [chain for chain in groupsAsLists if len(chain) != 2];
	len2chains.sort(key=lambda chain: chain[1][0] - chain[0][0])
	print(len2chains[:2]);
	diffs = []
	percentDiffs = []
	for chain in len2chains:

		diff = chain[1][0] - chain[0][0];
		percentDiff = diff/(2*(chain[1][0]+chain[0][0]))
		diffs.append(diff);
		percentDiffs.append(percentDiff);
	# diffs.sort();

	axdiff = plt.subplot(2,1,1)
	axpdiff = plt.subplot(2, 1, 2)
	axdiff.scatter([i for i in range(len(diffs))], diffs);
	axdiff.plot([0,len(diffs)],[0,0])
	axpdiff.scatter([i for i in range(len(percentDiffs))], percentDiffs);
	axpdiff.plot([0, len(percentDiffs)], [0, 0])

	plt.show();
	exit();






