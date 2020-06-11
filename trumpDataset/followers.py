from .visualization import computeXandYForPlotting,graphTwoDataSetsTogether
from .basisFuncs import *


def foo():
	print("bar")

def crawlFollowersThreadTarget(threadNum, dates, tupleDict, totalNumberOfDates):

	tuples = []
	seenDates = set()

	queryTimes = 0
	queries = 0;
	failed = False
	for d in dates:
		dDateTime = timeStampToDateTime(d);
		if dDateTime in seenDates:
			continue;
		try:
			url = "https://web.archive.org/web/" + str(d) + "/twitter.com/realdonaldtrump"
			t = time.time()
			text = requests.get(url).text
			queryTimes += time.time() - t
			queries += 1;
			followersCountAppearancesList = re.findall("followers_count&quot;:[0-9]+,", text)
			assert len(followersCountAppearancesList) == 1
			followersCount = re.findall("[0-9]+", followersCountAppearancesList[0])[0]
			tuples.append((dateTimeToMySQLTimeStamp(dDateTime),followersCount))
			seenDates.add(dDateTime);
			if failed:
				print("thread: ",threadNum," recuperated ",dDateTime);
				failed = False
		except:
			print("seen a failure for ",dDateTime, d);
			failed = True

		if queries % 20 == 0:
			print("thread: ",threadNum," average query time: ", queryTimes / queries, " percent fin: ",len(seenDates)/totalNumberOfDates)

	tupleDict[threadNum].extend(tuples);
	print("finished thread :",threadNum);



def crawlFollowerCounts():
	'''
	timeStamp format: YYYYMMDDhhmmss
	:return:
	'''

	mycursor.execute("delete from followersByDay")
	data = requests.get(
		url="http://web.archive.org/cdx/search/cdx?url=twitter.com/realdonaldtrump&limit=150000&output=json&from=20170220&fl=timestamp&filter=statuscode:200").json()

	#returns ~37047, so no limit
	params = data[0]
	data = [d[0] for d in data[1:]]

	followerCountsByDay = {}
	queryTimes = 0;
	queries = 0;
	numThreads = 10;
	tupleDict = {}
	datetimeDict = {}

	for d in data[1:]:
		correspondingObject = timeStampToDateTime(d)

		try:
			datetimeDict[correspondingObject].append(d)

		except:

			datetimeDict[correspondingObject] = [d]
	dateObjects = list(datetimeDict.keys())
	dateObjects.sort()

	threads = []
	for i in range(numThreads):
		tupleDict[i] = []
		partitions = [int(i*len(dateObjects)/numThreads) for i in range(numThreads+1)]
		dateObjectSet = dateObjects[partitions[i]:partitions[i+1]]
		dateSet = []
		for dateObject in dateObjectSet:
			dateSet.extend(datetimeDict[dateObject])
		x = threading.Thread(target=crawlFollowersThreadTarget,
							 args=(i, dateSet, tupleDict, len(dateObjectSet)))
		threads.append(x);
		x.start()
	for thread in threads:
		thread.join();

	insertFormula = "INSERT INTO followersByDay (day, followerCount) VALUES (%s, %s)"
	print(tupleDict)
	for threadNumber in tupleDict.keys():
		tuples = tupleDict[threadNumber]
		mycursor.executemany(insertFormula,tuples)
		mydb.commit()
	print("fin",tupleDict)

def getFollowerCount(timeFrame = "purePres"):
	if timeFrame != "purePres":
		raise("not yet implemented lol")


	mycursor.execute("select day, followerCount from followersByDay where day > \"2016:11:01 00:00:00\"")
	followers = mycursor.fetchall()

	return followers



def graphFollowerCountByFavCount():
	daysPerMonth = 30;
	followersCounts = getFollowerCount()

	favs = getTweetsFromDB(purePres=True,returnParams=["cleanedText","publishTime","rtCount","favCount"],orderBy="publishTime asc")

	avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, tweetIndicesByMonth = computeXandYForPlotting(favs,daysPerMonth=daysPerMonth);

	assert(months[0].day == followersCounts[0][0].day and months[0].month == followersCounts[0][0].month and months[0].year == followersCounts[0][0].year)

	followersCounts = followersCounts[::daysPerMonth]
	minLen = min(len(months),len(followersCounts));

	justFollowerCounts = [c[1] for c in followersCounts[:minLen]]

	mix = max(justFollowerCounts)
	print(mix)

	graphTwoDataSetsTogether(avgFavCounts[:minLen],"favCounts",justFollowerCounts,"follower counts")