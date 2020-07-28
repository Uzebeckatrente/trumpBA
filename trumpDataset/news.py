
from .basisFuncs import *
from .part3funcs import normalize_headline

import datetime
from pynytimes import NYTAPI
import time as time
import pytz
nyt = NYTAPI(os.getenv("nytimesPythonApiKey"));

def populateNewsRatioColumn():

	mycursor.execute("update " + mainTable + " set newsRatio = 0");
	mydb.commit();
	updateFormula = "UPDATE " + mainTable + " SET newsRatio = %s WHERE id = %s";

	allTweets = getTweetsFromDB(returnParams=["favCount","cleanedText","id","publishTime"],purePres=True);

	tweets = [t for t in allTweets if len(t[1]) > 1];
	ratios = [];
	tuples = [];
	favCounts = [t[0] for t in tweets]
	for i, t in enumerate(tweets):
		ratio = determineWhetherTweetWasInfluencedByNewsRatio(t);
		tuples.append((str(ratio),t[2]));
		ratios.append(ratio);
		if i%50 == 0:
			print(i/len(tweets));
			
	mycursor.executemany(updateFormula,tuples);
	mydb.commit();

	idx = np.argsort(ratios);
	ratios = [ratios[i] for i in idx];
	favCounts = [favCounts[i] for i in idx];
	plt.plot(ratios,favCounts);
	plt.show();



def populateNewsHeadlinesTable():
	est = pytz.timezone('US/Eastern')
	utc = pytz.utc
	fmt = '%Y-%m-%d %H:%M:%S'


	mycursor.execute("delete from newsHeadlines")
	mydb.commit();

	insertFormula = "INSERT INTO newsHeadlines (publishTime, topic, headline) VALUES (%s, %s, %s)"
	startingDateTime = datetime.datetime(2017, 1, 20)
	tuples = [];
	dirtyHeadlines = []
	topics = []
	timestamps = []
	lens = []

	while startingDateTime < datetime.datetime.now():
		data = nyt.archive_metadata(
			date=startingDateTime
		)


		dateTimes = {};
		maxDate = startingDateTime;
		for index, d in enumerate(data):
			try:
				startTime = time.time();
				# print(d);
				date = d["pub_date"];
				keywords = d["keywords"];
				doc = " ".join([k["value"] for k in keywords]);
				dirtyHeadlines.append(doc);
				startNormTime = time.time();
				doc = normalize_headline(doc);
				doc = doc[:1000];
				lenny = len(doc);
				lens.append(lenny);
				# if index in [487,488,489]:
	
				endNormTime = time.time();
				dateAndTime = date.split("T");
				try:
					topic = d["section_name"];
				except:
					topic = d["type_of_material"]
				date = dateAndTime[0]
				dateTokens = date.split("-");
				dateDateTime = datetime.datetime(int(dateTokens[0]), int(dateTokens[1]), int(dateTokens[2]));
				if dateDateTime > maxDate:
					maxDate = dateDateTime;
				timeStampTokens = dateAndTime[1].split("+");
				clockTime = timeStampTokens[0]
				clockTimeTokens = clockTime.split(":");
				hours = clockTimeTokens[0];
				minute = clockTimeTokens[1];
				timeZone =  timeStampTokens[1]
				dateDateTime += datetime.timedelta(hours=int(hours), minutes=int(minute));
				if timeZone != "0000":
					dateDateTime -= datetime.timedelta(hours=int(int(timeZone)/100)-5);
					raise Exception("fuck me i can't fuckin do it shittt!")
				else:
					# pass
					dateTimeMySQL = dateDateTime.astimezone(est).strftime(fmt)
	
					# dateDateTime -= datetime.timedelta(hours=5);
	
				# dateTimeMySQL =dateTimeToMySQLTimeStampWithHoursAndMinutes(dateDateTime);
				# dirtyHeadlines.append(doc);
				# topics.append(topic)
				# timestamps.append(dateTimeMySQL);
				tuples.append((dateTimeMySQL,topic,doc));
			except:
				print("well we have a failure here, who really gives a damn ",d);


			try:
				dateTimes[dateDateTime] += 1;
			except:
				dateTimes[dateDateTime] = 1;
			if index % 500 == 0:
				print(index / len(data));
				# print((endNormTime-startNormTime)/(time.time()-startTime));


		try:
			lens.sort(reverse=True)
			# time.sleep(6);
			prev = str(startingDateTime)
			startingDateTime  = maxDate + datetime.timedelta(days = 1);
			post = str(startingDateTime);
			mycursor.executemany(insertFormula, tuples)
			mydb.commit();
			tuples = [];
			print("committed ",prev," to ",post);
		except Exception as e:
			print("broken starting at ",prev);
			print(":'(")
			raise(e)
			# exit(1);
		# break;
	# normalize_corpus = np.vectorize(normalize_headline);
	# cleanHeadlines = normalize_corpus(dirtyHeadlines);
	# tuples = [(timestamps[i],topics[i],cleanHeadlines[i]) for i in range(len(topics))];







	"2019-01-03T20:14:10+0000".split("T");
	

def determineWhetherTweetWasInfluencedByNewsRatio(tweet):
	plainText = tweet[1]
	tweetTextTokens = plainText.split(" ");
	timeStamp = tweet[-1];
	intersectTokens = set();
	headlines = getHeadlinesUpToNHoursBeforeTime(timeStamp, 12);
	for headline in headlines:
		headlineTokens = headline.split(" ");
		intersect = np.intersect1d(headlineTokens,tweetTextTokens)
		if len(intersect) > 0:
			# print("gadzuk!");
			intersectTokens.update(intersect);
	return len(intersectTokens)/len(tweetTextTokens);




def getHeadlinesUpToNHoursAfterTime(t: datetime.datetime, n: int = 12):
	endDate = t+datetime.timedelta(hours=n);
	mycursor.execute("select headline from newsHeadlines where publishTime >= \"" + str(t) + "\" and publishTime <= \"" + str(endDate)+"\"");
	headlines =mycursor.fetchall()
	# for h in headlines:
	# 	print(h);
	return [h[0] for h in headlines];


def getHeadlinesUpToNHoursBeforeTime(t: datetime.datetime, n: int = 12):
	startDate = t-datetime.timedelta(hours=n);
	mycursor.execute("select headline from newsHeadlines where publishTime <= \"" + str(t) + "\" and publishTime >= \"" + str(startDate)+"\"");
	headlines =mycursor.fetchall()
	# for h in headlines:
	# 	print(h);
	return [h[0] for h in headlines];
