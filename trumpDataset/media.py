import re
from .basisFuncs import *


def populateMediaTypeColumn():
	tweets =getPresidentTweetsWithMedia(mediaType=True);

	knownMediaTypes = {};
	tuples = []
	for tweet in tweets:
		tweetText = tweet[0];
		id = tweet[1]
		if re.match("https://twitter.com.*(status)+",tweetText):
			tuples.append(("twitterStatus",str(id)))
		elif "video" in tweetText or ".tv" in tweetText:
			tuples.append(("twitterStatus", str(id)))

def calculateMediaThreadTarget(threadNum, tweets, tupleDict):
	tuples = []
	# print("thread: " ,threadNum," has :",len(tweets), " tweets")
	counter = 0;
	for t in tweets:
		tweetText = t[0];
		id = t[1]
		media = extractMediaFromTweet(tweetText);
		root = getRootUrl(media)[:150]
		tuples.append((root,id));
		counter += 1
		if counter % 10 == 0:
			print("t: ",threadNum," ",counter/len(tweets))

	# print("thread : ",threadNum," fin")
	tupleDict[threadNum].extend(tuples)


def populateMediaColumn():
	# tweets = getPresidentTweetsWithMedia()
	# for t in tweets:
	# 	mycursor.execute("update tta set media = \"expecting\" where id = " + str(t[1]))
	# mydb.commit()
	updateFormula = "UPDATE "+mainTable+" SET media = %s WHERE id = %s";

	numThreads = 100;


	tweets = getPresidentTweetsWithMedia(expecting=True,limit=numThreads)
	counter = 1;
	while len(tweets)>0:
		# indices = [int(i * len(tweets) / numThreads) for i in range(numThreads)]
		indices = [i for i in range(len(tweets))]
		indices.append(len(tweets));
		tupleDict = {}
		threads = []

		for i in range(len(indices)-1):
			tupleDict[i] = [];
			myTweets = tweets[indices[i]:indices[i+1]]
			x = threading.Thread(target=calculateMediaThreadTarget,
								 args=( i,myTweets, tupleDict))
			threads.append(x);
			x.start()

		for i in range(len(threads)):
			t = threads[i]
			t.join(15)
			if t.is_alive():
				for tweet in tweets[indices[i]:indices[i+1]]:
					tupleDict[i].append(("checkme!",tweet[1]))
					print("damn ",tweet)
		print("all joined!")
		for i in range(len(threads)):
			myTuples = tupleDict[i]
			mycursor.executemany(updateFormula,myTuples);
			mydb.commit()
		print("all committed, done ",counter*numThreads," tweets")

		counter += 1
		tweets = getPresidentTweetsWithMedia(expecting=True, limit=numThreads)

def removeMediaFromTweet(tweet):
	try:
		tweet = tweet + " "
		beginningIndex = tweet.index("https://");
		endingIndex = tweet[beginningIndex:].index(" ");
		tweetSansMedia = tweet[:beginningIndex]+tweet[endingIndex:]
		return tweetSansMedia;
	except:
		return tweet



def getPresidentTweetsWithMedia(expecting = False, mediaType = False , limit = 100):
	if expecting:
		mycursor.execute("SELECT tweetText, id FROM "+mainTable+" where media = \"expecting\" limit "+str(limit))
	elif mediaType:
		mycursor.execute("SELECT tweetText, id FROM "+mainTable+" where mediaType <> \"none\"")
	else:
		mycursor.execute("SELECT tweetText, id FROM "+mainTable+" where tweetText like \"%https://%\"")
	tweetsWithMedia = mycursor.fetchall()
	return tweetsWithMedia


