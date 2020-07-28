
from hashlib import sha256 as hasher
import requests
import mysql.connector
import matplotlib.pyplot as plt
import datetime
import threading
import time
import pandas as pd
import pickle
import re
import datetime
from colorama import Fore, Style
import numpy as np
import spacy
nlp = spacy.load("en")
extraParamNames = ["allCapsRatios","daysOfWeek","timesOfDay","years"]

lastEpochAndGram = (None,None);

from .twitterApiConnection import *
relevancyDateThreshhold = "where publishTime > \"2016:11:01 00:00:00\""
#publishTime > "2017-02-20 00:00:00"

nGramStorageIndicesSkew = {"nGram":0, "count":1, "std":2, "skew":3, "favs":4}
nGramStorageIndicesProbs = {"nGram":0, "count":1, "std":2, "binProbs":3}


mydb = mysql.connector.connect(host="localhost",user="root",passwd="felixMySQL",database="trump");
mycursor = mydb.cursor(buffered=True)

mainTable = "tta2"
purePresConditions=["president","isRt = 0","deleted = 0", "favCount > 0","mediaType = \"none\""]

def flattenFolds(folds, holdoutIndex):
	if holdoutIndex != -1:
		train = folds[0:holdoutIndex] + folds[holdoutIndex + 1:];
		trainFlat = [item for sublist in train for item in sublist]
		holdOut = folds[holdoutIndex];
		return trainFlat, holdOut
	else:
		return [item for sublist in folds for item in sublist]

def getTweetById(id: int):
	return getTweetsById([id])

def getTweetsById(ids: [int]):
	offendingTweets = api.statuses_lookup(ids, tweet_mode= 'extended');
	return offendingTweets;

def computeBigrams(tweetWords):
	return [tweetWords[i] + " " + tweetWords[i + 1] for i in range(len(tweetWords) - 1)]

def splitTrainingTest(tweets, numFolds = 5):

	indices = np.random.permutation(len(tweets))
	foldsIndices = [indices[int(len(tweets) * (0.2 * i)):int(len(tweets) * (0.2 * (i + 1)))] for i in range(numFolds)]

	training_idx, test_idx = indices[:int(len(tweets) * 0.8)], indices[int(len(tweets) * 0.8):]
	training = []
	test = []
	folds = []
	for indices in foldsIndices:
		fold = []
		for i in indices:
			fold.append(tweets[i]);
		folds.append(fold);

	for i in range(len(tweets)):
		if len(tweets[i][1]) > 0:
			if i in training_idx:
				training.append(tweets[i])
			else:
				test.append(tweets[i])
	return training, test, folds

def splitTweetsByYear(tweets,numYears):
	tweets.sort(key = lambda t: t[4],reverse=False);
	totalTime = tweets[-1][4] - tweets[0][4];
	timePerYear = totalTime / numYears;
	startingDate = tweets[0][4]
	endingDate = startingDate + timePerYear;
	tweetsByYear = [];
	for i in range(numYears):
		theseTweets = [t for t in tweets if startingDate <= t[4] <= endingDate];
		tweetsByYear.append(theseTweets)
		startingDate = endingDate;
		endingDate += timePerYear;
	return tweetsByYear;

def hashTweets(tweets):
	tweets = [t[:2] for t in tweets];
	tweets.sort(key= lambda tup: tup[1]+str(tup[0]), reverse=True)
	myString = str(tweets);
	return hasher(myString.encode()).hexdigest()

def dateTimeToInt(obj, old = datetime.datetime(year=2017, day = 20, month = 2)):
	'''
	in order to be graphable abstractly
	:param obj:
	:return:
	'''
	return (obj-old).total_seconds()


def getTopiPercentOfList(i,l,ascending = True):
	if ascending:
		return l[len(l) - int(len(l) * (i / 100)):]
	else:
		return l[:int(len(l) * (i / 100))]

def getTopiElementsOfList(i,l,ascending = True):
	if ascending:
		return l[-i:]
	else:
		return l[:i]


def getTopBottomiPercentOfPurePresTweets(i,tweets = None):
	if tweets == None:
		tweets = getTweetsFromDB(purePres=True, orderBy="favCount asc",
								 returnParams=["favCount", "cleanedText"])

	return getTopiPercentOfList(i,tweets,True),getTopiPercentOfList(i,tweets,False)


def getTopBottomiTweetsOfPurePresTweets(i, tweets=None):
	if tweets == None:
		tweets = getTweetsFromDB(purePres=True, orderBy="favCount asc",
								 returnParams=["favCount", "cleanedText"])
	return getTopiElementsOfList(i,tweets,True),getTopiElementsOfList(i,tweets,False)


def getApprovalRatings():
	mycursor.execute(
		"select date, approveEstimate from approvalRatings order by date asc");
	ratings = mycursor.fetchall()
	return ratings;

def getFavsByNGram(keyword,tweets):

	relevantTweets = [t[0] for t in tweets if keyword in t[1]];
	return relevantTweets;
	# return [fav[0] for fav in favs]





def getRootUrl(url):
	try:
		session = requests.Session()  # so connections are recycled
		resp = session.head(url, allow_redirects=True)
		return resp.url;
	except:
		return "checkme!"

def timeStampToDateTime(ts):
	return datetime.datetime(int(ts[0:4]),int(ts[4:6]),int(ts[6:8]))

def paddify(s):
	if len(s) == 1:
		return "0"+s;

def dateTimeToMySQLTimeStampWithHoursAndMinutes(dt: datetime.datetime):
	return str(dt.year)+":"+str(dt.month)+":"+str(dt.day)+" "+str(dt.hour)+":"+str(dt.minute)+":00"

def dateTimeToMySQLTimeStamp(dt):
	return str(dt.year)+":"+str(dt.month)+":"+str(dt.day)+" 00:00:00"

def convertYYYYMMDDtoMySQLTimeStamp(date):
	return date[:4]+":"+date[4:6]+":"+date[6:]+" 00:00:00";

def camlify(string, firstLetterCapital = True):
	string = string.rstrip().lstrip()


	return string.replace(" ","_")

	toks = string.split(" ")
	ret = toks[0].lower()
	for s in toks[1:]:
		ret += s[0].upper()+s[1:].lower()
	return ret;

def on_plot_hover(event,plot,nGramsForAllEpochsRegularized):
	# Iterating over each data member plotted
	for curve in plot.get_lines():
		# Searching which data member corresponds to current mouse position
		if curve.contains(event)[0]:
			epoch = int(round(event.xdata,0));
			gram = curve.get_label();
			if gram[:5] == "_line":
				continue;
			global lastEpochAndGram;
			if (epoch,gram) == lastEpochAndGram:
				return;
			else:
				lastEpochAndGram=(epoch,gram)
			print("gram: ",gram," epoch: ",epoch," topo:",round(nGramsForAllEpochsRegularized[epoch][gram][2],2)," skew: ",round(nGramsForAllEpochsRegularized[epoch][gram][1],2)," count: ",round(nGramsForAllEpochsRegularized[epoch][gram][0],2))


	# for line in plot.get_lines():
	# 	print(line);

def randomHexColor():
	#generates a random color string
	# return '#D6FC36'
	return "#"+('000000'+hex(np.random.randint(16777216))[2:])[-6:].upper();

def getTweetsFromDB(n=-1,purePres = False, conditions=[], returnParams="*",orderBy = "publishTime desc"):#president = False, allParams = False,inclRts = False):
	'''
	returns all columns of latest n tweets of DJT
	:param n: -1 will return all tweets
	:param president: True will only return tweets sent while president
	:return: if allParams: [(tweetText, publishTime, rtCount, favCount, isRt, deleted, id),...];
			else: [tweetText,...]
	'''
	if purePres:
		conditions.extend(purePresConditions)

	if len(conditions)>0:
		whereString = "where "
		for index, condition in enumerate(conditions):
			if condition == "president":
				whereString += ("publishTime > \"2017-02-20 00:00:00\"")
				favCountUpThresh = datetime.datetime.now()-datetime.timedelta(days=0);
				mySqlTimeStamp = dateTimeToMySQLTimeStamp(favCountUpThresh);
				whereString += (" and publishTime <= \""+mySqlTimeStamp+"\"");
			else:
				whereString += condition
			if index < len(conditions)-1:
				whereString += " and "
	else:
		whereString = ""
	limiter = ""

	if n != -1:
		limiter = " limit " +str(n)

	if returnParams != "*":
		returnParams = ", ".join(returnParams)
	query = "select "+returnParams+" from "+mainTable+" "+whereString + " order by "+orderBy+" "+limiter;
	mycursor.execute(query)
	tweets = mycursor.fetchall()

	return tweets




def insertIntoDB(corpus):
	# corpus = corpus[:101]

	# mycursor.execute("delete from trumptwitterarchive");
	ind = 0;
	tuples = []
	insertFormula = "INSERT INTO "+mainTable+" (id, tweetText, publishTime, rtCount, favCount, isRt, deleted, mediaType, isReply, allCapsRatio) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s,%s)"
	# firstTweet =corpus[0]

	counter = 0;
	hundredTweetIds = [int(tweet['id_str\n']) for tweet in corpus[counter * 100:(counter + 1) * 100]];
	while len(hundredTweetIds)>0:
		print(counter*100/len(corpus))
		try:
			[int(tweet['id_str\n']) for tweet in corpus[counter*100:(counter+1)*100]];
		except:
			print("broken on : ",corpus[counter*100:(counter+1)*100])
			break;

		hundredTweetsFromTTA = [tweet for tweet in corpus[counter * 100:(counter + 1) * 100]];
		# hundredTweetIds = [1247641140507049986]#,1257656121898283010, 1257736426206031874,1257753237790158848, 1257714716492722176, 1194615176823218176]
		tweets = getTweetsById(hundredTweetIds)
		returnedIds = [tweet.id for tweet in tweets]


		for tweet in tweets:

			###not deleted

			tweetText = tweet.full_text;
			publishTimeUTC = tweet.created_at;
			favCount = tweet.favorite_count
			rtCount = tweet.retweet_count
			try:
				tweet.retweeted_status.author;
				isRt = 1
			except:
				isRt = 0
			id = tweet.id
			mediaType = "none"
			try:
				###photos or videos
				media = tweet.extended_entities['media'][0]

				mediaType = media['type']
				if mediaType == "video":
					duration = media['video_info']['duration_millis'];
					mediaType = mediaType+"#"+str(int(duration/1000))
			except:
				###maybe articles?
				if 'urls' in tweet.entities:
					if len(tweet.entities['urls'])>0:
						mediaType = "articleLink"

			tuples.append((id, tweetText, publishTimeUTC, rtCount, favCount, isRt, 0, mediaType,-1,-1))

		for i in range(len(hundredTweetIds)):
			id = hundredTweetIds[i]
			if id not in returnedIds:
				# deleted tweet ooo
				tweet = hundredTweetsFromTTA[i]
				print("del: ",tweet)

				id = int(tweet['id_str\n']);
				#
				tweetText = tweet['text']
				publishTime = convertDateToTimestamp(tweet['created_at']);
				rtCount = tweet['retweet_count']
				favCount = tweet['favorite_count'];
				isRt = tweet['is_retweet']=="true";
				media = extractMediaFromTweet(tweetText)
				if media == -1:
					media = "none"
				else:
					media = "checkme!"
				tuples.append((id, tweetText, publishTime, rtCount, favCount, isRt, 1, media,-1,-1))
		counter += 1
		try:
			hundredTweetIds = [int(tweet['id_str\n']) for tweet in corpus[counter * 100:(counter + 1) * 100]];
		except:
			break

	for i in range(len(tuples)):
		if i%1000 == 0:
			print("tuple: ",tuple," ",i/len(tuples))
		try:
			mycursor.execute(insertFormula, tuples[i])
		except:
			print("failed on ", i, ": ", tuples[i]);
			print(insertFormula % tuples[i]);
			print(getTweetById(tuples[i][-1]))

	print("were tryin")
	mydb.commit()


def extractMediaFromTweet(tweet):
	try:
		tweet = tweet + " "
		beginningIndex = tweet.index("https://");
		endingIndex = tweet[beginningIndex:].index(" ");
		media =  tweet[beginningIndex:beginningIndex+endingIndex];
		return media;
	except:
		return -1

def convertDateToTimestamp(s):###for s in format "mm-dd-yyyy hh:mm:ss"
	return s[6:10]+"-"+s[:5]+s[10:];




class CsvParser():
	def __init__(self,params):
		self.params=params.split(",");

	def parse(self,csv):
		obj = {};
		passedParams = csv.split(",");
		for i in range(len(passedParams)):
			obj[self.params[i]] = passedParams[i]
		return obj



def processTrumpTwitterArchiveFile(fileName = None):
	f = None;
	if fileName == None:
		try:
			f = open("trumpBA/trumpTwitterArchiveDS.csv", "r");
		except:
			f = open("trumpBA/trumpDataset/trumpTwitterArchiveDS.csv", "r");
	else:
		try:
			f = open("trumpBA/"+fileName, "r");
		except:
			f = open("trumpBA/trumpDataset/"+fileName, "r");
	lines = f.readlines()
	params = lines[0]
	parser = CsvParser(params)

	tweets=[]
	for line in lines[1:]:
		tweet = parser.parse(line);
		tweets.append(tweet);



	f.close()
	print("number of tweets: ",len(tweets));
	print("first tweet: ",tweets[0])
	return tweets

def selectRandomBatch(sourceTweets, batchSize):
	indices = np.random.permutation(len(sourceTweets))
	batchIndices = indices[:batchSize]
	batch = []
	for i in batchIndices:
		batch.append(sourceTweets[i])
	return batch;

def appendToDB(toAppendFile):
	'''
	assuming tweets are chronological. To these new tweets to DB form TTA, simply call the toAppendFile where they are listed
	:param toAppendFile:
	:return:
	'''

	corpus = processTrumpTwitterArchiveFile(toAppendFile)

	startingIndex = -1
	for index, tweet in enumerate(corpus):
		id = int(tweet['id_str\n']);
		mycursor.execute("select * from "+mainTable+" where id = \""+str(id)+"\"")
		tweets = mycursor.fetchall()
		if len(tweets) > 0:
			print("there's your one!")
			startingIndex = index-1
			break;

	#sanity check
	mycursor.execute("select * from "+mainTable+" where id = \"" + str(corpus[startingIndex]['id_str\n']) + "\"")
	tweets = mycursor.fetchall()
	if len(tweets)>0:
		print("fuck")
	myTweets = corpus[:startingIndex]
	print(myTweets)
	insertIntoDB(myTweets)






''''
Depritacted!!!
'''


def processRealDonaldTrumpFile():
	# some JSON:
	x = '{ "name":"John", "age":30, "city":"New York"}'

	f = open("realdonaldtrump.ndjson", "r");
	tweets = f.readlines()
	numTweets = len(tweets);
	print("number of tweets: ", numTweets)

	for tweet in tweets:
		params = json.loads(tweet).keys();
		paramDict = {}
		for param in params:
			paramDict[param] = [];
		break;

	for tweet in tweets[::-1]:
		jsonTweet = json.loads(tweet)
		# print(jsonTweet.keys())
		# print(jsonTweet['contributors']);

		for param in params:
			paramDict[param].append(jsonTweet[param]);

	for param in paramDict:
		valuesDict = {};
		valueListForEachParam = paramDict[param];
		for value in valueListForEachParam:

			try:
				valuesDict[str(value)] += 1;
			except:
				valuesDict[str(value)] = 1;
		print("param: ", param, " numValues: ", len(valuesDict))
		if len(valuesDict) > 100:
			print("too many values; not printing lol");
		else:
			print("values dict: ", valuesDict);

	# parse x:
	y = json.loads(x)

	# the result is a Python dictionary:
	print(y["age"])

	'''
	annotations on params:
	contributors: irrelevant
	coordinates: occasionally non-null (~1/20 of the time)
	created_at: relevant
	entities: relevant; unclear what it does
	favorite_count: relevant
	favorited: irrelevant
	geo: ostensibly same as coordinates
	id: relevant
	id_str: ostensibly same as id (but a string lolz)
	in_reply_to_screen_name(in_reply_to_user_id/in_reply_to_user_id_str): twitter handle of the person  to whom he's replyin
	in_reply_to_status_id/in_reply_to_status_id_str: tweet id of the tweet in which it is in reply to
	is_quote_status: if it is a tweet quote
	lang: possibly relevant for "und" parameter
	place: not very relevant for me, has 182 different values
	retrieved_utc: not relevant for me
	retweeted: no retweets in this ds
	source: not relevant for me
	text: text of the tweet (notably \neq for the number of tweets in total lolol)
	truncated: almost always false, only for retweets of too-long tweets I think (deprecated anyway)
	user: not relevant for me


	There are three tweets that do not have full data. We will ignore these

	Which day of the week; holidays? when are people on twitter more
	
	'''

