
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
from colorama import Fore, Style
import numpy as np

from trumpBA.trumpDataset.twitterApiConnection import *
relevancyDateThreshhold = "where publishTime > \"2016:11:01 00:00:00\""


mydb = mysql.connector.connect(host="localhost",user="root",passwd="felixMySQL",database="trump");
mycursor = mydb.cursor(buffered=True)

mainTable = "tta2"
purePresConditions=["president","isRt = 0","deleted = 0"]

def getTweetById(id: int):
	return getTweetsById([id])

def getTweetsById(ids: [int]):
	offendingTweets = api.statuses_lookup(ids, tweet_mode= 'extended');
	return offendingTweets;

def computeBigrams(tweetWords):
	return [tweetWords[i] + " " + tweetWords[i + 1] for i in range(len(tweetWords) - 1)]


def tweetHash(tweets):
	tweets.sort(key= lambda tup: tup[1]+str(tup[0]), reverse=True)
	myString = str(tweets);
	return hasher(myString.encode()).hexdigest()


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
		tweets = getTweetsFromDB(conditions="purePres", orderBy="favCount asc",
								 returnParams=["favCount", "cleanedText"])

	return getTopiPercentOfList(i,tweets,True),getTopiPercentOfList(i,tweets,False)


def getTopBottomiTweetsOfPurePresTweets(i, tweets=None):
	if tweets == None:
		tweets = getTweetsFromDB(conditions="purePres", orderBy="favCount asc",
								 returnParams=["favCount", "cleanedText"])
	return getTopiElementsOfList(i,tweets,True),getTopiElementsOfList(i,tweets,False)


def getApprovalRatings():
	mycursor.execute(
		"select date, approveEstimate from approvalRatings order by date asc");
	ratings = mycursor.fetchall()
	return ratings;

def getFavsByKeyword(keyword,rts = True):
	conditions = ["cleanedText like \"%" + keyword + "%\"","president","deleted = 0"]
	if not rts:
		conditions.append("isRt = 0")
	tweets = getTweetsFromDB(n=-1, conditions=conditions, returnParams=[ "favCount"]);
	return tweets




def getRootUrl(url):
	try:
		session = requests.Session()  # so connections are recycled
		resp = session.head(url, allow_redirects=True)
		return resp.url;
	except:
		return "checkme!"


def camlify(string, firstLetterCapital = True):
	string = string.rstrip().lstrip()


	return string.replace(" ","_")

	toks = string.split(" ")
	ret = toks[0].lower()
	for s in toks[1:]:
		ret += s[0].upper()+s[1:].lower()
	return ret;

def getTweetsFromDB(n=-1, conditions=[], returnParams="*",orderBy = "publishTime desc"):#president = False, allParams = False,inclRts = False):
	'''
	returns all columns of latest n tweets of DJT
	:param n: -1 will return all tweets
	:param president: True will only return tweets sent while president
	:return: if allParams: [(tweetText, publishTime, rtCount, favCount, isRt, deleted, id),...];
			else: [tweetText,...]
	'''
	if conditions == "purePres":
		conditions = purePresConditions
	if len(conditions)>0:
		whereString = "where "
		for index, condition in enumerate(conditions):
			if condition == "president":
				whereString += ("publishTime > \"2017-02-20 00:00:00\"")
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
	query = "select "+returnParams+" from "+mainTable+" "+whereString + "  order by "+orderBy+" "+limiter;

	mycursor.execute(query)
	tweets = mycursor.fetchall()

	return tweets





def insertIntoDB(corpus):
	# corpus = corpus[:101]

	# mycursor.execute("delete from trumptwitterarchive");
	ind = 0;
	tuples = []
	insertFormula = "INSERT INTO "+mainTable+" (id, tweetText, publishTime, rtCount, favCount, isRt, deleted, mediaType) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
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

			tuples.append((id, tweetText, publishTimeUTC, rtCount, favCount, isRt, 0, mediaType))

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
				tuples.append((id, tweetText, publishTime, rtCount, favCount, isRt, 1, media))
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
			print("failed on ", i, ": ", tuples[i][-1]);
			print(insertFormula % tuples[i]);
			print(getTweetById(tuples[i][-1]))

	print("were tryin")
	mydb.commit()


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
			f = open("trumpTwitterArchiveDS.csv", "r");
		except:
			f = open("trumpDataset/trumpTwitterArchiveDS.csv", "r");
	else:
		try:
			f = open(fileName, "r");
		except:
			f = open("trumpDataset/"+fileName, "r");
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