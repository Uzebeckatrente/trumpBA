
from .basisFuncs import *;
from .lda import *;
from .vocabChangeOverTime import compareVocabOverTime;

def tweetLengths(tweets):




	tweetLengths = [len(t[2].split(" ")) for t in tweets];
	tweetLengths.sort();

	tweetLengthsClean = [len(t[1].split(" ")) for t in tweets];
	tweetLengthsClean.sort();
	# plt.scatter([i for i in range(len(tweetLengths))],tweetLengths);
	# plt.show();

	plt.hist(tweetLengths,bins=[i for i in range(max(tweetLengths))]);
	plt.title("Length of Tweets")
	plt.show();

	plt.hist(tweetLengthsClean, bins=[i for i in range(max(tweetLengthsClean))]);
	plt.title("Length of Preprocessed Tweets")
	plt.show();

def wordCloud(cleanedTexts):
	long_string = ','.join(cleanedTexts)
	wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
	print("about to gener8")
	wordcloud.generate(long_string)
	wordcloud.to_image()
	plt.imshow(wordcloud)
	plt.show();

def wordDistribution(cleanedTexts):
	pass;




def dataVisMain():
	allTweets = getTweetsFromDB(purePres=True, returnParams=["favCount", "cleanedText", "tweetText", "publishTime"],conditions=["isReply = 0"]);
	allTweets.sort(key=lambda t: t[0]);
	cleanedTexts = [t[1] for t in allTweets]
	tweetLengths(allTweets)
	# wordCloud(cleanedTexts)
	# compareVocabOverTime(epochCount=5);
	bottomTenthAndTopTenth = allTweets[:int(len(allTweets) * 0.06)] + allTweets[int(len(allTweets) * (1 - 0.06)):]
	mainTrumpLDA(bottomTenthAndTopTenth)

	#todo: violin plots