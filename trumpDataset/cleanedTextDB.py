from .basisFuncs import *
from .part3funcs import normalize_document


	# tokens = doc.split(" ");
	# # filter stopwords out of document
	#
	#
	#
	# filtered_tokens = [token for token in tokens if token not in stop_words]
	# # re-create document from filtered tokens
	# doc = ' '.join(filtered_tokens)
	# return doc






def populateCleanedTextColumnThreadTarget(threadNumber, tweets,tuplesDict):
	tuples = []
	counter = 1;

	for tweet in tweets:

		cleanedText = normalize_document(tweet[1])

		###filter out urls


		tuples.append((cleanedText,tweet[0]))
		# words = [word in wordLemmas if word not in nlp.]
		if counter%100 == 0:
			print("thread: ",threadNumber," ",counter/len(tweets))
		counter += 1


	tuplesDict[threadNumber].extend(tuples)


def populateCleanedTextColumn():
	updateFormula = "UPDATE "+mainTable+" SET cleanedText = %s WHERE id = %s";

	tweets=getTweetsFromDB(n=-1, returnParams=["id","tweetText"])
	# tweets = getTweetsFromDB(n=-1, conditions=["cleanedText is null"], returnParams=["id", "tweetText"])
	print("we've got ",len(tweets)," tweets to update!")
	numThreads = 100;
	tuples = [];
	tuplesDict = {}
	indices = [int(i*len(tweets)/numThreads) for i in range(numThreads)]
	indices.append(len(tweets))
	threads=[]
	for i in range(numThreads):
		tuplesDict[i] = []
		x = threading.Thread(target=populateCleanedTextColumnThreadTarget,args=(i, tweets[indices[i]:indices[i+1]], tuplesDict))
		threads.append(x)
		x.start()
	print("sent em off")

	for index, thread in enumerate(threads):
		thread.join()
		tuples.extend(tuplesDict[index])
	print("all joined",len(tuples))

	mycursor.executemany(updateFormula,tuples)
	mydb.commit()
	print("committed")