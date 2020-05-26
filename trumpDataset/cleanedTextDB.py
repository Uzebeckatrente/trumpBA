from trumpDataset.basisFuncs import *
import emoji


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
	doc = doc.lower()
	doc = doc.strip()
	doc = doc.replace('\n', ' ')


	rtIndex = doc.find("rt @");

	if doc[-1:] == "…":

		lastSpaceIndex = len(doc)-doc[::-1].index(" ")-1
		doc = doc[:lastSpaceIndex]

	###filter out rts
	if rtIndex != -1:
		firstAtIndex = doc.find(":")
		doc = doc[2 + firstAtIndex:];

	doc = emoji.get_emoji_regexp().sub("xemojix ", doc)

	doc = re.sub("&amp;", "and", doc)
	doc = re.sub("w/", "with", doc)
	doc = re.sub("(https://.* )|(https://[\S]*)", "", doc)
	doc = re.sub('[/\-]', ' ', doc)
	doc = re.sub(r'[^a-zA-Z0-9\s\@\#]', '', doc, re.I|re.A)
	doc=re.sub("\.", "", doc)


	nlpTweetText = nlp(doc)

	wordLemmas = []
	for token in nlpTweetText:
		if unworthynessOfToken(token):
			continue;

		elif token.lemma_ in ['pm','am']:
			wordLemmas.append(token.lemma_[0]+"."+token.lemma_[1]+".")
		elif token.string.strip()[0] == "@":
			wordLemmas.append(token.string.strip())
		else:
			wordLemmas.append(token.lemma_)


	return " ".join(wordLemmas);
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

		tweetText = normalize_document(tweet[0])

		###filter out urls




		tuples.append((tweetText,tweet[-3]))
		# words = [word in wordLemmas if word not in nlp.]
		if counter%100 == 0:
			print("thread: ",threadNumber," ",counter/len(tweets))
		counter += 1


	tuplesDict[threadNumber].extend(tuples)


def populateCleanedTextColumn():
	updateFormula = "UPDATE "+mainTable+" SET cleanedText = %s WHERE id = %s";

	tweets=getTweetsFromDB(n=-1, conditions=[], returnParams="*")
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
	print("all joined")

	mycursor.executemany(updateFormula,tuples)
	mydb.commit()