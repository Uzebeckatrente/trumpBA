import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import spacy
nlp = spacy.load("en")
from trumpDataset.basisFuncs import *

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
wpt = nltk.WordPunctTokenizer() ###
stop_words = nltk.corpus.stopwords.words('english')



###makes sense for tweets
stop_words.append("rt")
###

pd.options.display.max_colwidth = 200


def getMostPopularWordsOverCountNMostPopularBigramsOverCount(n,m):
	'''
	Depricated! replace plz
	:param n:
	:param m:
	:return:
	'''


	wordsInDescendingFrequencyFile = open("trumpDataset/npStores/wordsInDescendingFrequency.p", "rb")
	wordsInDescendingFrequency = pickle.load(wordsInDescendingFrequencyFile);

	topNWords = [w for w in wordsInDescendingFrequency if w[1] >= n]

	if m != -1:
		bigramsInDescendingFrequencyFile = open("trumpDataset/npStores/bigramsInDescendingFrequency.p", "rb")
		bigramsInDescendingFrequency = pickle.load(bigramsInDescendingFrequencyFile);
		topNBigrams = [bg for bg in bigramsInDescendingFrequency if bg[1] >= m]

		return topNWords,topNBigrams
	return topNWords,[]


def computeMostCommonnGrams(tweets, n):
	'''
	cleanedText must be at position 1!
	selects all presedential cleantexts, computes word count/bigram matrices;
	counts the frequency of each word and bigram, sorts by descending frequency, dumps
	:return:
	'''

	corpus = []
	ids = []
	for tweet in tweets:
		id = tweet[0]
		cleanedText = tweet[1]
		corpus.append(cleanedText)
		ids.append(id);

	nGramsMatrix, nGramsVocab = computenGramsMatrix(corpus,n)


	sumsnGramCount = list(np.sum(nGramsMatrix, axis=0))
	vocabAndnGramCounts = list(zip(nGramsVocab, sumsnGramCount))

	vocabAndnGramCounts.sort(key=lambda tup: tup[1], reverse=True)

	#2428433025033787798
	tweetsHash =tweetHash(tweets)

	with open('trumpDataset/npStores/'+str(n)+'GramsInDescendingFrequency'+tweetsHash+'.p', 'wb') as fp:
		print("writing file: ",'trumpDataset/npStores/'+str(n)+'GramsInDescendingFrequency'+tweetsHash+'.p')
		pickle.dump(vocabAndnGramCounts, fp)


def getMMostLeastPopularnGramsWithCounts(m, n,tweetHashes, percentage = False):
	'''

	:param m:
	:param n:
	:param percentage: compute n-grams in top m percent rather than the m highest
	:return:
	'''

	if len(tweetHashes) != 2:
		print("tweetHashes must be [hashTop, hashBottom]")
		exit()

	try:
		filePath = "trumpDataset/npStores/"+str(n)+"GramsInDescendingFrequency"+tweetHashes[0]+".p"
		nGramsInDescendingFrequencyFileTop = open(filePath, "rb");

	except:
		print(filePath," doesn't exist! exiting")
		exit(1)
	try:
		filePath = "trumpDataset/npStores/"+str(n)+"GramsInDescendingFrequency"+tweetHashes[1]+".p";
		nGramsInDescendingFrequencyFileBottom = open(filePath, "rb");
	except:
		print(filePath+" doesn't exist! exiting")
		exit(1)


	nGramsInDescendingFrequencyBottom = pickle.load(nGramsInDescendingFrequencyFileBottom);
	nGramsInDescendingFrequencyTop = pickle.load(nGramsInDescendingFrequencyFileTop);
	if percentage:
		topM = getTopiPercentOfList(m, nGramsInDescendingFrequencyTop,False);
		bottomM = getTopiPercentOfList(m, nGramsInDescendingFrequencyBottom, False);
	else:
		topM = getTopiElementsOfList(m, nGramsInDescendingFrequencyTop,False);
		bottomM = getTopiElementsOfList(m, nGramsInDescendingFrequencyBottom, False);

	return topM,bottomM



def numberButNotYear(token):
	return token.like_num and not (str(token).isdigit() and 1800 < int(str(token)) < 2100)

def unworthynessOfToken(token):


	return token.is_stop or token.is_punct or token.is_space or numberButNotYear(token) or len(token.string)<2;


#function for making pre-processing/verschoening each document


normalize_corpus = np.vectorize(normalize_document) ##to apply on multiple elements in a vector



# normCorpus = normalize_corpus(corpus)

# print(normCorpus)



def computeWordCountsMatrix(normCorpus):
	#returns wordCountsMatrix,vocabs


	cv = CountVectorizer(min_df=0., max_df=1.,token_pattern = '[^\s]+') ##include words that have a frequency between 0 and 1 (so all words)
	wordCountsMatrix = cv.fit_transform(normCorpus)
	wordCountsMatrix = wordCountsMatrix.toarray() ###we implemented this less efficiently in processTweetData.py
	vocab = cv.get_feature_names();
	return wordCountsMatrix,vocab

def visualizeWordCountsMatrix(wordCountsMatrix,vocab):
	wordCountsMatrixPandas = pd.DataFrame(wordCountsMatrix, columns=[vocab])
	print("wordCountsMatrix:\n",wordCountsMatrixPandas);

def computeBigramsMatrix(normCorpus):
	# returns wordCountsMatrix,vocabs
	bv = CountVectorizer(ngram_range=(2,2),token_pattern = '[^\s]+') ##for 2-grams
	bigramsMatrix = bv.fit_transform(normCorpus)
	bigramsMatrix = bigramsMatrix.toarray()

def computenGramsMatrix(normCorpus,n):
	# returns wordCountsMatrix,vocabs
	bv = CountVectorizer(ngram_range=(n, n), token_pattern='[^\s]+')  ##for 2-grams
	bigramsMatrix = bv.fit_transform(normCorpus)
	bigramsMatrix = bigramsMatrix.toarray()

	vocab = bv.get_feature_names()
	return bigramsMatrix,vocab

def visualizeBigramsMatrix(bigramsMatrix,vocab):
	bigramsMatrixPandas = pd.DataFrame(bigramsMatrix, columns=[vocab])
	print("bigramsMatrix:\n",bigramsMatrixPandas);


def computeTFIDFMatrix(normCorpus):
	##tf-idf is the frequency of words in Documents scaled by document frequency
	##which is to say for very common words it will get scaled down; not so aussagekraeftig
	tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True,token_pattern="[a-zA-Z0-9\@\#]+") ###same deal except with the tf-idf metric instead of just word/n-gram frequency
	tfidfMatrix = tv.fit_transform(normCorpus)
	tfidfMatrix = tfidfMatrix.toarray()

	vocab = tv.get_feature_names()
	return tfidfMatrix,vocab
def visualizeTFIDFMatrix(tfidfMatrix,vocab):


	tfidfMatrixPandas = pd.DataFrame(np.round(tfidfMatrix, 2), columns=vocab)
	print("tfidfMatrix:\n",tfidfMatrixPandas);
def computeSimilarityMatrix(tfidfMatrix):
	similarityMatrix = cosine_similarity(tfidfMatrix) ##shows how similar the tfidf's
	# of each document are based on the cosine distance formula. Could be relevant for Trump tweet differences!
	return similarityMatrix

def visualizeSimilarityMatrix(similarityMatrix,toShow = False):
	similarityMatrixPandas = pd.DataFrame(similarityMatrix)
	print("similarityMatrix:\n",similarityMatrixPandas);
	Z = linkage(similarityMatrix, 'ward')
	zPandas = pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2', 'Distance', 'Cluster Size'],
						   dtype='object')
	print(zPandas);
	# a visualization of Z
	plt.figure(figsize=(8, 3))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('Data point')
	plt.ylabel('Distance')
	dendrogram(Z)
	plt.axhline(y=1.0, c='k', ls='--', lw=0.5)
	if toShow:
		plt.show()
	return Z;
'''
the goal is to find the documents that are hierarchichly most similar based on the cosine metric
Z finds the two clusters that are the nearest, then merges them
'''




#
# maxDistBetweenClusters = 1.5 ##max distance between clusters to belong togezza
# clusterLabelsByDistance = fcluster(Z, maxDistBetweenClusters, criterion='distance')
# clusterLabelsByDistance = pd.DataFrame(clusterLabelsByDistance, columns=['ClusterLabel'])
# # extendedCorpusPandas = pd.concat([corpusPandas, clusterLabelsByDistance], axis=1)


'''
LDA (lol) is for assigning topics to words. Based on the principle that documents
are seen as bags of words that are created by sampling for topics of that document
and then sampling assigned words to that topic. LDA works backwards, looking
at each document d iteratively and, for each w \in d, for each topic t, says:
"what proportion of words in this document are assigned to topic t?" 
AND says
"what proportion of words assigned to topic t are examplare of w?" then it multiplies
these probabilities togezza and re-assigns topic to w
'''

def computeLDA(numTopics,wordCountsMatrix, vocab):
	lda = LatentDirichletAllocation(n_components=numTopics, max_iter=10000, random_state=1)
	###interestingly doesn't always converge to this result; try with random_state = 0(!)

	##we know there are four topics
	print("fitting LDA")
	ldaMatrix = lda.fit_transform(wordCountsMatrix) ###runs lda to fit topics
	print("done fitting LDA :)")
	features = pd.DataFrame(ldaMatrix, columns=['T'+str(i) for i in range(numTopics)]);
	print(features)




	###words and persuasion to each topic by topic
	wordTopicDistribution = lda.components_
	topicIndex = 0;
	for topic_weights in wordTopicDistribution: #for each topic
		topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
		topic = sorted(topic, key=lambda x: -x[1])
		topic = [item for item in topic if item[1] > 0.6]
		print("topic: ",topicIndex);
		for t in topic:
			print(t);
		topicIndex += 1;
		for i in range(10):
			print()
		print("newtopic!!!!")
	return ldaMatrix



'''
How could this relate to thesis:
• Could do LDA   on tweets to determine thematic grouping
• Could use TF-IDF to find spiky words, maybe they're trump words?
• Can compare tweet vectors to determine closeness

Next steps:
• continue learning about word embeddings - do tutorial, read glove, vec papers
• do some shakespeare experimenting, look at how to obtain Twitter data

Another idea: predict which trends originated from within Twitter, which externally... 
Trump would be mostly inside twitter (Hamberders, Covfefe, etc.), maybe amplify outside
trends; but outside trends (like TV shows, Soccer, corona, etc.) don't have a kick-off


Market fluctuation by trump tweet? mmmmmm
tweet
'''








plt.show();