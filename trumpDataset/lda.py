import pandas as pd
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
sns.set_style('whitegrid')
from sklearn.decomposition import LatentDirichletAllocation as LDA
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
from .basisFuncs import *;
from .visualization import boxAndWhiskerForKeyWordsFavs;
from .favs import calculateAndStorenGramsWithFavsMinusMeanMedian,loadnGramsWithFavsMinusMeanMedian



def getHotWordsFromLDA(model, count_vectorizer, n_top_words):
	words = count_vectorizer.get_feature_names()
	hotTopics = [];
	for topic_idx, topic in enumerate(model.components_):
		hotTopicsForTopicI = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]];
		hotTopics.append(hotTopicsForTopicI);
		print("\nTopic #%d:" % topic_idx)
		print(" ".join(hotTopicsForTopicI))
	return hotTopics;

def plot_n_most_common_words(count_data, count_vectorizer,n):
	import matplotlib.pyplot as plt
	words = count_vectorizer.get_feature_names()
	total_counts = np.zeros(len(words))
	for t in count_data:
		total_counts += t.toarray()[0]
	count_dict = (zip(words, total_counts))
	count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:n]
	words = [w[0] for w in count_dict]
	counts = [w[1] for w in count_dict]
	x_pos = np.arange(len(words))
	plt.figure(2, figsize=(15, 15 / 1.6180))
	plt.subplot(title=str(n)+' most common words')
	sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
	sns.barplot(x_pos, counts, palette='husl')
	plt.xticks(x_pos, words, rotation=90)
	plt.xlabel('words')
	plt.ylabel('counts')
	plt.show()

def mainTrumpLDA(tweets):


	favCounts = [t[0] for t in tweets];
	cleanedTexts = [t[1] for t in tweets];
	long_string = ','.join(cleanedTexts)

	# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
	print("about to gener8")
	# wordcloud.generate(long_string)
	# wordcloud.to_image()
	# plt.imshow(wordcloud)
	# plt.show();

	count_vectorizer = CountVectorizer(stop_words='english')

	count_data = count_vectorizer.fit_transform(cleanedTexts)
	# plot_n_most_common_words(count_data, count_vectorizer,10)


	number_topics = 2
	number_words = 10

	lda = LDA(n_components=number_topics, n_jobs=-1,doc_topic_prior=0.000000001,max_iter=50)
	lda.fit(count_data)
	ldaHotWordsByTopic = getHotWordsFromLDA(lda, count_vectorizer, number_words)
	allHotWordsForTopics = flattenList(ldaHotWordsByTopic);

	tweetsHash = calculateAndStorenGramsWithFavsMinusMeanMedian([1,2,3,4], tweets, retabulate=True);
	nGramsWithFavsMinusMean, nGramsWithFavsMinusMedian = loadnGramsWithFavsMinusMeanMedian([1,2,3,4], tweetsHash)
	nGramsWithFavsMinusMedianWithTopicsWords = [];
	for i in range(number_topics):
		nGramsWithFavsMinusMedianWithTopicsWords.append([tup for tup in nGramsWithFavsMinusMedian if tup[nGramStorageIndicesSkew["nGram"]] in ldaHotWordsByTopic[i]])


	# nGramsWithFavsMinusMean = nGramsWithFavsMinusMean[-20:]
	# nGramsWithFavsMinusMedian = nGramsWithFavsMinusMedian[-20:]

	# dataFrameDict = {}

	# dataFrameDict["n-grams furthest median from ds median :"] = [nGram[nGramStorageIndicesSkew["nGram"]] + "; " + str(nGram[nGramStorageIndicesSkew["count"]]) for nGram in nGramsWithFavsMinusMedian[:40]]

	# df = pd.DataFrame(dataFrameDict)
	# print(df)
		medianXs = [tup[nGramStorageIndicesSkew["nGram"]].replace(" ", "\n") + "\n" + str(tup[nGramStorageIndicesSkew["count"]]) for tup in nGramsWithFavsMinusMedianWithTopicsWords[i]]
		medianYs = [tup[nGramStorageIndicesSkew["favs"]] for tup in nGramsWithFavsMinusMedianWithTopicsWords[i]];
		boxAndWhiskerForKeyWordsFavs(medianXs, medianYs, np.median(favCounts), title="mediansDifference for hot words identified by LD for topic: " + str(i))


	tweetsLDAVisualizationPath = os.path.join('./tweetsLDAVisualizationPath_'+str(number_topics))

	reload = True;
	if reload:
		LDATweets = sklearn_lda.prepare(lda, count_data, count_vectorizer)
		with open(tweetsLDAVisualizationPath, 'wb') as f:
			pickle.dump(LDATweets, f)

	# load the pre-prepared pyLDAvis data from disk
	else:
		with open(tweetsLDAVisualizationPath,"rb") as f:
			LDATweets = pickle.load(f)


	classes = lda.transform(count_data);
	print(classes);
	classesClone = classes.copy();
	print(classesClone);
	for c in classesClone:
		c.sort();

	nthRankings = [];
	for i in range(len(classesClone[0])):
		nthRankings.append([l[i] for l in classesClone]);
		# print(np.min(nthRankings[i]),np.max(nthRankings[i]),np.mean(nthRankings[i]));
		# print(nthRankings[i])
		# plt.scatter([j for j in range(len(nthRankings[i]))], nthRankings[i], label=str(i), marker='x', linewidth=1, s=10)
		# plt.legend()
		# plt.show();
	# print(nthRankings[0],"\n",nthRankings[-1])
	# plt.clf()
	# for i in range(len(nthRankings)):
	# 	plt.scatter([j for j in range(len(nthRankings[i]))],nthRankings[i],label=str(i),marker='x',linewidth=1,s=10)
	# 	# plt.scatter([i for i in range(len(nthRankings[i]))], [i for i in range], label=str(i), marker='x', linewidth=1, s=10)
	# plt.legend()
	# plt.show();



	maxClasses = [np.argmax(x) for x in classes];


	favCountsByTopic = {i:[] for i in range(number_topics)};
	favCountsByTopic.update({"ambiguous":[]});
	for i in range(len(maxClasses)):
		if classes[i][maxClasses[i]] <= 0.8:
			favCountsByTopic["ambiguous"].append(favCounts[i]);
		else:
			favCountsByTopic[maxClasses[i]].append(favCounts[i]);

	favCountsByTopicList = [];

	topicNames = [];
	for topicNum in favCountsByTopic.keys():
		meanFavCount = np.mean(favCountsByTopic[topicNum]);
		medianFavCount = np.median(favCountsByTopic[topicNum]);
		topicNames.append("Topic " + str(topicNum) + " (count "+str(len(favCountsByTopic[topicNum]))+")\nmean FavCount: "+str(int(round(meanFavCount,0)))+ "\nmedian FavCount:" + str(int(round(medianFavCount,0))));
		favCountsByTopicList.append(favCountsByTopic[topicNum]);



	boxAndWhiskerForKeyWordsFavs(topicNames,favCountsByTopicList,np.mean(favCounts),"LDA With Two Topics And The Corresponding Popularity")
	# plt.show();

	for i in range(len(favCountsByTopicList)):
		favCountsForTopic = favCountsByTopicList[i];
		favCountsForTopic.sort();
		c=randomHexColor()
		plt.scatter([i/len(favCountsForTopic) for i in range(len(favCountsForTopic))],favCountsForTopic,label="Fav Counts for Topic " + str(i),c=c,alpha=0.7);
		plt.plot([0, 1], [np.mean(favCountsForTopic), np.mean(favCountsForTopic)], label="Mean Fav Count for Topic " + str(i),c=c);

	plt.plot([0,1],[np.mean(favCounts),np.mean(favCounts)],label="mean FavCounts");
	plt.plot([0, 1], [np.median(favCounts), np.median(favCounts)], label="median FavCounts");
	plt.legend(fontsize=20);
	plt.yticks(fontsize=20)
	plt.title("Fav Count Distribution for Topics (Unsupervised LDA)")
	plt.show();



	pyLDAvis.save_html(LDATweets, './tweetsLDAVisualizationPath_' + str(number_topics) + '.html')


'''
Topics TODO:
• DEPR implement Gibbs-sampling algorithm as described by Zheng and Han 
• DEPR implement GSDMM as described here: https://towardsdatascience.com/short-text-topic-modeling-70e50a57c883 (and here: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf)
• cross-validation for correlation with tweet popularity
In General TODO:
• interpret results. It's not as interesting to just say "oh look I can separate tweets," I want to have some sot of analysis for why the parameters are working,
like maybe some kind of classification of words that get high weights in regression, how they are connected, etc.

Results: LDA separates tweet popularity on 2-class ~0 alpha for 0.1, 0.2 ends of tweets. Wild!
TODO: we have established that Wortwahl changes over time - apply DTM in order to see how words in each topic change over time

...
word embeddings for tweets? 
=> nearest neighbours classifier with word embeddings?
K-Means vs LDA ooo this will be interesting :)


Korrektur:
box-plots words from tweets corresponding only to proper topic 
Perhaps with mean - it's less stable/more verzerrt
'''