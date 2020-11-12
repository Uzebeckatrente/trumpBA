import datetime
import numpy as np
import matplotlib.pyplot as plt

from .stats import pearsonCorrelationCoefficient
from .basisFuncs import *

def getTimeObjectOfLastSecondOfGivenDayForTimestamp(timestamp):
	return timestamp - datetime.timedelta(hours=timestamp.hour, minutes=timestamp.minute,seconds=timestamp.second) + datetime.timedelta(days=1) - datetime.timedelta(seconds=1);





def graphTwoDataSetsTogether(dataset1, label1, dataset2, label2, scatter = False, xes = None,title=None):
	print("scatty d: ",scatter)
	minLen = min(len(dataset1),len(dataset2))
	print("minny: ",minLen)

	fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	fig.subplots_adjust(left=0.06, right=0.94)

	host = fig.add_subplot(111)

	par1 = host.twinx()

	# host.set_xlabel("months")
	host.set_ylabel(label1)
	par1.set_ylabel(label2)

	color1 = plt.cm.viridis(0)
	color2 = plt.cm.viridis(0.5)
	# color3 = plt.cm.viridis(1.0)

	try:
		xes = xes[:minLen]
	except:
		xes = [i for i in range(minLen)]

	dataset1 = dataset1[:minLen]
	dataset2 = dataset2[:minLen]

	if not scatter:
		pDS1, = host.plot(xes, dataset1, color=color1, label=label1)
	else:
		pDS1, = host.scatter(xes, dataset1, color=color1, label=label1)
	host.set_ylim(min(dataset1), max(dataset1))
	par1.set_ylim(min(dataset2), max(dataset2))
	# host.set_ylim(0,10000)
	# par1.set_ylim(0, 10000)

	if not scatter:
		pDS2, = par1.plot(xes, dataset2, color=color2, label=label2)
	else:
		pDS2, = par1.scatter(xes, dataset2, color=color2, label=label2)

	lns = [pDS1, pDS2]
	host.legend(handles=lns, loc='best')
	pearson = pearsonCorrelationCoefficient(dataset1, dataset2);
	if title == None:
		plt.title("pearson correlation coefficient: "+str(round(pearson,2)))
	else:
		plt.title(title);

	par1.yaxis.label.set_color(pDS2.get_color())
	plt.show()





def computeXandYForPlotting(tweets, daysPerMonth):
	'''

	:param tweets: *
	:param daysPerMonth:
	:return:
	'''
	avgFavCounts = [];
	avgRtCounts = []

	numberOfTweetsPerMonth = [];
	months = []

	def updateFavAndRtCounts(totalFavCountThisMonth, totalRtCountThisMonth, tweetsThisMonth):

		if tweetsThisMonth > 0:
			avgFavCounts.append(totalFavCountThisMonth / tweetsThisMonth);
			avgRtCounts.append(totalRtCountThisMonth / tweetsThisMonth);
		else:
			avgFavCounts.append(0);
			avgRtCounts.append(0);
		numberOfTweetsPerMonth.append(tweetsThisMonth);

	tweetIndex = 0;
	lastTweetTimeObj = getTimeObjectOfLastSecondOfGivenDayForTimestamp(tweets[-1][1]);
	tweetIndicesByMonth = {};
	currentMonth = getTimeObjectOfLastSecondOfGivenDayForTimestamp(tweets[0][1]);


	while currentMonth<= lastTweetTimeObj:
		totalFavCountThisMonth = 0
		totalRtCountThisMonth = 0
		tweetsThisMonth = 0;
		for i in range(tweetIndex,len(tweets)):
			tweet = tweets[i]
			if tweets[i][1]>currentMonth+ datetime.timedelta(days=daysPerMonth):
				nextTweetIndex = i;
				break;
			else:
				totalFavCountThisMonth+=tweet[3]
				totalRtCountThisMonth += tweet[2]
				tweetsThisMonth += 1;
		# print("a month has passed : ",tweetsThisMonth);

		updateFavAndRtCounts(totalFavCountThisMonth,totalRtCountThisMonth,tweetsThisMonth);
		tweetIndicesByMonth[currentMonth] = (tweetIndex,nextTweetIndex);
		months.append(currentMonth)
		currentMonth =currentMonth+ datetime.timedelta(days=daysPerMonth);
		tweetIndex = nextTweetIndex;
	tweetIndicesByMonth[currentMonth-datetime.timedelta(days=daysPerMonth)] = tweetIndicesByMonth[currentMonth-datetime.timedelta(days=daysPerMonth)][0],len(tweets)

	fifteenIndices = [int(len(months) * i / 15) for i in range(15)]
	fifteenIndices.append(len(months) - 1)
	# lens = (len(months), fifteenIndices);
	monthsCorrespondingToFifteenIndices = [str(months[index])[:10] for index in fifteenIndices];
	return avgFavCounts, avgRtCounts, months, fifteenIndices, monthsCorrespondingToFifteenIndices, tweetIndicesByMonth

def histogramFavCountsPoisson(favCounts):
	favCounts.sort(reverse=False)
	favCounts = np.array([int(fvc/1000)for fvc in favCounts])
	maxFavCount = favCounts[-1]
	colors = ['b', 'r', 'black']

	# fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	# fig.subplots_adjust(left=0.06, right=0.94)

	bins = list(range(min(favCounts),max(favCounts),2));

	N, bins, patches = plt.hist(favCounts,bins=bins)
	for i in range(len(patches)):
		patches[i].set_facecolor(colors[i % 3])

	mean = np.mean(favCounts);
	variance = np.var(favCounts);
	print("mean: ",mean, " variance: ",variance);
	print(favCounts)
	plt.title("Poisson-like Distribution of purePres Tweets Favourite Counts",fontsize=25)

	plt.xticks([10]+[i*100 for i in range(1,6)],[str(i*100)+"k" for i in range(6)],fontsize=20)
	plt.ylabel("Density",fontsize=20)
	plt.yticks([i*50 for i in range(5)],[str(round(i*50/len(favCounts),3)) for i in range(5)],fontsize=20)
	plt.xlabel("Favourite Counts",fontsize=20)
	plt.xlim(0,500)
	#
	plt.show()

def histogramFavCounts(favCounts):
	favCounts.sort(reverse=False)
	favCounts = np.array([int(round(np.log(fvc),2)*100) if fvc >0 else 0 for fvc in favCounts])
	maxFavCount = favCounts[-1]
	colors = ['b', 'r', 'black']

	# fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	# fig.subplots_adjust(left=0.06, right=0.94)

	bins = list(range(min(favCounts),max(favCounts),2));

	N, bins, patches = plt.hist(favCounts,bins=bins)
	for i in range(len(patches)):
		patches[i].set_facecolor(colors[i % 3])

	mean = np.mean(favCounts);
	variance = np.var(favCounts);
	print("mean: ",mean, " variance: ",variance);
	print(favCounts)
	plt.title("Normal-like Distribution of Logarithmized purePres Tweets Favourite Counts\n",fontsize=25)
	xes = [1010,1100,1180,1250,1320]
	plt.xticks(xes,[str(int(np.exp(x/100)/1000))+"k" for x in xes],fontsize=20)
	plt.ylabel("Density",fontsize=20)
	plt.yticks([i*50 for i in range(5)],[str(round(i*50/len(favCounts),3)) for i in range(5)],fontsize=20)
	plt.xlabel("Favourite Counts",fontsize=20)
	plt.xlim(1000,1320)
	#
	plt.show()




def boxAndWhiskerForKeyWordsFavs(keywords, favsForTweetsWithkeyword, avgFav,title,yLabel = None, yMax = -1,medianprops = None):


	data = favsForTweetsWithkeyword
	# y = data.mean(axis=0)

	x = [i*3-3 for i in range(len(keywords))]
	favX = [i*3 for i in range(-1,len(keywords)-1)]
	favX[0] += -1
	favX[-1] += 1

	# Plot a line between the means of each dataset
	# plt.plot(x, y, 'b-')

	# Save the default tick positions, so we can reset them...

	plt.plot(favX,[avgFav]*len(favX),linestyle='--',label="Average for whole Dataset",linewidth=4)
	plt.boxplot(data, positions=x, notch=False,showfliers=True,medianprops=medianprops);

	plt.title(title,fontsize=20)
	plt.legend(fontsize=15)
	plt.yticks(fontsize=15)
	if yMax != -1:
		plt.ylim(0,yMax);
	if yLabel != None:
		plt.ylabel(yLabel,fontsize=15)

	plt.xticks(x,keywords,fontsize=14)
	plt.show()