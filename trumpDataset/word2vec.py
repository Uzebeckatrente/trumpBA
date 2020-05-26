import numpy as np
from sklearn.manifold import TSNE
from sklearn import preprocessing

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
corpus_raw = 'Nose is purple. he is a purple guy. nose is seriously gregarious'

corpus_raw = corpus_raw.lower()

words = [];
for word in corpus_raw.split(" "):
	if word != "." and len(word)>0:
		words.append(word);

words = set(words)


word2int = {}
int2word = {}
vocabSize = len(words)
for i,word in enumerate(words):
	word2int[word] = i
	int2word[i] = word

print(word2int['queen'])
print(int2word[word2int[int2word[word2int['queen']]]]) #lol

raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
	sentences.append(sentence.split())


data = []
himData = []
WINDOW_SIZE = 2
senLens = [len(s) for s in sentences];


for s in sentences:
	print("s",s)
	for middleWordIndex in range(len(s)):
		for i in range(max(0,middleWordIndex-WINDOW_SIZE),middleWordIndex):
			data.append((s[middleWordIndex],s[i]))
			data.append(( s[i],s[middleWordIndex]))

def toOneHot(index, vecSize):
	vec = np.zeros(vecSize);
	vec[index] = 1;
	return vec;


xTrain = [];
yTrain = [];
for pair in data:
	xTrain.append(toOneHot(word2int[pair[0]],vocabSize))
	yTrain.append(toOneHot(word2int[pair[1]],vocabSize));

xTrain = np.asarray(xTrain);
yTrain = np.asarray(yTrain);
print(xTrain)


x = tf.placeholder(tf.float32, shape=(None, vocabSize))
y_label = tf.placeholder(tf.float32, shape=(None, vocabSize))
embeddingDim = 5;


W = tf.Variable(tf.random_normal([vocabSize, embeddingDim]))
b = tf.Variable(tf.random_normal([embeddingDim]))
WPrime = tf.Variable(tf.random_normal([embeddingDim, vocabSize]))
bPrime = tf.Variable(tf.random_normal([vocabSize]))

hidden_representation = tf.add(tf.matmul(x,W), b)
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, WPrime), bPrime))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10000
for i in range(n_iters):
	sess.run(train_step, feed_dict={x: xTrain, y_label: yTrain})
	if i%200 == 0:
		print('training loss is : ', sess.run(cross_entropy_loss, feed_dict={x: xTrain, y_label: yTrain}))



print("\n\n")

print(sess.run(W))
print('----------')
print(sess.run(b))
print('----------')

vectors = sess.run(W + b)
print(vectors[ word2int['queen'] ])

def euclidean_dist(vec1, vec2):
	return np.sqrt(np.sum((vec1-vec2)**2))

def findClosest(vecIndex):
	vec=vectors[vecIndex]
	closestIndex = -1;

	closestDist = np.inf;
	for i in range(len(vectors)):
		otherVec = vectors[i]
		if i != vecIndex:
			dist = euclidean_dist(vec,otherVec);
			if dist < closestDist:
				closestIndex = i;
				closestDist = dist


	return closestIndex;

for i in range(len(vectors)):
	print("closest vector to : ",int2word[i], " is : ",int2word[findClosest(i)])





model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
tsneVectors = model.fit_transform(vectors)


normalizer = preprocessing.Normalizer()
tsneVectors =  normalizer.fit_transform(tsneVectors, 'l2')


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlim(min([vectors[word2int[w]][0] for w in words])-1, max([vectors[word2int[w]][0] for w in words])+1)
ax.set_ylim(min([vectors[word2int[w]][1] for w in words])-1, max([vectors[word2int[w]][1] for w in words])+1)
for word in words:
    print(word, tsneVectors[word2int[word]][1])
    ax.annotate(word, (tsneVectors[word2int[word]][0],tsneVectors[word2int[word]][1] ))
plt.show()