from .basisFuncs import *;

def gibbs(docs,numClusters):
	m = np.zeros((numClusters,1));
	n = np.zeros((numClusters, 1));
	nw = np.zeros((numClusters, 1));



