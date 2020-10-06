import numpy as np


def pearsonCorrelationCoefficient(vec1, vec2):

	vec1 = np.asarray(vec1);
	vec2 = np.asarray(vec2);
	# print("lennus: ", vec1.shape, vec2.shape)
	vec1 = vec1-np.mean(vec1)
	vec2 = vec2-np.mean(vec2);

	cov =np.dot(vec1,vec2)/vec1.shape[0];
	std1 = np.std(vec1);
	std2 = np.std(vec2);
	return cov/(std1*std2)

def zScore(favs, fav):
	mean = np.mean(favs);
	std = np.std(favs);
	z = (fav-mean)/std;
	return z;


def ols(x, y):
	x = np.array(x);
	y = np.array(y);
	vec1WithConstant = np.ones((x.shape[0], 2));
	vec1WithConstant[:, 0] = x
	x = vec1WithConstant;
	m,c = np.linalg.lstsq(x, y, rcond=None)[0]
	return m, c;


def reduceDimensPCA(data,n):
	'''
	assuming data \in R^{d,n}
	:param data:
	:param n:
	:return:
	'''
	cov = np.cov(data)
	eigenvalues, eigenvectors = np.linalg.eigh(cov)
	idx = np.argsort(eigenvalues)[::-1]##indices of eigenvalues by size
	eigenvectors = eigenvectors[:,idx]#ith column is the ith eigenvector
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,:n]###only the first n

	reducedData = np.dot(eigenvectors.T,data)
	return reducedData