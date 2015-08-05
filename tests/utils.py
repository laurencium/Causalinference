import numpy as np


def random_data(N=0, K=0, Y_cur=None, D_cur=None, X_cur=None):

	if X_cur is not None:
		N, K = X_cur.shape
	elif D_cur is not None:
		N = D_cur.shape[0]
	elif Y_cur is not None:
		N = Y_cur.shape[0]

	if N == 0 and K == 0:
		K = np.random.random_integers(1, 5)
		N = np.random.random_integers(4, 4*K)
	elif N != 0 and K == 0:
		K = np.random.random_integers(1, N-1)
	elif N == 0 and K != 0:
		N = np.random.random_integers(4, 4*K)

	data = []
	if Y_cur is None:
		Y_data = np.random.rand(N)
		data.append(Y_data)
	if D_cur is None:
		D_data = np.random.random_integers(0, 1, N)
		# loop to ensure at least two subjects in each group
		while D_data.sum() <= 1 or D_data.sum() >= N-1:
			D_data = np.random.random_integers(0, 1, N)
		data.append(D_data)
	if X_cur is None:
		X_data = np.random.rand(N, K)
		data.append(X_data)

	if len(data) == 1:
		return data[0]
	else:
		return data

