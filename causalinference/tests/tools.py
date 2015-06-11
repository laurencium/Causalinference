import numpy as np


def random_data_base(N, K, Y=True, D=True, X=True):

	data = []
	if Y:
		Y_data = np.random.rand(N)
		data.append(Y_data)
	if D:
		D_data = np.random.random_integers(0, 1, N)
		data.append(D_data)
	if X:
		X_data = np.random.rand(N, K)
		data.append(X_data)

	if len(data) == 1:
		return data[0]
	else:
		return data

