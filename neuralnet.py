# 10-301 HW 4
# Jonathan Roman
# 2020/10/14
# feature

import sys
import string
import math
import numpy as np

# takes in path a csv dataset
# returns a numpy array with the data inside
def readData(dataFilePath):
	lines = 0
	tempArr = []
	with open(dataFilePath) as f:
		for line in f:
			newRow = line.strip().split(",")
			tempArr.append(newRow)
	# one np arr for x's and one for labels
	resX = np.zeros((len(tempArr), len(tempArr[0])))
	resY = np.zeros((len(tempArr)))
	for i in range(len(tempArr)):
		resX[i,0] = 1
		resY[i] = float(tempArr[i][0])
		for j in range(1, len(tempArr[i])):
			resX[i,j] = float(tempArr[i][j])
	return (resY, resX)

# creates alpha or beta
# takes in flag for whether or not to initialize as random or all zeros
# and row and col #s
# returns corresponding np array
def creation(flag, rows, cols):
	# random
	if flag == "1":
		# make random np arr and scale so values are in [-0.1,0.1]
		res = (np.random.rand(rows, cols) / 5) - 0.1
		# make bias terms initialized to zero
		zeros = np.zeros((rows, 1))
		res[:,0] = zeros[:,0]
		return res
	# zeros
	elif flag == "2":
		return np.zeros((rows, cols))

# take the dot product of two vectors or matricies
def linearForward(m1, m2):
	return np.dot(m1, m2.transpose()).transpose()

# take the sigmoid of every value in an array or matrix
def sigmoid(v):
	return 1 / (1 + np.exp(-v))

# takes the softmax of a vector
def softmaxForward(v):
	return np.exp(v) / np.sum(np.exp(v))

# returns the entropy: l(y, yhat)
def crossEntropyForward(y, yhat, i):
	return (-1) * math.log(yhat[0,int(y[i])])

# returns the derivative of l wrt to y: dl/dyhat
def crossEntropyBackward(y, yhat, J, gJ):
	res = np.zeros((1, len(yhat[0])))
	res[0,int(y)] = -1 / yhat[0,int(y)]
	return res

# returns the derivative of softmax: dy/db
def softMaxBackward(y, yhat, b, gyhat):
	res = np.zeros((1, len(yhat[0])))
	for i in range(len(yhat[0])):
		res[0,i] = yhat[0,i]
		if i == int(y):
			res[0,i] -= 1
	return res

# returns the derivative of an entire matrix of weights: dl/dalpha or dl/dbeta
def linearBackward(z, b, gb, beta):
	gbeta = np.matmul(gb.transpose(), z)
	gz = np.matmul(gb, beta[:,1:])
	return (gbeta, gz)

# returns the derivative of a sigmoid * gz: dl/da
def sigmoidBackward(a, z, gz):
	return gz * z * (1 - z)

# returns the total entropy for a set of labels and data
def entropy(labels, data, alpha, beta):
	res = 0
	for i in range(len(data)):
		x = data[i:i+1,]
		a = linearForward(alpha, x)
		z = sigmoid(a)
		Z = np.zeros((1, len(z[0]) + 1))
		Z[0,0] = 1
		for j in range(len(z[0])):
			Z[0, j + 1] = z[0, j]
		b = linearForward(beta, Z)
		yhat = softmaxForward(b)
		res += crossEntropyForward(labels, yhat, i)
	return res / len(data)

# performs stochastic gradient descent
# takes in training and validation file paths, M is the # of parameters in each
# subject, D is the # of items in the hidden layer, K is the number of possible
# labels, # of epochs, a flag or whether alpha and beta shoudl be initialized to
# zero or random floats [-0.1,0.1], and learning rate
# returns predicted alpha, beta, and a list containing the train and validation
# entropy after each epoch
def SGD(train, valid, M, D, K, epochs, flag, r):
	entrpy = np.zeros((epochs, 2))
	# data
	trainY, trainX = readData(train)
	validY, validX = readData(valid)

	# initialize alpha and beta with the init flag and dimensions
	alpha = creation(init_flag, D, M + 1)
	beta = creation(init_flag, K, D + 1)

	# loop for each epoch
	for e in range(epochs):
		# loop for each data point
		for i in range(len(trainX)):
			# forwards
			x = trainX[i:i+1,]
			a = linearForward(alpha, x)
			z = sigmoid(a)
			# add the bias term
			Z = np.zeros((1, len(z[0]) + 1))
			Z[0,0] = 1
			for j in range(len(z[0])):
				Z[0, j + 1] = z[0, j]
			b = linearForward(beta, Z)
			yhat = softmaxForward(b)
			J = crossEntropyForward(trainY, yhat, i)

			# backwards NN
			gJ = 1
			gyhat = crossEntropyBackward(trainY[i], yhat, J, gJ)
			gb = softMaxBackward(trainY[i], yhat, b, gyhat)
			gbeta, gz = linearBackward(Z, b, gb, beta)
			ga = sigmoidBackward(a, z, gz)
			galpha, gx = linearBackward(x, a, ga, alpha)
			# update
			alpha -= r * galpha
			beta -= r * gbeta
		# evaluate train mean cross entropy
		entrpy[e, 0] = entropy(trainY, trainX, alpha, beta)
		# evaluate valid mean cross entropy
		entrpy[e, 1] = entropy(validY, validX, alpha, beta)
	return alpha, beta, entrpy

# returns the index of the largest elem in an array
def maxIndex(arr):
	maxVal = -1
	maxI = -1
	for i in range(len(arr[0])):
		if arr[0,i] > maxVal:
			maxVal = arr[0,i]
			maxI = i
	return maxI

# takes in filpath to data, alpha, beta, and output filepath
# writes the predicted labels using alpha and beta to the output filepath
def prediction(data, alpha, beta, filePath):
	Y, X = readData(data)
	labels = np.zeros((len(X), 1))
	for i in range(len(X)):
		x = X[i:i+1,]
		#print("alpha", alpha)
		#print("x", x)
		a = linearForward(alpha, x)
		#print("a", a)
		z = sigmoid(a)
		Z = np.zeros((1, len(z[0]) + 1))
		Z[0,0] = 1
		for j in range(len(z[0])):
			Z[0, j + 1] = z[0, j]
		b = linearForward(beta, Z)
		yhat = softmaxForward(b)
		label = maxIndex(yhat)
		labels[i, 0] = label
	with open(filePath, "w") as f:
		for i in range(len(labels)):
			f.write(str(labels[i, 0]))
			f.write("\n")
	return labels

# gets the percentage of two arrays that are different
def err(y, yhat):
	wrong = 0
	for i in range(len(y)):
		if y[i] != yhat[i, 0]:
			wrong += 1
	return wrong / len(y)

# takes in alpha, beta, filepath to metrics file, train and validation
# filepaths, entropy list for train and validation for each epoch, and the 
# predicted labels for the train and validation data
# writes the entropy at each epoch to the metrics file and then the train 
# and validation error
def metrics(alpha, beta, filePath, train_input, valid_input, entrpy, trainLabels, validLabels):
	trainY, trainData = readData(train_input)
	validY, validData = readData(valid_input)

	trainError = err(trainY, trainLabels)
	validError = err(validY, validLabels)

	with open(filePath, "w") as f:
		for i in range(len(entrpy)):
			# train
			f.write("epoch=")
			f.write(str(i))
			f.write(" crossentropy(train): ")
			f.write(str(entrpy[i, 0]))
			f.write("\n")
			# valid
			f.write("epoch=")
			f.write(str(i))
			f.write(" crossentropy(validation): ")
			f.write(str(entrpy[i, 1]))
			f.write("\n")
		f.write("error(train): %f" % trainError)
		f.write("\n")
		f.write("error(validation): %f" % validError)


if __name__ == "__main__":
	# arguments
	train_input = sys.argv[1]
	valid_input = sys.argv[2]
	train_out = sys.argv[3]
	valid_out = sys.argv[4]
	metrics_out = sys.argv[5]
	num_epoch = sys.argv[6]
	hidden_units = sys.argv[7]
	init_flag = sys.argv[8]
	learning_rate = sys.argv[9]

	# make a numpy arr out of train and validation data
	trainY, trainData = readData(train_input)
	validY, validData = readData(valid_input)

	# size of data params
	M = len(trainData[0]) - 1
	D = int(hidden_units)
	K = 10

	# learning
	alpha, beta, entrpy = SGD(train_input, valid_input, M, D, K, int(num_epoch), init_flag, float(learning_rate))

	# prediction
	trainLabels = prediction(train_input, alpha, beta, train_out)
	validLabels = prediction(valid_input, alpha, beta, valid_out)

	# metrics
	metrics(alpha, beta, metrics_out, train_input, valid_input, entrpy, trainLabels, validLabels)

	print("Done!\n")