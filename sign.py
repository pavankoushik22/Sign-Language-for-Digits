# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x = np.load("data/X.npy")
y = np.load("data/Y.npy")

x = x.reshape(x.shape[0], 64, 64, 1)
trainx, testx, trainy, testy = train_test_split(x,y,test_size = 0.15, random_state = 42)

def create_placeholders(nh0, nw0, nc0, ny):
	x = tf.placeholder('float', shape = (None, nh0, nw0, nc0))
	y = tf.placeholder('float', shape = (None, ny))
	return x,y

def init_weights():
	tf.set_random_seed(2)
	w1 = tf.get_variable('w1',[4,4,1,8], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
	w2 = tf.get_variable('w2',[2,2,8,16], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
	w3 = tf.get_variable('w3',[2,2,16,32], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
	params = {"w1":w1, "w2":w2, "w3":w3}
	return params

def forward_prop(x, params):
	w1 = params["w1"]
	w2 = params["w2"]
	w3 = params["w3"]

	z1 = tf.nn.conv2d(x, w1, [1,1,1,1], padding = "SAME")
	a1 = tf.nn.relu(z1)
	p1 = tf.nn.max_pool(a1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

	z2 = tf.nn.conv2d(p1, w2, [1,1,1,1], padding = 'SAME')
	a2 = tf.nn.relu(z2)
	p2 = tf.nn.max_pool(a2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
	z3 = tf.nn.conv2d(p2, w3, [1,1,1,1], padding = "SAME")
	a3 = tf.nn.relu(z3)
	p3 = tf.nn.max_pool(a3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	fc1 = tf.contrib.layers.flatten(p3)
	fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs = 10, activation_fn = None)

	return fc2

def loss(fc2, y):
	cost = tf.nn.softmax_cross_entropy_with_logits(logits = fc2, labels = y)
	cost = tf.reduce_mean(cost)
	return cost


def model(trainx, trainy, testx, testy, learning_rate = 0.009, num_epochs = 500):
	(m,nh0,nw0,nc0) = trainx.shape
	(m,ny) = trainy.shape
	seed = 2
	x,y = create_placeholders(nh0, nw0, nc0, ny)
	costs = []
	params = init_weights()	
	fc2 = forward_prop(x, params)
	cost = loss(fc2, y)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			_, lcost = sess.run([optimizer, cost], feed_dict = {x:trainx, y:trainy})
			#costs.append(lcost)
			if epoch%5 == 0:
				costs.append(lcost)
				print(epoch, lcost)
		ypred = tf.argmax(fc2, axis = 1)
		yreal = tf.argmax(y, axis = 1)
		correct = tf.equal(ypred, yreal)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		train_acc = accuracy.eval({x:trainx, y:trainy})
		#print(testx.shape, testy.shape)
		test_acc = accuracy.eval({x:testx, y:testy})
		print(train_acc, test_acc)
		return train_acc,test_acc, params
_,_,params = model(trainx, trainy, testx, testy)

