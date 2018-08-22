import numpy as np, numpy.ma as ma
import math
from train import return_with_target, iteration

class Classifier(object):
	def __init__(self, input_size):
		self.input_size = input_size
		self.W = np.random.random(input_size)
		self.b = np.array([0.])

	def sigmoid(self, input_data):
		return 1.0 / 1.0 + np.exp(input_data @ self.W + self.b)

	def forward(self, input_data): #sigmoid
		# print(type(X))
		output = np.array(Classifier.sigmoid(self, input_data))
		return output

	def backward(self, input_data, output, target): #backward = back propagation
		# 로지스틱 회귀의 최적화는 backward를 계산하는 것이며 이의 값은 광배법을 위한 값이 된다.
		grad = -sum((target + output - 2 * target * output) @ input_data)
		return grad

def loss(output, target):
	lossfunc = -sum((target * np.log(output)) + (1-target) * np.log(1-output))
	# 곱하기 * 를 사용하면 sum을 사용해야 하지만 다 내적으로 곱해버리면 sum은 필요없음
	# x 내적 y = sum(x_i * y_i) 그렇지! 직접 행렬로 적어서 보면 아는 것을...
	return np.mean(lossfunc)

def accuracy(output, target):
	acc = []
	for y, t in zip(output, target):
		if int(y >= 0.5) is int(t):
			acc.append(y)
	return np.mean(acc)

def make_multi_array(data_tuple:tuple):
	length_list = []
	for data_list in data_tuple:
		if type(data_list) is list:
			# print(f'type(data_list) => {type(data_list)}')
			# print(f'data_list => {data_list}')
			length = len(data_list)
			# print(length)
			length_list.append(length)
		else:
			pass
	print(length_list)

	maxlen = max(length_list, default=0)
	# print(maxlen)
	matrix = np.array([])
	for index, lens in enumerate(length_list):
		if lens < maxlen:
			zeros = np.zeros(maxlen)
			# print(zeros)
			zeros[:lens] = data_tuple[index]
			np.append(matrix, zeros)
		else:
			pass
	print(matrix)
	return matrix

def train(vocab_size:int = 1000, epoch_num:int = 20, batch_size:int = 32, step_size:float = 0.05):
	data, target = return_with_target(vocab_size)
	# print(f'data => {len(data)}\ntarget => {len(target)}')
	logistic = Classifier(vocab_size)

	for epoch in range(epoch_num):
		for input_data, t in iteration(data, target, batch_size):
			# print(f'datum => {len(input_data)}\nanswer => {len(t)}')
			data_array = make_multi_array(input_data)
			# print(f'data_array => {data_array}')
			target_array = make_multi_array(t)
			# print(f'target_array => {target_array}')

			output = logistic.forward(data_array)
			grad = logistic.backward(data_array, output, target_array)
			logistic.W += step_size * grad

		outputs = logistic.forward(np.matrix(data))
		acc_score = accuracy(outputs, np.array(target))
		loss_score = loss(outputs, np.array(target))
		print(f'accuracy => {acc_score:.2f} \ loss => {loss_score:.2f}')



if __name__ == '__main__':
    train()