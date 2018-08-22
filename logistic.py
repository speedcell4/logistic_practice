import numpy as np
from train import return_with_target, iteration

class Classifier(object):
	def __init__(self, input_size):
		self.input_size = input_size
		self.W = np.random.random(input_size)
		self.b = np.array([0.])

	def forward(self, X): #sigmoid
		output = np.array([1.0 / 1.0 + (X @ self.W + self.b)])
		return output

	def backward(self, X, output, target): #backward = back propagation
		# 로지스틱 회귀의 최적화는 backward를 계산하는 것이며 이의 값은 광배법을 위한 값이 된다.
		pd = -sum((target + output - 2 * target * output) @ X)
		return pd

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

def train(vocab_size:int = 1000, epoch_num:int = 20, batch_size:int = 32, step_size:float = 0.05):
	data, target = return_with_target(vocab_size)
	logistic = Classifier(vocab_size)

	for epoch in range(epoch_num):
		for X, t in iteration(data, target, batch_size):
			output = logistic.forward(np.matrix(X))
			grad = logistic.backward(np.matrix(X), output, np.array(t))
			logistic.W += step_size * grad

		outputs = logistic.forward(np.matrix(data))
		acc_score = accuracy(outputs, np.array(target))
		loss_score = loss(outputs, np.array(target))
		print(f'accuracy => {acc_score:.2f} \ loss => {loss_score:.2f}')



if __name__ == '__main__':
    train()