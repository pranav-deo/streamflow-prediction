import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self, input_size, hidden1_size, hidden2_size):

		super(Net, self).__init__()
		self.lstm1 = nn.LSTM(input_size, hidden1_size, num_layers = 2, bidirectional = False)
		# self.lstm2 = nn.LSTM(hidden1_size, hidden2_size, num_layers = 2, bidirectional = False)
		self.softmax1 = nn.Softmax()
		self.fc2 = nn.Linear(hidden1_size, hidden2_size)
		self.softmax2 = nn.Softmax()
		self.fc3 = nn.Linear(hidden2_size, 1)

	def forward(self, x):
		out = self.lstm1(x)
		out = self.softmax1(out[0])
		# out = F.relu(out[0])
		# out = self.lstm2(out)	
		# out = self.softmax1(out[0])
		# print(out)
		# out = F.relu(out[0])
		# print(out.size)
		out = self.fc2(out)
		out = self.softmax1(out)
		# out = F.relu(out)
		out = self.fc3(out)
		return out


