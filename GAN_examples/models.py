import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class DCGAN_Generator(nn.Module):
	def __init__(self, input_shape):
		super(DCGAN_Generator, self).__init__()
		filters = 64

		self.main = nn.Sequential(OrderedDict([
				('conv1', nn.ConvTranspose2d(input_shape, filters * 16, 4)),   # 4x4
				('batch_norm1', nn.BatchNorm2d(filters * 16)),
				('relu_conv1', nn.ReLU(True)),

				('conv2', nn.ConvTranspose2d(filters * 16, filters * 8, 4, 2, 1)),  # 8x8
				('batch_norm2', nn.BatchNorm2d(filters * 8)),
				('relu_conv2', nn.ReLU(True)),

				('conv3', nn.ConvTranspose2d(filters * 8, filters * 4, 4, 2, 1)),  # 16x16
				('batch_norm3', nn.BatchNorm2d(filters * 4)),
				('relu_conv3', nn.ReLU(True)),

				('conv4', nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1)),  # 32x32
				('batch_norm4', nn.BatchNorm2d(filters * 2)),
				('relu_conv4', nn.ReLU(True)),

				('conv5', nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1)),  # 64x64
				('batch_norm5', nn.BatchNorm2d(filters)),
				('relu_conv5', nn.ReLU(True)),
			])
		)


		for m in self.main:
			if isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
				torch.nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1.0)
				torch.nn.init.constant_(m.bias, 0.0)

		self.output = nn.ConvTranspose2d(filters, 3, 4, 2, 1)  # 128x128
		torch.nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('tanh'))
		torch.nn.init.constant_(self.output.bias, 0.0)

	def forward(self, input):
		x = self.main(input)
		x = F.tanh(self.output(x))
		return x




class DCGAN_Discriminator(nn.Module):
	def __init__(self, output_type="SIGMOID"):
		super(DCGAN_Discriminator, self).__init__()

		filters = 64
		self.filters = filters


		self.main = nn.Sequential(OrderedDict([
				('conv1', nn.Conv2d(3, filters, 4, 2, 1)), #64x64
				('layer_norm1', nn.LayerNorm([filters, 64, 64])),
				('lrelu_conv1', nn.LeakyReLU(True)),

				('conv2', nn.Conv2d(filters, filters * 2, 4, 2, 1)), #32x32
				('layer_norm2', nn.LayerNorm([filters * 2, 32, 32])),
				('lrelu_conv2', nn.LeakyReLU(True)),

				('conv3', nn.Conv2d(filters * 2, filters * 4, 4, 2, 1)), #16x16
				('layer_norm3', nn.LayerNorm([filters * 4, 16, 16])),
				('lrelu_conv3', nn.LeakyReLU(True)),

				('conv4', nn.Conv2d(filters * 4, filters * 8, 4, 2, 1)), #8x8
				('layer_norm4', nn.LayerNorm([filters * 8, 8, 8])),
				('lrelu_conv4', nn.LeakyReLU(True)),

				('conv5', nn.Conv2d(filters * 8, filters * 16, 4, 2, 1)), #4x4
				('layer_norm5', nn.LayerNorm([filters * 16, 4, 4])),
				('lrelu_conv5', nn.LeakyReLU(True)),
			])
		)


		for m in self.main:
			if isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.kaiming_uniform_(m.weight)
				torch.nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.LayerNorm):
				m.weight.data.fill_(1.0)
				torch.nn.init.constant_(m.bias, 0.0)


		if output_type == "SIGMOID":
			self.output = nn.Conv2d(filters * 16, 1, 4, 1, 0)
			torch.nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('sigmoid'))
			torch.nn.init.constant_(self.output.bias, 0.0)
		else:
			self.output = nn.Linear(filters * 16 * 4 * 4, 1)
			torch.nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('linear'))
			torch.nn.init.constant_(self.output.bias, 0.0)

		self.output_type = output_type



	def forward(self, input):
		x = self.main(input)

		if self.output_type == "SIGMOID":
			x = F.sigmoid(self.output(x))
		else:
			x = self.output(x.view(-1, self.filters * 16 * 4 * 4))
		return x