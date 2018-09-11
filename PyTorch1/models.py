import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGAN_Generator(nn.Module):
	def __init__(self, input_shape):
		super(DCGAN_Generator, self).__init__()
		filters = 64
		self.deconv1 = nn.ConvTranspose2d(input_shape, filters * 16, 4)  # 4x4
		torch.nn.init.kaiming_normal_(self.deconv1.weight, nonlinearity='relu')
		self.deconv1_bn = nn.BatchNorm2d(filters * 16)

		self.deconv2 = nn.ConvTranspose2d(filters * 16, filters * 8, 4, 2, 1)  # 8x8
		torch.nn.init.kaiming_normal_(self.deconv2.weight, nonlinearity='relu')
		self.deconv2_bn = nn.BatchNorm2d(filters * 8)

		self.deconv3 = nn.ConvTranspose2d(filters * 8, filters * 4, 4, 2, 1)  # 16x16
		torch.nn.init.kaiming_normal_(self.deconv3.weight, nonlinearity='relu')
		self.deconv3_bn = nn.BatchNorm2d(filters * 4)

		self.deconv4 = nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1)  # 32x32
		torch.nn.init.kaiming_normal_(self.deconv4.weight, nonlinearity='relu')
		self.deconv4_bn = nn.BatchNorm2d(filters * 2)

		self.deconv5 = nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1)  # 64x64
		torch.nn.init.kaiming_normal_(self.deconv5.weight, nonlinearity='relu')
		self.deconv5_bn = nn.BatchNorm2d(filters)

		self.deconv6 = nn.ConvTranspose2d(filters, 3, 4, 2, 1)  # 128x128
		torch.nn.init.xavier_normal_(self.deconv6.weight, gain=nn.init.calculate_gain('tanh'))


	def forward(self, input):
		x = F.relu(self.deconv1_bn(self.deconv1(input)))
		x = F.relu(self.deconv2_bn(self.deconv2(x)))
		x = F.relu(self.deconv3_bn(self.deconv3(x)))
		x = F.relu(self.deconv4_bn(self.deconv4(x)))
		x = F.relu(self.deconv5_bn(self.deconv5(x)))
		x = F.tanh(self.deconv6(x))
		return x


class DCGAN_Discriminator(nn.Module):
	def __init__(self):
		super(DCGAN_Discriminator, self).__init__()
		filters = 64
		self.conv1 = nn.Conv2d(3, filters, 4, 2, 1) #64x64
		torch.nn.init.kaiming_normal_(self.conv1.weight)

		self.conv2 = nn.Conv2d(filters, filters * 2, 4, 2, 1) #32x32
		torch.nn.init.kaiming_normal_(self.conv2.weight)
		self.conv2_bn = nn.BatchNorm2d(filters * 2)

		self.conv3 = nn.Conv2d(filters * 2, filters * 4, 4, 2, 1) #16x16
		torch.nn.init.kaiming_normal_(self.conv3.weight)
		self.conv3_bn = nn.BatchNorm2d(filters * 4)

		self.conv4 = nn.Conv2d(filters * 4, filters * 8, 4, 2, 1) #8x8
		torch.nn.init.kaiming_normal_(self.conv4.weight)
		self.conv4_bn = nn.BatchNorm2d(filters * 8)

		self.conv5 = nn.Conv2d(filters * 8, filters * 16, 4, 2, 1) #4x4
		torch.nn.init.kaiming_normal_(self.conv5.weight)
		self.conv5_bn = nn.BatchNorm2d(filters * 16)

		self.conv6 = nn.Conv2d(filters * 16, 1, 4, 1, 0)
		torch.nn.init.xavier_normal_(self.conv6.weight, gain=nn.init.calculate_gain('sigmoid'))


	def forward(self, input):
		x = F.leaky_relu(self.conv1(input), 0.2)
		x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
		x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
		x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
		x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
		x = F.sigmoid(self.conv6(x))
		return x