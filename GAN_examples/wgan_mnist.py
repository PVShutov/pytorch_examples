import logging
from collections import OrderedDict

import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import visdom_wrapper
import GAN_examples.config as config



def get_mnist_dataset(train=True):
	return datasets.MNIST(config.Datasets_Path+"/mnist", train=train, download=True,
		               transform=transforms.Compose([
			               transforms.Resize((32, 32)),
			               transforms.ToTensor(),
			               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		               ]))

class Generator(nn.Module):
	def __init__(self, input_shape):
		super(Generator, self).__init__()
		filters = 32

		self.main = nn.Sequential(OrderedDict([
				('conv1', nn.ConvTranspose2d(input_shape, filters * 4, 4)),   # 4x4
				('batch_norm1', nn.BatchNorm2d(filters * 4)),
				('relu_conv1', nn.ReLU()),

				('conv2', nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1)),  # 8x8
				('batch_norm2', nn.BatchNorm2d(filters * 2)),
				('relu_conv2', nn.ReLU()),

				('conv3', nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1)),  # 16x16
				('batch_norm3', nn.BatchNorm2d(filters )),
				('relu_conv3', nn.ReLU()),
			])
		)


		for m in self.main:
			if isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
				torch.nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1.0)
				torch.nn.init.constant_(m.bias, 0.0)

		self.output = nn.ConvTranspose2d(filters, 1, 4, 2, 1)  # 32x32
		torch.nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('tanh'))
		torch.nn.init.constant_(self.output.bias, 0.0)

	def forward(self, input):
		x = self.main(input)
		x = F.tanh(self.output(x))
		return x




class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		filters = 16
		self.filters = filters


		self.main = nn.Sequential(OrderedDict([
				('conv1', nn.Conv2d(1, filters, 4, 2, 1)), #16x16
				('layer_norm1', nn.LayerNorm([filters, 16, 16])),
				('lrelu_conv1', nn.LeakyReLU()),

				('conv2', nn.Conv2d(filters, filters * 2, 4, 2, 1)), #8x8
				('layer_norm2', nn.LayerNorm([filters * 2, 8, 8])),
				('lrelu_conv2', nn.LeakyReLU()),

				('conv3', nn.Conv2d(filters * 2, filters * 4, 4, 2, 1)), #4x4
				('layer_norm3', nn.LayerNorm([filters * 4, 4, 4])),
				('lrelu_conv3', nn.LeakyReLU()),

				('conv4', nn.Conv2d(filters * 4, filters * 8, 4, 2, 1)), #2x2
				('layer_norm4', nn.LayerNorm([filters * 8, 2, 2])),
				('lrelu_conv4', nn.LeakyReLU())
			])
		)


		for m in self.main:
			if isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.kaiming_uniform_(m.weight)
				torch.nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.LayerNorm):
				m.weight.data.fill_(1.0)
				torch.nn.init.constant_(m.bias, 0.0)

		self.output = nn.Linear(filters * 8 * 2 * 2, 1)
		torch.nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('linear'))
		torch.nn.init.constant_(self.output.bias, 0.0)


	def forward(self, input):
		x = self.main(input)
		x = self.output(x.view(-1, self.filters * 8 * 2 * 2))
		return x



def train(G, D, dataloader):
	vis = visdom.Visdom()
	cuda0 = torch.device('cuda:0')
	cpu0 = torch.device('cpu')

	# LOSS + OPTIMIZER
	G_optimizer = optim.RMSprop(G.parameters(), lr=2e-4)
	D_optimizer = optim.RMSprop(D.parameters(), lr=2e-4)

	logging.info("WGAN-GP training start")


	z_fixed = torch.randn((16, 100)).view(-1, 100, 1, 1).to(cuda0)

	d_iter = 5
	g_iter = 1
	g_iter_count = 0

	num_iter = 0
	for epoch in range(10000):
		for batch in dataloader:
			batch = batch[0]
			mini_batch = batch.size()[0]

			batch = batch.to(cuda0)

			# Discriminator step


			real_d_iter = d_iter#100 if g_iter_count < 20 or g_iter_count % 500 == 0 else d_iter
			for i in range(real_d_iter):
				X = batch.view(-1, 1, 32, 32)
				G_result = G(torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0))
				D.zero_grad()
				D_real_loss = D(X).mean()
				D_fake_loss = D(G_result).mean()
				D_train_loss = D_fake_loss - D_real_loss

				if num_iter % 10 == 0 and i == real_d_iter-1:
					visdom_wrapper.line_pass(vis, num_iter, -D_train_loss.item(), 'rgb(235, 99, 99)', "Train", name='d_loss', title='Loss')

				D_train_loss.backward()
				D_optimizer.step()

				for p in D.parameters():
					p.data.clamp_(-0.01, 0.01)


			# Generator step
			for i in range(g_iter):
				G_result = G(torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0))
				G.zero_grad()
				G_train_loss = -D(G_result).mean()

				if num_iter % 10 == 0:
					visdom_wrapper.line_pass(vis, num_iter, G_train_loss.item(), 'rgb(99, 153, 235)', "Train", name='g_loss', title='Loss')

				G_train_loss.backward()
				G_optimizer.step()

				g_iter_count += 1


			# Other tasks
			num_iter += 1
			if num_iter % 100 == 0:
				torch.save(G, config.Models_Path + "/wgan_mnist_g")
				torch.save(D, config.Models_Path + "/wgan_mnist_d")

			if num_iter % 10 == 0:
				vis.images(G(z_fixed).to(cpu0)*0.5 + 0.5,	nrow=4, opts=dict(title='Generator updates'), win="Generator_out")

		logging.info("WGAN-GP epoch {0} end".format(epoch))
