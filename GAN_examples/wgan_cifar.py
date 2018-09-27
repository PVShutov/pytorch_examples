import logging
from collections import OrderedDict

import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import visdom_wrapper, utils
import GAN_examples.config as config



def get_dataset(train=True):
	return datasets.CIFAR10(config.Datasets_Path+"/cifar", train=train, download=True,
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
				('conv1', nn.ConvTranspose2d(input_shape, filters * 8, 4)),   # 4x4
				('batch_norm1', nn.BatchNorm2d(filters * 8)),
				('relu_conv1', nn.ReLU()),

				('conv2', nn.ConvTranspose2d(filters * 8, filters * 4, 4, 2, 1)),  # 8x8
				('batch_norm2', nn.BatchNorm2d(filters * 4)),
				('relu_conv2', nn.ReLU()),

				('conv3', nn.Conv2d(filters * 4, filters * 4, 3, 1, 1)),  # 8x8
				('batch_norm3', nn.BatchNorm2d(filters * 4)),
				('relu_conv3', nn.ReLU()),

				('conv4', nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1)),  # 16x16
				('batch_norm4', nn.BatchNorm2d(filters * 2)),
				('relu_conv4', nn.ReLU()),

				('conv5', nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1)),  # 32x32
				('batch_norm5', nn.BatchNorm2d(filters)),
				('relu_conv5', nn.ReLU()),

		])
		)


		for m in self.main:
			if isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.normal_(m.weight, 0, 0.002)
				torch.nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1.0)
				torch.nn.init.constant_(m.bias, 0.0)

		self.output = nn.ConvTranspose2d(filters, 3, 3, 1, 1)  # 32x32
		torch.nn.init.normal_(self.output.weight, 0, 0.002)
		torch.nn.init.constant_(self.output.bias, 0.0)

	def forward(self, input):
		x = self.main(input)
		x = F.tanh(self.output(x))
		return x




class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		filters = 128
		self.filters = filters


		self.main = nn.Sequential(OrderedDict([
				('conv1', nn.Conv2d(3, filters, 4, 2, 1)), #16x16
				('layer_norm1', nn.LayerNorm([filters, 16, 16])),
				('lrelu_conv1', nn.LeakyReLU()),

				('conv2', nn.Conv2d(filters, filters * 2, 4, 2, 1)), #8x8
				('layer_norm2', nn.LayerNorm([filters * 2, 8, 8])),
				('lrelu_conv2', nn.LeakyReLU()),

				('conv3', nn.Conv2d(filters * 2, filters * 4, 4, 2, 1)), #4x4
				('layer_norm3', nn.LayerNorm([filters * 4, 4, 4])),
				('lrelu_conv3', nn.LeakyReLU()),

				('conv4', nn.Conv2d(filters * 4, filters * 4, 3, 1, 1)),  # 4x4
				('batch_norm4', nn.LayerNorm([filters * 4, 4, 4])),
				('relu_conv4', nn.LeakyReLU()),

				('conv5', nn.Conv2d(filters * 4, filters * 8, 4, 1, 0)), #1x1
				('layer_norm5', nn.LayerNorm([filters * 8, 1, 1])),
				('lrelu_conv5', nn.LeakyReLU())
			])
		)


		for m in self.main:
			if isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.kaiming_uniform_(m.weight)
				torch.nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.LayerNorm):
				m.weight.data.fill_(1.0)
				torch.nn.init.constant_(m.bias, 0.0)

		self.output = nn.Linear(filters * 8, 1)
		torch.nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('linear'))
		torch.nn.init.constant_(self.output.bias, 0.0)

	def forward(self, input):
		x = self.main(input)
		x = self.output(x.view(-1, self.filters * 8))
		return x



from torch import autograd
def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
	cuda0 = torch.device('cuda:0')
	alpha = torch.rand(BATCH_SIZE, 1).to(cuda0)
	alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 32, 32)
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	disc_interpolates = netD(interpolates)
	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
	                          grad_outputs=torch.ones(disc_interpolates.size()).to(cuda0),
	                          create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
	return gradient_penalty




def train(G, D, dataloader, mode="vanilla"):
	vis = visdom.Visdom(base_url=config.Visdom_Base_Url)
	cuda0 = torch.device('cuda:0')
	cpu0 = torch.device('cpu')

	# LOSS + OPTIMIZER

	if mode != 'vanilla':
		G_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0., 0.9))
		D_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0., 0.9))
	else:
		G_optimizer = optim.RMSprop(G.parameters(), lr=0.0001)
		D_optimizer = optim.RMSprop(D.parameters(), lr=0.0001)

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


			utils.exp_lr_scheduler(G_optimizer, num_iter, lr=0.0001, factor=0.986)
			utils.exp_lr_scheduler(D_optimizer, num_iter, lr=0.0001, factor=0.986)


			# Discriminator step
			#real_d_iter = 100 if g_iter_count < 20 or g_iter_count % 500 == 0 else d_iter
			for i in range(d_iter):
				X = batch.view(-1, 3, 32, 32)

				G_result = G(torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0))

				D.zero_grad()
				D_real_loss = D(X).mean()
				D_fake_loss = D(G_result).mean()
				D_train_loss = D_fake_loss - D_real_loss

				if mode == 'gp':
					D_train_loss += calc_gradient_penalty(D, X, G_result, mini_batch)

				D_train_loss.backward()
				D_optimizer.step()

				if mode == 'vanilla':
					for p in D.parameters():
						p.data.clamp_(-0.01, 0.01)

				if num_iter % 10 == 0 and i == d_iter-1:
					visdom_wrapper.line_pass(vis, num_iter, -D_train_loss.item(), 'rgb(235, 99, 99)', "Train", name='d_loss', title='Loss')



			# Generator step
			for i in range(g_iter):
				G_result = G(torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0))
				G.zero_grad()
				G_train_loss = -D(G_result).mean()
				G_train_loss.backward()
				G_optimizer.step()

				if num_iter % 10 == 0:
					visdom_wrapper.line_pass(vis, num_iter, G_train_loss.item(), 'rgb(99, 153, 235)', "Train", name='g_loss', title='Loss')

				g_iter_count += 1


			# Other tasks
			num_iter += 1
			if num_iter % 100 == 0:
				torch.save(G, config.Models_Path + "/wgan_cifar_g")
				torch.save(D, config.Models_Path + "/wgan_cifar_d")

			if num_iter % 10 == 0:
				vis.images(G(z_fixed).to(cpu0)*0.5 + 0.5,	nrow=4, opts=dict(title='Generator updates'), win="Generator_out")

		logging.info("WGAN-GP epoch {0} end".format(epoch))
