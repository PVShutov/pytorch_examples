import logging

import visdom
import torch
import torch.nn as nn
import torch.optim as optim


def interpret_by_model_model(source, path_to_generator):
	vis = visdom.Visdom()
	cuda0 = torch.device('cuda:0')
	cpu0 = torch.device('cpu')

	G = torch.load(path_to_generator)

	logging.info("DCGAN interpret start")

	vis.images(
		source*0.5 + 0.5,
		opts=dict(title='source'),
		win="source",
	)
	source = source.to(cuda0)

	# LOSS + OPTIMIZER
	L1Loss = nn.L1Loss()

	def get_uniform(size, a, b):
		return (b-a)*torch.rand(size, requires_grad=True, device=cuda0) + a

	z_0 = torch.randn(100, requires_grad=True, device=cuda0)#get_uniform(100, -1.0, 1.0)
	G_optimizer = optim.Adam([z_0], lr=0.5)

	for i in range(500):

		G_result = G(z_0.view(-1, 100, 1, 1))

		vis.images(
			G_result.to(cpu0),
			opts=dict(title='new'),
			win="new",
		)

		G_loss = L1Loss(G_result.squeeze(), source)
		G_loss.backward()
		G_optimizer.step()
		z_0.grad.data.zero_()

		vis.line(
			X=[i],
			Y=[G_loss.item()],
			win="GAN_Minimize",
			update='append',
			name='minimize'
		)

	logging.info("DCGAN interpret finish")



def show_generator_example(path_to_generator, example_count=8):
	cuda0 = torch.device('cuda:0')
	cpu0 = torch.device('cpu')
	vis = visdom.Visdom()

	G = torch.load(path_to_generator)


	def get_uniform(size, a, b):
		return (b-a)*torch.rand(size) + a
	z_fixed = get_uniform((8, 100), -1.0, 1.0).view(-1, 100, 1, 1).to(cuda0)
	vis.images(
		G(z_fixed).to(cpu0)*0.5 + 0.5,
		opts=dict(title='Generator updates'),
		win="Generator_out",
	)