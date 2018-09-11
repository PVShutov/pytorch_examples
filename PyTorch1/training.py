import time
import visdom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



import logging


def DCGAN(G, D, dataloader):
	vis = visdom.Visdom()
	cuda0 = torch.device('cuda:0')
	cpu0 = torch.device('cpu')


	# LOSS + OPTIMIZER
	BCE_loss = nn.BCELoss()
	G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.9))
	D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.9))

	logging.info("DCGAN training start")


	z_fixed = torch.randn((8, 100)).view(-1, 100, 1, 1).to(cuda0)

	num_iter = 0
	for epoch in range(10000):
		for batch in dataloader:

			mini_batch = batch.size()[0]

			y_real = torch.ones(mini_batch, device=cuda0)
			y_fake = torch.zeros(mini_batch, device=cuda0)

			batch = batch.to(cuda0)

			# Discriminator step
			D.zero_grad()

			D_real_loss = BCE_loss(D(batch.view(-1, 3, 128, 128)).squeeze(), y_real)

			z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0)
			G_result = G(z)

			D_result = D(G_result).squeeze()
			D_fake_loss = BCE_loss(D_result, y_fake)
			D_fake_score = D_result.data.mean()
			D_train_loss = D_real_loss + D_fake_loss


			vis.line(X=[num_iter], Y=[D_fake_score.item()], win="DCGAN_Fakse_Score", update='append', name='Fake_score')
			vis.line(X=[num_iter], Y=[D_train_loss.item()], win="DCGAN_Train", update='append', name='D_loss')

			D_train_loss.backward()
			D_optimizer.step()




			# Generator step
			G.zero_grad()

			z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0)

			G_result = G(z_)
			G_train_loss = BCE_loss(D(G_result).squeeze(), y_real)

			vis.line(X=[num_iter], Y=[G_train_loss.item()],	win="DCGAN_Train", update='append',	name='G_loss')

			G_train_loss.backward()
			G_optimizer.step()




			# Other tasks
			num_iter += 1
			if num_iter % 10:
				torch.save(G, "./models/pokemon_g")
				torch.save(D, "./models/pokemon_d")

			vis.images(G(z_fixed).to(cpu0),	opts=dict(title='Generator updates'), win="Generator_out")

		logging.info("DCGAN epoch {0} end".format(epoch))