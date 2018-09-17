import time
import visdom
import torch
import torch.nn as nn
import torch.optim as optim


import logging


def line_pass(visdom, x, y, color, win, name, title):
	if win is None or not visdom.win_exists(win, None):
		update = False
	else:
		update = True
	trace = dict(x=[x], y=[y], mode="lines", type='custom', line=dict(color=color), name=name)
	layout = dict(title=title, margin=dict(l=35, r=5, b=25, t=25, pad=4))
	visdom._send({'data': [trace], 'layout': layout, 'win': win, 'name': name, 'append': update}, endpoint = 'update' if update else 'events')


def show_gradient_norm(visdom, model, win, name):
	X = []
	rownames = []
	for n, f in model.named_parameters():
		norm = f.grad.norm().to('cpu')
		X.append(norm)
		rownames.append(n)

	visdom.bar(X=X, win=win,
        opts=dict(
	        title=name,
            rownames=rownames,
	        marginleft=35,
	        marginright=15,
	        margintop=25,
	        marginbottom=35
        ))



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

			line_pass(vis, num_iter, D_train_loss.item(), 'rgb(235, 99, 99)', "DCGAN_Train", name='d_loss',
			          title='Loss')
			line_pass(vis, num_iter, D_fake_score, 'rgb(96, 204, 173)', "DCGAN_FakeScore", name='fake score',
			          title='Fake score')

			D_train_loss.backward()
			D_optimizer.step()

			show_gradient_norm(vis, D, 'D_Grad', 'Discriminator gradients norm')

			# Generator step
			G.zero_grad()

			z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(cuda0)

			G_result = G(z_)
			G_train_loss = BCE_loss(D(G_result).squeeze(), y_real)

			vis.line(X=[num_iter], Y=[G_train_loss.item()],	win="DCGAN_Train", update='append',	name='G_loss')

			G_train_loss.backward()
			G_optimizer.step()

			# Other tasks
			show_gradient_norm(vis, G, 'G_Grad', 'Generator gradients norm')

			# Other tasks
			num_iter += 1
			#if num_iter % 10:
			#	torch.save(G, "./models/pokemon_g")
			#	torch.save(D, "./models/pokemon_d")
			vis.images(G(z_fixed).to(cpu0),	nrow=4, opts=dict(title='Generator updates'), win="Generator_out")

		logging.info("DCGAN epoch {0} end".format(epoch))


from torch import autograd
def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
	cuda0 = torch.device('cuda:0')
	alpha = torch.rand(BATCH_SIZE, 1).to(cuda0)
	alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
	alpha = alpha.view(BATCH_SIZE, 3, 128, 128)
	alpha = alpha.to(cuda0)
	interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
	interpolates = interpolates.to(cuda0)
	interpolates = autograd.Variable(interpolates, requires_grad=True)
	disc_interpolates = netD(interpolates)
	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
	                          grad_outputs=torch.ones(disc_interpolates.size()).to(cuda0),
	                          create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
	return gradient_penalty


def WGAN(G, D, dataloader):
	vis = visdom.Visdom()
	cuda0 = torch.device('cuda:0')
	cpu0 = torch.device('cpu')


	# LOSS + OPTIMIZER
	G_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0., 0.9))
	D_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0., 0.9))
	#G_optimizer = optim.RMSprop(G.parameters(), lr=2e-4)
	#D_optimizer = optim.RMSprop(D.parameters(), lr=2e-4)


	logging.info("WGAN-GP training start")


	def get_uniform(size, a, b):
		return (b-a)*torch.rand(size) + a

	z_fixed = get_uniform((8, 100), -1.0, 1.0).view(-1, 100, 1, 1).to(cuda0)


	d_iter = 5
	g_iter = 1
	g_iter_count = 0

	num_iter = 0
	for epoch in range(10000):
		for batch in dataloader:
			mini_batch = batch.size()[0]

			batch = batch.to(cuda0)

			# Discriminator step


			real_d_iter = 100 if g_iter_count < 20 or g_iter_count % 500 == 0 else d_iter
			for i in range(real_d_iter):
				X = batch.view(-1, 3, 128, 128)
				G_result = G(get_uniform((mini_batch, 100), -1.0, 1.0).view(-1, 100, 1, 1).to(cuda0))
				D.zero_grad()
				D_real_loss = D(X).squeeze().mean()
				D_fake_loss = D(G_result).squeeze().mean()
				D_train_loss = D_fake_loss - D_real_loss + calc_gradient_penalty(D, X, G_result, mini_batch)

				if i == real_d_iter-1:
					line_pass(vis, num_iter, D_train_loss.item(), 'rgb(235, 99, 99)', "DCGAN_Train", name='d_loss', title='Loss')
					line_pass(vis, num_iter, D_fake_loss.item(), 'rgb(96, 204, 173)', "DCGAN_FakeScore", name='fake score', title='Fake loss')

				D_train_loss.backward()
				D_optimizer.step()


			show_gradient_norm(vis, D, 'D_Grad', 'Discriminator gradients norm')


			# Generator step
			for i in range(g_iter):
				G_result = G(get_uniform((mini_batch, 100), -1.0, 1.0).view(-1, 100, 1, 1).to(cuda0))
				G.zero_grad()
				G_train_loss = -D(G_result).squeeze().mean()

				line_pass(vis, num_iter, G_train_loss.item(), 'rgb(99, 153, 235)', "DCGAN_Train", name='g_loss', title='Loss')

				G_train_loss.backward()
				G_optimizer.step()

				g_iter_count += 1

			show_gradient_norm(vis, G, 'G_Grad', 'Generator gradients norm')

			# Other tasks
			num_iter += 1
			if num_iter % 100 == 0:
				torch.save(G, "./models/pokemon_g")
				torch.save(D, "./models/pokemon_d")
			vis.images(G(z_fixed).to(cpu0),	nrow=4, opts=dict(title='Generator updates'), win="Generator_out")

		logging.info("WGAN-GP epoch {0} end".format(epoch))
