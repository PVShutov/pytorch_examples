import warnings

warnings.filterwarnings('ignore')


import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import GAN_examples.models as models, GAN_examples.datasets as datasets, GAN_examples.training as training, GAN_examples.test as test


def main():
	# DATASET
	dataloader = DataLoader(datasets.Dataset(), batch_size=16, shuffle=True, num_workers=4)


	# MODELS
	G = models.DCGAN_Generator(100)
	D = models.DCGAN_Discriminator(output_type="LINEAR")

	G.cuda()
	D.cuda()
	# TRAINING

	training.WGAN(G, D, dataloader)



if __name__ == '__main__':
	main()
