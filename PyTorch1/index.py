import warnings

warnings.filterwarnings('ignore')


import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import PyTorch1.models as models, PyTorch1.datasets as datasets, PyTorch1.training as training, PyTorch1.test as test


def main():
	# DATASET
	pokemon_dataset = datasets.GetPokemonDataset()
	dataloader = DataLoader(pokemon_dataset, batch_size=16, shuffle=True, num_workers=4)


	# MODELS
	G = models.DCGAN_Generator(100)
	D = models.DCGAN_Discriminator()

	G.cuda()
	D.cuda()
	# TRAINING

	training.DCGAN(G, D, dataloader)



if __name__ == '__main__':
	main()
