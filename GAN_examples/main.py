import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
import GAN_examples.wgan_mnist as WGAN_MNIST
import GAN_examples.dcgan_mnist as DCGAN_MNIST
import GAN_examples.wgan_cifar as WGAN_CIFAR
import GAN_examples.test as test

def main():
	# DATASET
	dataloader = DataLoader(WGAN_CIFAR.get_dataset(), batch_size=16, shuffle=True, num_workers=1)

	# MODELS
	G = WGAN_CIFAR.Generator(100)
	D = WGAN_CIFAR.Discriminator()

	G.cuda()
	D.cuda()
	# TRAINING

	WGAN_CIFAR.train(G, D, dataloader, mode='gp')


if __name__ == '__main__':
	main()

	#dataset = WGAN_MNIST.get_mnist_dataset()
	#test.interpret_by_model_model(dataset[18710][0], '../models/wgan_mnist_g')

	#test.show_generator_example('../models/dcgan_mnist_g', 64)
