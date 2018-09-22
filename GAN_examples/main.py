import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
import GAN_examples.wgan_mnist as WGAN_MNIST
import GAN_examples.dcgan_mnist as DCGAN_MNIST
import GAN_examples.test as test

def main():
	# DATASET
	dataloader = DataLoader(DCGAN_MNIST.get_mnist_dataset(), batch_size=8, shuffle=True, num_workers=4)

	# MODELS
	G = DCGAN_MNIST.Generator(100)
	D = DCGAN_MNIST.Discriminator()

	G.cuda()
	D.cuda()
	# TRAINING

	DCGAN_MNIST.train(G, D, dataloader)


if __name__ == '__main__':
	main()

	#dataset = DCGAN_MNIST.get_mnist_dataset()
	#test.interpret_by_model_model(dataset[15000][0], '../models/mnist_g')

	#test.show_generator_example('../models/mnist_g', 64)
