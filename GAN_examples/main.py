import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
import GAN_examples.wgan_mnist as WGAN_MNIST

def main():
	# DATASET
	dataloader = DataLoader(WGAN_MNIST.get_mnist_dataset(), batch_size=16, shuffle=True, num_workers=4)

	# MODELS
	G = WGAN_MNIST.Generator(100)
	D = WGAN_MNIST.Discriminator()

	G.cuda()
	D.cuda()
	# TRAINING

	WGAN_MNIST.train(G, D, dataloader)


if __name__ == '__main__':
	main()
