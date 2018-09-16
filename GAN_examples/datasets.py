import os
import torch

from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import GAN_examples.config as config

class ExampleDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		images_dir = os.listdir(root_dir)
		self.images = []
		for image in images_dir:
			if '.png' in image:
				self.images.append(image)
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_path = os.path.join(self.root_dir, self.images[idx])
		sample = Image.open(img_path)
		if self.transform:
			sample = self.transform(sample)
		return torch.from_numpy(np.array(sample))

def GetDataset():
	transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
	return ExampleDataset(config.ExampleDataset_path, transform)