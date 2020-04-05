from functools import reduce, partial
from random import random, randrange
import torch
from torch.utils.data import Dataset, DataLoader


# Constants
k_default_size = 1000
k_default_in_shape = (10,)
k_default_min_length = 20
k_default_max_length = 20
k_default_include_stop = True


# Distributions
class Distribtuion:
	def sample(self):
		raise NotImplementedError

	def __call__(self):
		return self.sample()


class UniformDistribution(Distribtuion):
	def sample(self):
		return random()


# Tasks
class AlgorithmicTask(Dataset):
	# f: callable that maps input tensor sequences 
	#          to output tensor sequences
	#
	# distribution: 
	def __init__(self, f=lambda x: x, distribution=UniformDistribution, 
				 size=k_default_size, in_shape=k_default_in_shape, 
				 min_length=k_default_min_length, max_length=k_default_max_length,
				 include_stop=k_default_include_stop):
		self._f = f
		self._distribution = distribution()
		self._size = size
		self._in_shape = in_shape
		self._min_length = min_length
		self._max_length = max_length
		self._include_stop = include_stop
		self._items = {}


	def __len__(self):
		return self._size


	def __getitem__(self, idx):
		if idx in self._items:
			return self._items[idx]

		length = randrange(self._min_length, self._max_length + 1)
		size = reduce(lambda x, y: x * y, [length, *self._in_shape])
		data = [self._distribution() for _ in range(size)]
		x = torch.Tensor(data).view(length, *self._in_shape)
		y = self._f(x)

		self._items[idx] = (x, y)
		return (x, y)


class Identity(AlgorithmicTask):
	def __init__(self, *args, **kwargs):
		super().__init__(lambda x: x, **args)


class Copy(AlgorithmicTask):
	def __init__(self, num_copies=2, *args, **kwargs):
		self._num_copies = num_copies
		super().__init__(f=self._copy, *args, **kwargs)

	def _copy(self, x):
		return torch.cat([x, x], dim=0)



dataset = Copy(num_copies=2, size=20, min_length=5, max_length=5)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for x, y in loader:
	print(x)
	print(y)
