from random import randint, random


class Distribution:
	def sample(self):
		raise NotImplementedError

	def __call__(self):
		return self.sample()


class UniformDistribution(Distribution):
	def sample(self):
		return random()


class BitDistribution(Distribution):
	def sample(self):
		return randint(0,1)


class SignedBitDistribution(Distribution):
	def sample(self):
		return (randint(0,1) * 2) - 1