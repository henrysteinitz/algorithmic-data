from random import random


class Distribtuion:
	def sample(self):
		raise NotImplementedError

	def __call__(self):
		return self.sample()


class UniformDistribution(Distribtuion):
	def sample(self):
		return random()

