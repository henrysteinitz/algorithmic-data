from task import Copy, Identity, Reverse
import torch
from torch.utils.data import DataLoader
import unittest


class TaskTests(unittest.TestCase):
	def test_identity(self):
		dataset = Identity(size=20, min_length=5, max_length=5)
		loader = DataLoader(dataset, batch_size=1, shuffle=True)

		for x, y in loader:
			x_strip = x[:, 0:-1, 0:-1]
			self.assertTrue(torch.equal(x_strip, y))


	def test_copy(self):
		dataset = Copy(num_copies=3, size=20, min_length=5, max_length=5)
		loader = DataLoader(dataset, batch_size=1, shuffle=True)

		for x, y in loader:
			x_strip = x[:, 0:-1, 0:-1]

			# The tensors are copied along dimension 1 since dimension 0
			# is the batch dimension.
			self.assertTrue(torch.equal(
				torch.cat([x_strip, x_strip, x_strip], dim=1),
				y
			))

	def test_reverse(self):
		dataset = Reverse(size=20, min_length=5, max_length=5)
		loader = DataLoader(dataset, batch_size=1, shuffle=True)

		for x, y in loader:
			x_strip = x[:, 0:-1, 0:-1]

			# The tensors are copied along dimension 1 since dimension 0
			# is the batch dimension.
			self.assertTrue(torch.equal(torch.flip(x_strip, dims=[1]), y))


if __name__ == '__main__':
    unittest.main()