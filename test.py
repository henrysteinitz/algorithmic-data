from task import Identity, Copy
import torch
from torch.utils.data import DataLoader
import unittest


class TaskTests(unittest.TestCase):
	def test_identity(self):
		dataset = Identity(size=20, min_length=5, max_length=5)
		loader = DataLoader(dataset, batch_size=1, shuffle=True)

		for x, y in loader:
			# TODO: Account for stop token
			self.assertTrue(torch.equal(x, y))


	def test_copy(self):
		dataset = Copy(num_copies=3, size=20, min_length=5, max_length=5)
		loader = DataLoader(dataset, batch_size=1, shuffle=True)

		for x, y in loader:
			# TODO: Account for stop token

			# The tensors are copied along dimension 1 since dimension 0
			# is the batch dimension.
			self.assertTrue(torch.equal(torch.cat([x, x, x], dim=1), y))


if __name__ == '__main__':
    unittest.main()