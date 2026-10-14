"""Tests for the normative modeling pipeline (learned CO-normative deviations).

Follows the unittest style of the existing suite (tests/test_rawdata.py, ...).
"""
import unittest

import numpy as np
import torch

from begad_norm.normative import seeding


class TestSeeding(unittest.TestCase):
    def test_seed_all_deterministic(self):
        """Same seed -> identical torch weights; also seeds numpy."""
        seeding.seed_all(42)
        w1 = torch.nn.Linear(4, 4).weight.detach().clone()
        np1 = np.random.rand(4)
        seeding.seed_all(42)
        w2 = torch.nn.Linear(4, 4).weight.detach().clone()
        np2 = np.random.rand(4)
        self.assertTrue(torch.equal(w1, w2))
        self.assertTrue(np.allclose(np1, np2))


if __name__ == "__main__":
    unittest.main()
