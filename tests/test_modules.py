"""ProtoTorch modules test suite."""

import unittest

import numpy as np
import torch

from prototorch.modules import prototypes, losses


class TestPrototypes(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[0, -1, -2], [10, 11, 12], [0, 0, 0], [2, 2, 2]],
            dtype=torch.float32)
        self.y = torch.tensor([0, 0, 1, 1])
        self.gen = torch.manual_seed(42)

    def test_addprototypes1d_init_without_input_dim(self):
        with self.assertRaises(NameError):
            _ = prototypes.AddPrototypes1D(nclasses=1)

    def test_addprototypes1d_init_without_nclasses(self):
        with self.assertRaises(NameError):
            _ = prototypes.AddPrototypes1D(input_dim=1)

    def test_addprototypes1d_init_without_pdist(self):
        p1 = prototypes.AddPrototypes1D(input_dim=6,
                                        nclasses=2,
                                        prototypes_per_class=4,
                                        prototype_initializer='ones')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.ones(8, 6)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_addprototypes1d_init_without_data(self):
        pdist = [2, 2]
        p1 = prototypes.AddPrototypes1D(input_dim=3,
                                        prototype_distribution=pdist,
                                        prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(4, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    # def test_addprototypes1d_init_torch_pdist(self):
    #     pdist = torch.tensor([2, 2])
    #     p1 = prototypes.AddPrototypes1D(input_dim=3,
    #                                     prototype_distribution=pdist,
    #                                     prototype_initializer='zeros')
    #     protos = p1.prototypes
    #     actual = protos.detach().numpy()
    #     desired = torch.zeros(4, 3)
    #     mismatch = np.testing.assert_array_almost_equal(actual,
    #                                                     desired,
    #                                                     decimal=5)
    #     self.assertIsNone(mismatch)

    def test_addprototypes1d_init_with_ppc(self):
        p1 = prototypes.AddPrototypes1D(data=[self.x, self.y],
                                        prototypes_per_class=2,
                                        prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(4, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_addprototypes1d_init_with_pdist(self):
        p1 = prototypes.AddPrototypes1D(data=[self.x, self.y],
                                        prototype_distribution=[6, 9],
                                        prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(15, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_addprototypes1d_func_initializer(self):
        def my_initializer(*args, **kwargs):
            return torch.full((2, 99), 99), torch.tensor([0, 1])

        p1 = prototypes.AddPrototypes1D(input_dim=99,
                                        nclasses=2,
                                        prototypes_per_class=1,
                                        prototype_initializer=my_initializer)
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = 99 * torch.ones(2, 99)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_addprototypes1d_forward(self):
        p1 = prototypes.AddPrototypes1D(data=[self.x, self.y])
        protos, _ = p1()
        actual = protos.detach().numpy()
        desired = torch.ones(2, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x, self.y, self.gen
        _ = torch.seed()


class TestLosses(unittest.TestCase):
    def setUp(self):
        pass

    def test_glvqloss_init(self):
        _ = losses.GLVQLoss()

    def tearDown(self):
        pass
