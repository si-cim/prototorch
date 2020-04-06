"""ProtoTorch functions test suite."""

import unittest

import numpy as np
import torch

from prototorch.functions import (activations, competitions, distances,
                                  initializers)


class TestDistances(unittest.TestCase):
    def setUp(self):
        self.nx, self.mx = 32, 2048
        self.ny, self.my = 8, 2048
        self.x = torch.randn(self.nx, self.mx)
        self.y = torch.randn(self.ny, self.my)

    def test_manhattan(self):
        actual = distances.lpnorm_distance(self.x, self.y, p=1)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=1,
                    keepdim=False,
                )
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=2)
        self.assertIsNone(mismatch)

    def test_euclidean(self):
        actual = distances.euclidean_distance(self.x, self.y)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=3)
        self.assertIsNone(mismatch)

    def test_squared_euclidean(self):
        actual = distances.squared_euclidean_distance(self.x, self.y)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )**2
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=2)
        self.assertIsNone(mismatch)

    def test_lpnorm_p0(self):
        actual = distances.lpnorm_distance(self.x, self.y, p=0)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=0,
                    keepdim=False,
                )
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=4)
        self.assertIsNone(mismatch)

    def test_lpnorm_p2(self):
        actual = distances.lpnorm_distance(self.x, self.y, p=2)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=4)
        self.assertIsNone(mismatch)

    def test_lpnorm_p3(self):
        actual = distances.lpnorm_distance(self.x, self.y, p=3)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=3,
                    keepdim=False,
                )
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=4)
        self.assertIsNone(mismatch)

    def test_lpnorm_pinf(self):
        actual = distances.lpnorm_distance(self.x, self.y, p=float('inf'))
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=float('inf'),
                    keepdim=False,
                )
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=4)
        self.assertIsNone(mismatch)

    def test_omega_identity(self):
        omega = torch.eye(self.mx, self.my)
        actual = distances.omega_distance(self.x, self.y, omega=omega)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )**2
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=2)
        self.assertIsNone(mismatch)

    def test_lomega_identity(self):
        omega = torch.eye(self.mx, self.my)
        omegas = torch.stack([omega for _ in range(self.ny)], dim=0)
        actual = distances.lomega_distance(self.x, self.y, omegas=omegas)
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )**2
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=2)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x, self.y


class TestActivations(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(1024, 1)

    def test_registry(self):
        self.assertIsNotNone(activations.ACTIVATIONS)

    def test_funcname_deserialization(self):
        flist = ['identity', 'sigmoid_beta', 'swish_beta']
        for funcname in flist:
            f = activations.get_activation(funcname)
            iscallable = callable(f)
            self.assertTrue(iscallable)

    def test_callable_deserialization(self):
        def dummy(x, **kwargs):
            return x

        for f in [dummy, lambda x: x]:
            f = activations.get_activation(f)
            iscallable = callable(f)
            self.assertTrue(iscallable)
            self.assertEqual(1, f(1))

    def test_unknown_deserialization(self):
        for funcname in ['blubb', 'foobar']:
            with self.assertRaises(NameError):
                _ = activations.get_activation(funcname)

    def test_identity(self):
        actual = activations.identity(self.x)
        desired = self.x
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_sigmoid_beta1(self):
        actual = activations.sigmoid_beta(self.x, beta=1)
        desired = torch.sigmoid(self.x)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_swish_beta1(self):
        actual = activations.swish_beta(self.x, beta=1)
        desired = self.x * torch.sigmoid(self.x)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x


class TestCompetitions(unittest.TestCase):
    def setUp(self):
        pass

    def test_wtac(self):
        d = torch.tensor([[2., 3., 1.99, 3.01], [2., 3., 2.01, 3.]])
        labels = torch.tensor([0, 1, 2, 3])
        actual = competitions.wtac(d, labels)
        desired = torch.tensor([2, 0])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_wtac_one_hot(self):
        d = torch.tensor([[1.99, 3.01], [3., 2.01]])
        labels = torch.tensor([[0, 1], [1, 0]])
        actual = competitions.wtac(d, labels)
        desired = torch.tensor([[0, 1], [1, 0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_knnc_k1(self):
        d = torch.tensor([[2., 3., 1.99, 3.01], [2., 3., 2.01, 3.]])
        labels = torch.tensor([0, 1, 2, 3])
        actual = competitions.knnc(d, labels, k=1)
        desired = torch.tensor([2, 0])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        pass


class TestInitializers(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[0, -1, -2], [10, 11, 12], [0, 0, 0], [2, 2, 2]],
            dtype=torch.float32)
        self.y = torch.tensor([0, 0, 1, 1])
        self.gen = torch.manual_seed(42)

    def test_registry(self):
        self.assertIsNotNone(initializers.INITIALIZERS)

    def test_funcname_deserialization(self):
        flist = [
            'zeros', 'ones', 'rand', 'randn', 'stratified_mean',
            'stratified_random'
        ]
        for funcname in flist:
            f = initializers.get_initializer(funcname)
            iscallable = callable(f)
            self.assertTrue(iscallable)

    def test_callable_deserialization(self):
        def dummy(x):
            return x

        for f in [dummy, lambda x: x]:
            f = initializers.get_initializer(f)
            iscallable = callable(f)
            self.assertTrue(iscallable)
            self.assertEqual(1, f(1))

    def test_unknown_deserialization(self):
        for funcname in ['blubb', 'foobar']:
            with self.assertRaises(NameError):
                _ = initializers.get_initializer(funcname)

    def test_zeros(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.zeros(self.x, self.y, pdist)
        desired = torch.zeros(2, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_ones(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.ones(self.x, self.y, pdist)
        desired = torch.ones(2, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_rand(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.rand(self.x, self.y, pdist)
        desired = torch.rand(2, 3, generator=torch.manual_seed(42))
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_randn(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.randn(self.x, self.y, pdist)
        desired = torch.randn(2, 3, generator=torch.manual_seed(42))
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_mean_equal1(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.stratified_mean(self.x, self.y, pdist)
        desired = torch.tensor([[5., 5., 5.], [1., 1., 1.]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_random_equal1(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.stratified_random(self.x, self.y, pdist)
        desired = torch.tensor([[0., -1., -2.], [2., 2., 2.]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_mean_equal2(self):
        pdist = torch.tensor([2, 2])
        actual, _ = initializers.stratified_mean(self.x, self.y, pdist)
        desired = torch.tensor([[5., 5., 5.], [5., 5., 5.], [1., 1., 1.],
                                [1., 1., 1.]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_mean_unequal(self):
        pdist = torch.tensor([1, 3])
        actual, _ = initializers.stratified_mean(self.x, self.y, pdist)
        desired = torch.tensor([[5., 5., 5.], [1., 1., 1.], [1., 1., 1.],
                                [1., 1., 1.]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_random_unequal(self):
        pdist = torch.tensor([1, 3])
        actual, _ = initializers.stratified_random(self.x, self.y, pdist)
        desired = torch.tensor([[0., -1., -2.], [2., 2., 2.], [0., 0., 0.],
                                [0., 0., 0.]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x, self.y, self.gen
        _ = torch.seed()
