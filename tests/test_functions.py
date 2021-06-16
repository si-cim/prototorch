"""ProtoTorch functions test suite."""

import unittest

import numpy as np
import torch

from prototorch.functions import (activations, competitions, distances,
                                  initializers, losses, pooling)


class TestActivations(unittest.TestCase):
    def setUp(self):
        self.flist = ["identity", "sigmoid_beta", "swish_beta"]
        self.x = torch.randn(1024, 1)

    def test_registry(self):
        self.assertIsNotNone(activations.ACTIVATIONS)

    def test_funcname_deserialization(self):
        for funcname in self.flist:
            f = activations.get_activation(funcname)
            iscallable = callable(f)
            self.assertTrue(iscallable)

    # def test_torch_script(self):
    #     for funcname in self.flist:
    #         f = activations.get_activation(funcname)
    #         self.assertIsInstance(f, torch.jit.ScriptFunction)

    def test_callable_deserialization(self):
        def dummy(x, **kwargs):
            return x

        for f in [dummy, lambda x: x]:
            f = activations.get_activation(f)
            iscallable = callable(f)
            self.assertTrue(iscallable)
            self.assertEqual(1, f(1))

    def test_unknown_deserialization(self):
        for funcname in ["blubb", "foobar"]:
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
        actual = activations.sigmoid_beta(self.x, beta=1.0)
        desired = torch.sigmoid(self.x)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_swish_beta1(self):
        actual = activations.swish_beta(self.x, beta=1.0)
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
        d = torch.tensor([[2.0, 3.0, 1.99, 3.01], [2.0, 3.0, 2.01, 3.0]])
        labels = torch.tensor([0, 1, 2, 3])
        actual = competitions.wtac(d, labels)
        desired = torch.tensor([2, 0])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_wtac_unequal_dist(self):
        d = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 1.0]])
        labels = torch.tensor([0, 1, 1])
        actual = competitions.wtac(d, labels)
        desired = torch.tensor([0, 1])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_wtac_one_hot(self):
        d = torch.tensor([[1.99, 3.01], [3.0, 2.01]])
        labels = torch.tensor([[0, 1], [1, 0]])
        actual = competitions.wtac(d, labels)
        desired = torch.tensor([[0, 1], [1, 0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_knnc_k1(self):
        d = torch.tensor([[2.0, 3.0, 1.99, 3.01], [2.0, 3.0, 2.01, 3.0]])
        labels = torch.tensor([0, 1, 2, 3])
        actual = competitions.knnc(d, labels, k=1)
        desired = torch.tensor([2, 0])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        pass


class TestPooling(unittest.TestCase):
    def setUp(self):
        pass

    def test_stratified_min(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        actual = pooling.stratified_min_pooling(d, labels)
        desired = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_min_one_hot(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        labels = torch.eye(3)[labels]
        actual = pooling.stratified_min_pooling(d, labels)
        desired = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_min_trivial(self):
        d = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0, 1]])
        labels = torch.tensor([0, 1, 2])
        actual = pooling.stratified_min_pooling(d, labels)
        desired = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_max(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0, 9.0], [9.0, 8.0, 0, 1, 7.0]])
        labels = torch.tensor([0, 0, 3, 2, 0])
        actual = pooling.stratified_max_pooling(d, labels)
        desired = torch.tensor([[9.0, 3.0, 2.0], [9.0, 1.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_max_one_hot(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0, 9.0], [9.0, 8.0, 0, 1, 7.0]])
        labels = torch.tensor([0, 0, 2, 1, 0])
        labels = torch.nn.functional.one_hot(labels, num_classes=3)
        actual = pooling.stratified_max_pooling(d, labels)
        desired = torch.tensor([[9.0, 3.0, 2.0], [9.0, 1.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_sum(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.LongTensor([0, 0, 1, 2])
        actual = pooling.stratified_sum_pooling(d, labels)
        desired = torch.tensor([[1.0, 2.0, 3.0], [17.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_sum_one_hot(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        labels = torch.eye(3)[labels]
        actual = pooling.stratified_sum_pooling(d, labels)
        desired = torch.tensor([[1.0, 2.0, 3.0], [17.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_prod(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0, 9.0], [9.0, 8.0, 0, 1, 7.0]])
        labels = torch.tensor([0, 0, 3, 2, 0])
        actual = pooling.stratified_prod_pooling(d, labels)
        desired = torch.tensor([[0.0, 3.0, 2.0], [504.0, 1.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        pass


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
                desired[i][j] = (torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )**2)
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
        actual = distances.lpnorm_distance(self.x, self.y, p=float("inf"))
        desired = torch.empty(self.nx, self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                desired[i][j] = torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=float("inf"),
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
                desired[i][j] = (torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )**2)
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
                desired[i][j] = (torch.nn.functional.pairwise_distance(
                    self.x[i].reshape(1, -1),
                    self.y[j].reshape(1, -1),
                    p=2,
                    keepdim=False,
                )**2)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=2)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x, self.y


class TestInitializers(unittest.TestCase):
    def setUp(self):
        self.flist = [
            "zeros",
            "ones",
            "rand",
            "randn",
            "stratified_mean",
            "stratified_random",
        ]
        self.x = torch.tensor(
            [[0, -1, -2], [10, 11, 12], [0, 0, 0], [2, 2, 2]],
            dtype=torch.float32)
        self.y = torch.tensor([0, 0, 1, 1])
        self.gen = torch.manual_seed(42)

    def test_registry(self):
        self.assertIsNotNone(initializers.INITIALIZERS)

    def test_funcname_deserialization(self):
        for funcname in self.flist:
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
        for funcname in ["blubb", "foobar"]:
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
        actual, _ = initializers.stratified_mean(self.x, self.y, pdist, False)
        desired = torch.tensor([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_random_equal1(self):
        pdist = torch.tensor([1, 1])
        actual, _ = initializers.stratified_random(self.x, self.y, pdist,
                                                   False)
        desired = torch.tensor([[0.0, -1.0, -2.0], [0.0, 0.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_mean_equal2(self):
        pdist = torch.tensor([2, 2])
        actual, _ = initializers.stratified_mean(self.x, self.y, pdist, False)
        desired = torch.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0],
                                [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_random_equal2(self):
        pdist = torch.tensor([2, 2])
        actual, _ = initializers.stratified_random(self.x, self.y, pdist,
                                                   False)
        desired = torch.tensor([[0.0, -1.0, -2.0], [0.0, -1.0, -2.0],
                                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_mean_unequal(self):
        pdist = torch.tensor([1, 3])
        actual, _ = initializers.stratified_mean(self.x, self.y, pdist, False)
        desired = torch.tensor([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_random_unequal(self):
        pdist = torch.tensor([1, 3])
        actual, _ = initializers.stratified_random(self.x, self.y, pdist,
                                                   False)
        desired = torch.tensor([[0.0, -1.0, -2.0], [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_mean_unequal_one_hot(self):
        pdist = torch.tensor([1, 3])
        y = torch.eye(2)[self.y]
        desired1 = torch.tensor([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0],
                                 [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        actual1, actual2 = initializers.stratified_mean(self.x, y, pdist)
        desired2 = torch.tensor([[1, 0], [0, 1], [0, 1], [0, 1]])
        mismatch = np.testing.assert_array_almost_equal(actual1,
                                                        desired1,
                                                        decimal=5)
        mismatch = np.testing.assert_array_almost_equal(actual2,
                                                        desired2,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_random_unequal_one_hot(self):
        pdist = torch.tensor([1, 3])
        y = torch.eye(2)[self.y]
        actual1, actual2 = initializers.stratified_random(self.x, y, pdist)
        desired1 = torch.tensor([[0.0, -1.0, -2.0], [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        desired2 = torch.tensor([[1, 0], [0, 1], [0, 1], [0, 1]])
        mismatch = np.testing.assert_array_almost_equal(actual1,
                                                        desired1,
                                                        decimal=5)
        mismatch = np.testing.assert_array_almost_equal(actual2,
                                                        desired2,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x, self.y, self.gen
        _ = torch.seed()


class TestLosses(unittest.TestCase):
    def setUp(self):
        pass

    def test_glvq_loss_int_labels(self):
        d = torch.stack([torch.ones(100), torch.zeros(100)], dim=1)
        labels = torch.tensor([0, 1])
        targets = torch.ones(100)
        batch_loss = losses.glvq_loss(distances=d,
                                      target_labels=targets,
                                      prototype_labels=labels)
        loss_value = torch.sum(batch_loss, dim=0)
        self.assertEqual(loss_value, -100)

    def test_glvq_loss_one_hot_labels(self):
        d = torch.stack([torch.ones(100), torch.zeros(100)], dim=1)
        labels = torch.tensor([[0, 1], [1, 0]])
        wl = torch.tensor([1, 0])
        targets = torch.stack([wl for _ in range(100)], dim=0)
        batch_loss = losses.glvq_loss(distances=d,
                                      target_labels=targets,
                                      prototype_labels=labels)
        loss_value = torch.sum(batch_loss, dim=0)
        self.assertEqual(loss_value, -100)

    def test_glvq_loss_one_hot_unequal(self):
        dlist = [torch.ones(100), torch.zeros(100), torch.zeros(100)]
        d = torch.stack(dlist, dim=1)
        labels = torch.tensor([[0, 1], [1, 0], [1, 0]])
        wl = torch.tensor([1, 0])
        targets = torch.stack([wl for _ in range(100)], dim=0)
        batch_loss = losses.glvq_loss(distances=d,
                                      target_labels=targets,
                                      prototype_labels=labels)
        loss_value = torch.sum(batch_loss, dim=0)
        self.assertEqual(loss_value, -100)

    def tearDown(self):
        pass
