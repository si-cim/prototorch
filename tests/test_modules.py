"""ProtoTorch modules test suite."""

import unittest

import numpy as np
import torch

from prototorch.modules import losses, prototypes


class TestPrototypes(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[0, -1, -2], [10, 11, 12], [0, 0, 0], [2, 2, 2]],
            dtype=torch.float32)
        self.y = torch.tensor([0, 0, 1, 1])
        self.gen = torch.manual_seed(42)

    def test_prototypes1d_init_without_input_dim(self):
        with self.assertRaises(NameError):
            _ = prototypes.Prototypes1D(nclasses=2)

    def test_prototypes1d_init_without_nclasses(self):
        with self.assertRaises(NameError):
            _ = prototypes.Prototypes1D(input_dim=1)

    def test_prototypes1d_init_with_nclasses_1(self):
        with self.assertWarns(UserWarning):
            _ = prototypes.Prototypes1D(nclasses=1, input_dim=1)

    def test_prototypes1d_init_without_pdist(self):
        p1 = prototypes.Prototypes1D(input_dim=6,
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

    def test_prototypes1d_init_without_data(self):
        pdist = [2, 2]
        p1 = prototypes.Prototypes1D(input_dim=3,
                                     prototype_distribution=pdist,
                                     prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(4, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_prototypes1d_proto_init_without_data(self):
        with self.assertWarns(UserWarning):
            _ = prototypes.Prototypes1D(
                input_dim=3,
                nclasses=2,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=None)

    def test_prototypes1d_init_torch_pdist(self):
        pdist = torch.tensor([2, 2])
        p1 = prototypes.Prototypes1D(input_dim=3,
                                     prototype_distribution=pdist,
                                     prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(4, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_prototypes1d_init_without_inputdim_with_data(self):
        _ = prototypes.Prototypes1D(nclasses=2,
                                    prototypes_per_class=1,
                                    prototype_initializer='stratified_mean',
                                    data=[[[1.], [0.]], [1, 0]])

    def test_prototypes1d_init_with_int_data(self):
        _ = prototypes.Prototypes1D(nclasses=2,
                                    prototypes_per_class=1,
                                    prototype_initializer='stratified_mean',
                                    data=[[[1], [0]], [1, 0]])

    def test_prototypes1d_init_one_hot_without_data(self):
        _ = prototypes.Prototypes1D(input_dim=1,
                                    nclasses=2,
                                    prototypes_per_class=1,
                                    prototype_initializer='stratified_mean',
                                    data=None,
                                    one_hot_labels=True)

    def test_prototypes1d_init_one_hot_labels_false(self):
        """Test if ValueError is raised when `one_hot_labels` is set to `False`
        but the provided `data` has one-hot encoded labels.
        """
        with self.assertRaises(ValueError):
            _ = prototypes.Prototypes1D(
                input_dim=1,
                nclasses=2,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=([[0.], [1.]], [[0, 1], [1, 0]]),
                one_hot_labels=False)

    def test_prototypes1d_init_1d_y_data_one_hot_labels_true(self):
        """Test if ValueError is raised when `one_hot_labels` is set to `True`
        but the provided `data` does not contain one-hot encoded labels.
        """
        with self.assertRaises(ValueError):
            _ = prototypes.Prototypes1D(
                input_dim=1,
                nclasses=2,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=([[0.], [1.]], [0, 1]),
                one_hot_labels=True)

    def test_prototypes1d_init_one_hot_labels_true(self):
        """Test if ValueError is raised when `one_hot_labels` is set to `True`
        but the provided `data` contains 2D targets but
        does not contain one-hot encoded labels.
        """
        with self.assertRaises(ValueError):
            _ = prototypes.Prototypes1D(
                input_dim=1,
                nclasses=2,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=([[0.], [1.]], [[0], [1]]),
                one_hot_labels=True)

    def test_prototypes1d_init_with_int_dtype(self):
        with self.assertRaises(RuntimeError):
            _ = prototypes.Prototypes1D(
                nclasses=2,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=[[[1], [0]], [1, 0]],
                dtype=torch.int32)

    def test_prototypes1d_inputndim_with_data(self):
        with self.assertRaises(ValueError):
            _ = prototypes.Prototypes1D(input_dim=1,
                                        nclasses=1,
                                        prototypes_per_class=1,
                                        data=[[1.], [1]])

    def test_prototypes1d_inputdim_with_data(self):
        with self.assertRaises(ValueError):
            _ = prototypes.Prototypes1D(
                input_dim=2,
                nclasses=2,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=[[[1.], [0.]], [1, 0]])

    def test_prototypes1d_nclasses_with_data(self):
        """Test ValueError raise if provided `nclasses` is not the same
        as the one computed from the provided `data`.
        """
        with self.assertRaises(ValueError):
            _ = prototypes.Prototypes1D(
                input_dim=1,
                nclasses=1,
                prototypes_per_class=1,
                prototype_initializer='stratified_mean',
                data=[[[1.], [2.]], [1, 2]])

    def test_prototypes1d_init_with_ppc(self):
        p1 = prototypes.Prototypes1D(data=[self.x, self.y],
                                     prototypes_per_class=2,
                                     prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(4, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_prototypes1d_init_with_pdist(self):
        p1 = prototypes.Prototypes1D(data=[self.x, self.y],
                                     prototype_distribution=[6, 9],
                                     prototype_initializer='zeros')
        protos = p1.prototypes
        actual = protos.detach().numpy()
        desired = torch.zeros(15, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_prototypes1d_func_initializer(self):
        def my_initializer(*args, **kwargs):
            return torch.full((2, 99), 99), torch.tensor([0, 1])

        p1 = prototypes.Prototypes1D(input_dim=99,
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

    def test_prototypes1d_forward(self):
        p1 = prototypes.Prototypes1D(data=[self.x, self.y])
        protos, _ = p1()
        actual = protos.detach().numpy()
        desired = torch.ones(2, 3)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_prototypes1d_dist_validate(self):
        p1 = prototypes.Prototypes1D(input_dim=0, prototype_distribution=[0])
        with self.assertWarns(UserWarning):
            _ = p1._validate_prototype_distribution()

    def test_prototypes1d_validate_extra_repr_not_empty(self):
        p1 = prototypes.Prototypes1D(input_dim=0, prototype_distribution=[0])
        rep = p1.extra_repr()
        self.assertNotEqual(rep, '')

    def tearDown(self):
        del self.x, self.y, self.gen
        _ = torch.seed()


class TestLosses(unittest.TestCase):
    def setUp(self):
        pass

    def test_glvqloss_init(self):
        _ = losses.GLVQLoss(0, 'swish_beta', beta=20)

    def test_glvqloss_forward(self):
        criterion = losses.GLVQLoss(margin=0,
                                    squashing='sigmoid_beta',
                                    beta=100)
        d = torch.stack([torch.ones(100), torch.zeros(100)], dim=1)
        labels = torch.tensor([0, 1])
        targets = torch.ones(100)
        outputs = [d, labels]
        loss = criterion(outputs, targets)
        loss_value = loss.item()
        self.assertAlmostEqual(loss_value, 0.0)

    def tearDown(self):
        pass
