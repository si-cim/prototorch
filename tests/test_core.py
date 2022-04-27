"""ProtoTorch core test suite"""

import unittest

import numpy as np
import pytest
import torch

import prototorch as pt
from prototorch.utils import parse_distribution


# Utils
def test_parse_distribution_dict_0():
    distribution = {"num_classes": 1, "per_class": 0}
    distribution = parse_distribution(distribution)
    assert distribution == {0: 0}


def test_parse_distribution_dict_1():
    distribution = dict(num_classes=3, per_class=2)
    distribution = parse_distribution(distribution)
    assert distribution == {0: 2, 1: 2, 2: 2}


def test_parse_distribution_dict_2():
    distribution = {0: 1, 2: 2, -1: 3}
    distribution = parse_distribution(distribution)
    assert distribution == {0: 1, 2: 2, -1: 3}


def test_parse_distribution_tuple():
    distribution = (2, 3)
    distribution = parse_distribution(distribution)
    assert distribution == {0: 3, 1: 3}


def test_parse_distribution_list():
    distribution = [1, 1, 0, 2]
    distribution = parse_distribution(distribution)
    assert distribution == {0: 1, 1: 1, 2: 0, 3: 2}


def test_parse_distribution_custom_labels():
    distribution = [1, 1, 0, 2]
    clabels = [1, 2, 5, 3]
    distribution = parse_distribution(distribution, clabels)
    assert distribution == {1: 1, 2: 1, 5: 0, 3: 2}


# Components initializers
def test_literal_comp_generate():
    protos = torch.rand(4, 3, 5, 5)
    c = pt.initializers.LiteralCompInitializer(protos)
    components = c.generate([])
    assert torch.allclose(components, protos)


def test_literal_comp_generate_from_list():
    protos = [[0, 1], [2, 3], [4, 5]]
    c = pt.initializers.LiteralCompInitializer(protos)
    with pytest.warns(UserWarning):
        components = c.generate([])
    assert torch.allclose(components, torch.Tensor(protos))


def test_shape_aware_raises_error():
    with pytest.raises(TypeError):
        _ = pt.initializers.ShapeAwareCompInitializer(shape=(2, ))


def test_data_aware_comp_generate():
    protos = torch.rand(4, 3, 5, 5)
    c = pt.initializers.DataAwareCompInitializer(protos)
    components = c.generate(num_components="IgnoreMe!")
    assert torch.allclose(components, protos)


def test_class_aware_comp_generate():
    protos = torch.rand(4, 2, 3, 5, 5)
    plabels = torch.tensor([0, 0, 1, 1]).long()
    c = pt.initializers.ClassAwareCompInitializer([protos, plabels])
    components = c.generate(distribution=[])
    assert torch.allclose(components, protos)


def test_zeros_comp_generate():
    shape = (3, 5, 5)
    c = pt.initializers.ZerosCompInitializer(shape)
    components = c.generate(num_components=4)
    assert torch.allclose(components, torch.zeros(4, 3, 5, 5))


def test_ones_comp_generate():
    c = pt.initializers.OnesCompInitializer(2)
    components = c.generate(num_components=3)
    assert torch.allclose(components, torch.ones(3, 2))


def test_fill_value_comp_generate():
    c = pt.initializers.FillValueCompInitializer(2, 0.0)
    components = c.generate(num_components=3)
    assert torch.allclose(components, torch.zeros(3, 2))


def test_uniform_comp_generate_min_max_bound():
    c = pt.initializers.UniformCompInitializer(2, -1.0, 1.0)
    components = c.generate(num_components=1024)
    assert components.min() >= -1.0
    assert components.max() <= 1.0


def test_random_comp_generate_mean():
    c = pt.initializers.RandomNormalCompInitializer(2, -1.0)
    components = c.generate(num_components=1024)
    assert torch.allclose(components.mean(),
                          torch.tensor(-1.0),
                          rtol=1e-05,
                          atol=1e-01)


def test_comp_generate_0_components():
    c = pt.initializers.ZerosCompInitializer(2)
    _ = c.generate(num_components=0)


def test_stratified_mean_comp_generate():
    # yapf: disable
    x = torch.Tensor(
        [[0,  -1, -2],
         [10, 11, 12],
         [0,   0,  0],
         [2,   2,  2]])
    y = torch.LongTensor([0, 0, 1, 1])
    desired = torch.Tensor(
        [[5.0, 5.0, 5.0],
         [1.0, 1.0, 1.0]])
    # yapf: enable
    c = pt.initializers.StratifiedMeanCompInitializer(data=[x, y])
    actual = c.generate([1, 1])
    assert torch.allclose(actual, desired)


def test_stratified_selection_comp_generate():
    # yapf: disable
    x = torch.Tensor(
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0],
         [1, 1, 1]])
    y = torch.LongTensor([0, 1, 0, 1])
    desired = torch.Tensor(
        [[0, 0, 0],
         [1, 1, 1]])
    # yapf: enable
    c = pt.initializers.StratifiedSelectionCompInitializer(data=[x, y])
    actual = c.generate([1, 1])
    assert torch.allclose(actual, desired)


# Labels initializers
def test_literal_labels_init():
    l = pt.initializers.LiteralLabelsInitializer([0, 0, 1, 2])
    with pytest.warns(UserWarning):
        labels = l.generate([])
    assert torch.allclose(labels, torch.LongTensor([0, 0, 1, 2]))


def test_labels_init_from_list():
    l = pt.initializers.LabelsInitializer()
    components = l.generate(distribution=[1, 1, 1])
    assert torch.allclose(components, torch.LongTensor([0, 1, 2]))


def test_labels_init_from_tuple_legal():
    l = pt.initializers.LabelsInitializer()
    components = l.generate(distribution=(3, 1))
    assert torch.allclose(components, torch.LongTensor([0, 1, 2]))


def test_labels_init_from_tuple_illegal():
    l = pt.initializers.LabelsInitializer()
    with pytest.raises(AssertionError):
        _ = l.generate(distribution=(1, 1, 1))


def test_data_aware_labels_init():
    data, targets = [0, 1, 2, 3], [0, 0, 1, 1]
    ds = pt.datasets.NumpyDataset(data, targets)
    l = pt.initializers.DataAwareLabelsInitializer(ds)
    labels = l.generate([])
    assert torch.allclose(labels, torch.LongTensor(targets))


# Reasonings initializers
def test_literal_reasonings_init():
    r = pt.initializers.LiteralReasoningsInitializer([0, 0, 1, 2])
    with pytest.warns(UserWarning):
        reasonings = r.generate([])
    assert torch.allclose(reasonings, torch.Tensor([0, 0, 1, 2]))


def test_random_reasonings_init():
    r = pt.initializers.RandomReasoningsInitializer(0.2, 0.8)
    reasonings = r.generate(distribution=[0, 1])
    assert torch.numel(reasonings) == 1 * 2 * 2
    assert reasonings.min() >= 0.2
    assert reasonings.max() <= 0.8


def test_zeros_reasonings_init():
    r = pt.initializers.ZerosReasoningsInitializer()
    reasonings = r.generate(distribution=[0, 1])
    assert torch.allclose(reasonings, torch.zeros(1, 2, 2))


def test_ones_reasonings_init():
    r = pt.initializers.ZerosReasoningsInitializer()
    reasonings = r.generate(distribution=[1, 2, 3])
    assert torch.allclose(reasonings, torch.zeros(6, 3, 2))


def test_pure_positive_reasonings_init_one_per_class():
    r = pt.initializers.PurePositiveReasoningsInitializer(
        components_first=False)
    reasonings = r.generate(distribution=(4, 1))
    assert torch.allclose(reasonings[0], torch.eye(4))


def test_pure_positive_reasonings_init_unrepresented_classes():
    r = pt.initializers.PurePositiveReasoningsInitializer()
    reasonings = r.generate(distribution=[9, 0, 0, 0])
    assert reasonings.shape[0] == 9
    assert reasonings.shape[1] == 4
    assert reasonings.shape[2] == 2


def test_random_reasonings_init_channels_not_first():
    r = pt.initializers.RandomReasoningsInitializer(components_first=False)
    reasonings = r.generate(distribution=[0, 0, 0, 1])
    assert reasonings.shape[0] == 2
    assert reasonings.shape[1] == 4
    assert reasonings.shape[2] == 1


# Transform initializers
def test_eye_transform_init_square():
    t = pt.initializers.EyeLinearTransformInitializer()
    I = t.generate(3, 3)
    assert torch.allclose(I, torch.eye(3))


def test_eye_transform_init_narrow():
    t = pt.initializers.EyeLinearTransformInitializer()
    actual = t.generate(3, 2)
    desired = torch.Tensor([[1, 0], [0, 1], [0, 0]])
    assert torch.allclose(actual, desired)


def test_eye_transform_init_wide():
    t = pt.initializers.EyeLinearTransformInitializer()
    actual = t.generate(2, 3)
    desired = torch.Tensor([[1, 0, 0], [0, 1, 0]])
    assert torch.allclose(actual, desired)


# Transforms
def test_linear_transform_default_eye_init():
    l = pt.transforms.LinearTransform(2, 4)
    actual = l.weights
    desired = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert torch.allclose(actual, desired)


def test_linear_transform_forward():
    l = pt.transforms.LinearTransform(4, 2)
    actual_weights = l.weights
    desired_weights = torch.Tensor([[1, 0], [0, 1], [0, 0], [0, 0]])
    assert torch.allclose(actual_weights, desired_weights)
    actual_outputs = l(torch.Tensor([[1.1, 2.2, 3.3, 4.4], \
                                     [1.1, 2.2, 3.3, 4.4], \
                                     [5.5, 6.6, 7.7, 8.8]]))
    desired_outputs = torch.Tensor([[1.1, 2.2], [1.1, 2.2], [5.5, 6.6]])
    assert torch.allclose(actual_outputs, desired_outputs)


def test_linear_transform_zeros_init():
    l = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pt.initializers.ZerosLinearTransformInitializer(),
    )
    actual = l.weights
    desired = torch.zeros(2, 4)
    assert torch.allclose(actual, desired)


def test_linear_transform_out_dim_first():
    l = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pt.initializers.OLTI(out_dim_first=True),
    )
    assert l.weights.shape[0] == 4
    assert l.weights.shape[1] == 2


# Components
def test_components_no_initializer():
    with pytest.raises(TypeError):
        _ = pt.components.Components(3, None)


def test_components_no_num_components():
    with pytest.raises(TypeError):
        _ = pt.components.Components(initializer=pt.initializers.OCI(2))


def test_components_none_num_components():
    with pytest.raises(TypeError):
        _ = pt.components.Components(None, initializer=pt.initializers.OCI(2))


def test_components_no_args():
    with pytest.raises(TypeError):
        _ = pt.components.Components()


def test_components_zeros_init():
    c = pt.components.Components(3, pt.initializers.ZCI(2))
    assert torch.allclose(c.components, torch.zeros(3, 2))


def test_labeled_components_dict_init():
    c = pt.components.LabeledComponents({0: 3}, pt.initializers.OCI(2))
    assert torch.allclose(c.components, torch.ones(3, 2))
    assert torch.allclose(c.labels, torch.zeros(3, dtype=torch.long))


def test_labeled_components_list_init():
    c = pt.components.LabeledComponents([3], pt.initializers.OCI(2))
    assert torch.allclose(c.components, torch.ones(3, 2))
    assert torch.allclose(c.labels, torch.zeros(3, dtype=torch.long))


def test_labeled_components_tuple_init():
    c = pt.components.LabeledComponents({0: 1, 1: 2}, pt.initializers.OCI(2))
    assert torch.allclose(c.components, torch.ones(3, 2))
    assert torch.allclose(c.labels, torch.LongTensor([0, 1, 1]))


# Labels
def test_standalone_labels_dict_init():
    l = pt.components.Labels({0: 3})
    assert torch.allclose(l.labels, torch.zeros(3, dtype=torch.long))


def test_standalone_labels_list_init():
    l = pt.components.Labels([3])
    assert torch.allclose(l.labels, torch.zeros(3, dtype=torch.long))


def test_standalone_labels_tuple_init():
    l = pt.components.Labels({0: 1, 1: 2})
    assert torch.allclose(l.labels, torch.LongTensor([0, 1, 1]))


# Losses
def test_glvq_loss_int_labels():
    d = torch.stack([torch.ones(100), torch.zeros(100)], dim=1)
    labels = torch.tensor([0, 1])
    targets = torch.ones(100)
    batch_loss = pt.losses.glvq_loss(distances=d,
                                     target_labels=targets,
                                     prototype_labels=labels)
    loss_value = torch.sum(batch_loss, dim=0)
    assert loss_value == -100


def test_glvq_loss_one_hot_labels():
    d = torch.stack([torch.ones(100), torch.zeros(100)], dim=1)
    labels = torch.tensor([[0, 1], [1, 0]])
    wl = torch.tensor([1, 0])
    targets = torch.stack([wl for _ in range(100)], dim=0)
    batch_loss = pt.losses.glvq_loss(distances=d,
                                     target_labels=targets,
                                     prototype_labels=labels)
    loss_value = torch.sum(batch_loss, dim=0)
    assert loss_value == -100


def test_glvq_loss_one_hot_unequal():
    dlist = [torch.ones(100), torch.zeros(100), torch.zeros(100)]
    d = torch.stack(dlist, dim=1)
    labels = torch.tensor([[0, 1], [1, 0], [1, 0]])
    wl = torch.tensor([1, 0])
    targets = torch.stack([wl for _ in range(100)], dim=0)
    batch_loss = pt.losses.glvq_loss(distances=d,
                                     target_labels=targets,
                                     prototype_labels=labels)
    loss_value = torch.sum(batch_loss, dim=0)
    assert loss_value == -100


# Activations
class TestActivations(unittest.TestCase):

    def setUp(self):
        self.flist = ["identity", "sigmoid_beta", "swish_beta"]
        self.x = torch.randn(1024, 1)

    def test_registry(self):
        self.assertIsNotNone(pt.nn.ACTIVATIONS)

    def test_funcname_deserialization(self):
        for funcname in self.flist:
            f = pt.nn.get_activation(funcname)
            iscallable = callable(f)
            self.assertTrue(iscallable)

    def test_callable_deserialization(self):

        def dummy(x, **kwargs):
            return x

        for f in [dummy, lambda x: x]:
            f = pt.nn.get_activation(f)
            iscallable = callable(f)
            self.assertTrue(iscallable)
            self.assertEqual(1, f(1))

    def test_unknown_deserialization(self):
        for funcname in ["blubb", "foobar"]:
            with self.assertRaises(NameError):
                _ = pt.nn.get_activation(funcname)

    def test_identity(self):
        actual = pt.nn.identity(self.x)
        desired = self.x
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_sigmoid_beta1(self):
        actual = pt.nn.sigmoid_beta(self.x, beta=1.0)
        desired = torch.sigmoid(self.x)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_swish_beta1(self):
        actual = pt.nn.swish_beta(self.x, beta=1.0)
        desired = self.x * torch.sigmoid(self.x)
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        del self.x


# Competitions
class TestCompetitions(unittest.TestCase):

    def setUp(self):
        pass

    def test_wtac(self):
        d = torch.tensor([[2.0, 3.0, 1.99, 3.01], [2.0, 3.0, 2.01, 3.0]])
        labels = torch.tensor([0, 1, 2, 3])
        competition_layer = pt.competitions.WTAC()
        actual = competition_layer(d, labels)
        desired = torch.tensor([2, 0])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_wtac_unequal_dist(self):
        d = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 1.0]])
        labels = torch.tensor([0, 1, 1])
        competition_layer = pt.competitions.WTAC()
        actual = competition_layer(d, labels)
        desired = torch.tensor([0, 1])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_wtac_one_hot(self):
        d = torch.tensor([[1.99, 3.01], [3.0, 2.01]])
        labels = torch.tensor([[0, 1], [1, 0]])
        competition_layer = pt.competitions.WTAC()
        actual = competition_layer(d, labels)
        desired = torch.tensor([[0, 1], [1, 0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_knnc_k1(self):
        d = torch.tensor([[2.0, 3.0, 1.99, 3.01], [2.0, 3.0, 2.01, 3.0]])
        labels = torch.tensor([0, 1, 2, 3])
        competition_layer = pt.competitions.KNNC(k=1)
        actual = competition_layer(d, labels)
        desired = torch.tensor([2, 0])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        pass


# Pooling
class TestPooling(unittest.TestCase):

    def setUp(self):
        pass

    def test_stratified_min(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        pooling_layer = pt.pooling.StratifiedMinPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_min_one_hot(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        labels = torch.eye(3)[labels]
        pooling_layer = pt.pooling.StratifiedMinPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_min_trivial(self):
        d = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0, 1]])
        labels = torch.tensor([0, 1, 2])
        pooling_layer = pt.pooling.StratifiedMinPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[0.0, 2.0, 3.0], [8.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_max(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0, 9.0], [9.0, 8.0, 0, 1, 7.0]])
        labels = torch.tensor([0, 0, 3, 2, 0])
        pooling_layer = pt.pooling.StratifiedMaxPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[9.0, 3.0, 2.0], [9.0, 1.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_max_one_hot(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0, 9.0], [9.0, 8.0, 0, 1, 7.0]])
        labels = torch.tensor([0, 0, 2, 1, 0])
        labels = torch.nn.functional.one_hot(labels, num_classes=3)
        pooling_layer = pt.pooling.StratifiedMaxPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[9.0, 3.0, 2.0], [9.0, 1.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_sum(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.LongTensor([0, 0, 1, 2])
        pooling_layer = pt.pooling.StratifiedSumPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[1.0, 2.0, 3.0], [17.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_sum_one_hot(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0], [9.0, 8.0, 0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        labels = torch.eye(3)[labels]
        pooling_layer = pt.pooling.StratifiedSumPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[1.0, 2.0, 3.0], [17.0, 0.0, 1.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def test_stratified_prod(self):
        d = torch.tensor([[1.0, 0.0, 2.0, 3.0, 9.0], [9.0, 8.0, 0, 1, 7.0]])
        labels = torch.tensor([0, 0, 3, 2, 0])
        pooling_layer = pt.pooling.StratifiedProdPooling()
        actual = pooling_layer(d, labels)
        desired = torch.tensor([[0.0, 3.0, 2.0], [504.0, 1.0, 0.0]])
        mismatch = np.testing.assert_array_almost_equal(actual,
                                                        desired,
                                                        decimal=5)
        self.assertIsNone(mismatch)

    def tearDown(self):
        pass


# Distances
class TestDistances(unittest.TestCase):

    def setUp(self):
        self.nx, self.mx = 32, 2048
        self.ny, self.my = 8, 2048
        self.x = torch.randn(self.nx, self.mx)
        self.y = torch.randn(self.ny, self.my)

    def test_manhattan(self):
        actual = pt.distances.lpnorm_distance(self.x, self.y, p=1)
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
        actual = pt.distances.euclidean_distance(self.x, self.y)
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
        actual = pt.distances.squared_euclidean_distance(self.x, self.y)
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
        actual = pt.distances.lpnorm_distance(self.x, self.y, p=0)
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
        actual = pt.distances.lpnorm_distance(self.x, self.y, p=2)
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
        actual = pt.distances.lpnorm_distance(self.x, self.y, p=3)
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
        actual = pt.distances.lpnorm_distance(self.x, self.y, p=float("inf"))
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
        actual = pt.distances.omega_distance(self.x, self.y, omega=omega)
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
        actual = pt.distances.lomega_distance(self.x, self.y, omegas=omegas)
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
