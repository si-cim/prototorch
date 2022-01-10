"""ProtoTorch datasets test suite"""

import os
import shutil
import unittest

import numpy as np
import torch

import prototorch as pt
from prototorch.datasets.abstract import Dataset, ProtoDataset


class TestAbstract(unittest.TestCase):

    def setUp(self):
        self.ds = Dataset("./artifacts")

    def test_getitem(self):
        with self.assertRaises(NotImplementedError):
            _ = self.ds[0]

    def test_len(self):
        with self.assertRaises(NotImplementedError):
            _ = len(self.ds)

    def tearDown(self):
        del self.ds


class TestProtoDataset(unittest.TestCase):

    def test_download(self):
        with self.assertRaises(NotImplementedError):
            _ = ProtoDataset("./artifacts", download=True)

    def test_exists(self):
        with self.assertRaises(RuntimeError):
            _ = ProtoDataset("./artifacts", download=False)


class TestNumpyDataset(unittest.TestCase):

    def test_list_init(self):
        ds = pt.datasets.NumpyDataset([1], [1])
        self.assertEqual(len(ds), 1)

    def test_numpy_init(self):
        data = np.random.randn(3, 2)
        targets = np.array([0, 1, 2])
        ds = pt.datasets.NumpyDataset(data, targets)
        self.assertEqual(len(ds), 3)


class TestCSVDataset(unittest.TestCase):

    def setUp(self):
        data = np.random.rand(100, 4)
        targets = np.random.randint(2, size=(100, 1))
        arr = np.hstack([data, targets])
        if not os.path.exists("./artifacts"):
            os.mkdir("./artifacts")
        np.savetxt("./artifacts/test.csv", arr, delimiter=",")

    def test_len(self):
        ds = pt.datasets.CSVDataset("./artifacts/test.csv")
        self.assertEqual(len(ds), 100)

    def tearDown(self):
        os.remove("./artifacts/test.csv")


class TestSpiral(unittest.TestCase):

    def test_init(self):
        ds = pt.datasets.Spiral(num_samples=10)
        self.assertEqual(len(ds), 10)


class TestIris(unittest.TestCase):

    def setUp(self):
        self.ds = pt.datasets.Iris()

    def test_size(self):
        self.assertEqual(len(self.ds), 150)

    def test_dims(self):
        self.assertEqual(self.ds.data.shape[1], 4)

    def test_dims_selection(self):
        ds = pt.datasets.Iris(dims=[0, 1])
        self.assertEqual(ds.data.shape[1], 2)


class TestBlobs(unittest.TestCase):

    def test_size(self):
        ds = pt.datasets.Blobs(num_samples=10)
        self.assertEqual(len(ds), 10)


class TestRandom(unittest.TestCase):

    def test_size(self):
        ds = pt.datasets.Random(num_samples=10)
        self.assertEqual(len(ds), 10)


class TestCircles(unittest.TestCase):

    def test_size(self):
        ds = pt.datasets.Circles(num_samples=10)
        self.assertEqual(len(ds), 10)


class TestMoons(unittest.TestCase):

    def test_size(self):
        ds = pt.datasets.Moons(num_samples=10)
        self.assertEqual(len(ds), 10)


# class TestTecator(unittest.TestCase):
#     def setUp(self):
#         self.artifacts_dir = "./artifacts/Tecator"
#         self._remove_artifacts()

#     def _remove_artifacts(self):
#         if os.path.exists(self.artifacts_dir):
#             shutil.rmtree(self.artifacts_dir)

#     def test_download_false(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         self._remove_artifacts()
#         with self.assertRaises(RuntimeError):
#             _ = pt.datasets.Tecator(rootdir, download=False)

#     def test_download_caching(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         _ = pt.datasets.Tecator(rootdir, download=True, verbose=False)
#         _ = pt.datasets.Tecator(rootdir, download=False, verbose=False)

#     def test_repr(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         train = pt.datasets.Tecator(rootdir, download=True, verbose=True)
#         self.assertTrue("Split: Train" in train.__repr__())

#     def test_download_train(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         train = pt.datasets.Tecator(root=rootdir,
#                                     train=True,
#                                     download=True,
#                                     verbose=False)
#         train = pt.datasets.Tecator(root=rootdir, download=True, verbose=False)
#         x_train, y_train = train.data, train.targets
#         self.assertEqual(x_train.shape[0], 144)
#         self.assertEqual(y_train.shape[0], 144)
#         self.assertEqual(x_train.shape[1], 100)

#     def test_download_test(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         test = pt.datasets.Tecator(root=rootdir, train=False, verbose=False)
#         x_test, y_test = test.data, test.targets
#         self.assertEqual(x_test.shape[0], 71)
#         self.assertEqual(y_test.shape[0], 71)
#         self.assertEqual(x_test.shape[1], 100)

#     def test_class_to_idx(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         test = pt.datasets.Tecator(root=rootdir, train=False, verbose=False)
#         _ = test.class_to_idx

#     def test_getitem(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         test = pt.datasets.Tecator(root=rootdir, train=False, verbose=False)
#         x, y = test[0]
#         self.assertEqual(x.shape[0], 100)
#         self.assertIsInstance(y, int)

#     def test_loadable_with_dataloader(self):
#         rootdir = self.artifacts_dir.rpartition("/")[0]
#         test = pt.datasets.Tecator(root=rootdir, train=False, verbose=False)
#         _ = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

#     def tearDown(self):
#         self._remove_artifacts()
