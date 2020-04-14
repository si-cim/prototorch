"""ProtoTorch datasets test suite."""

import os
import shutil
import unittest

import torch

from prototorch.datasets import abstract, tecator


class TestAbstract(unittest.TestCase):
    def test_getitem(self):
        with self.assertRaises(NotImplementedError):
            abstract.Dataset('./artifacts')[0]

    def test_len(self):
        with self.assertRaises(NotImplementedError):
            len(abstract.Dataset('./artifacts'))


class TestProtoDataset(unittest.TestCase):
    def test_getitem(self):
        with self.assertRaises(NotImplementedError):
            abstract.ProtoDataset('./artifacts')[0]

    def test_download(self):
        with self.assertRaises(NotImplementedError):
            abstract.ProtoDataset('./artifacts').download()


class TestTecator(unittest.TestCase):
    def setUp(self):
        self.artifacts_dir = './artifacts/Tecator'
        self._remove_artifacts()

    def _remove_artifacts(self):
        if os.path.exists(self.artifacts_dir):
            shutil.rmtree(self.artifacts_dir)

    def test_download_false(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        self._remove_artifacts()
        with self.assertRaises(RuntimeError):
            _ = tecator.Tecator(rootdir, download=False)

    def test_download_caching(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        _ = tecator.Tecator(rootdir, download=True, verbose=False)
        _ = tecator.Tecator(rootdir, download=False, verbose=False)

    def test_repr(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        train = tecator.Tecator(rootdir, download=True, verbose=True)
        self.assertTrue('Split: Train' in train.__repr__())

    def test_download_train(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        train = tecator.Tecator(root=rootdir,
                                train=True,
                                download=True,
                                verbose=False)
        train = tecator.Tecator(root=rootdir, download=True, verbose=False)
        x_train, y_train = train.data, train.targets
        self.assertEqual(x_train.shape[0], 144)
        self.assertEqual(y_train.shape[0], 144)
        self.assertEqual(x_train.shape[1], 100)

    def test_download_test(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        test = tecator.Tecator(root=rootdir, train=False, verbose=False)
        x_test, y_test = test.data, test.targets
        self.assertEqual(x_test.shape[0], 71)
        self.assertEqual(y_test.shape[0], 71)
        self.assertEqual(x_test.shape[1], 100)

    def test_class_to_idx(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        test = tecator.Tecator(root=rootdir, train=False, verbose=False)
        _ = test.class_to_idx

    def test_getitem(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        test = tecator.Tecator(root=rootdir, train=False, verbose=False)
        x, y = test[0]
        self.assertEqual(x.shape[0], 100)
        self.assertIsInstance(y, int)

    def test_loadable_with_dataloader(self):
        rootdir = self.artifacts_dir.rpartition('/')[0]
        test = tecator.Tecator(root=rootdir, train=False, verbose=False)
        _ = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    def tearDown(self):
        pass
