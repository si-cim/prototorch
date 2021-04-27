"""ProtoTorch kernels test suite."""

import unittest

import numpy as np
import torch

from prototorch.functions.distances import KernelDistance
from prototorch.functions.kernels import ExplicitKernel


class TestExplicitKernel(unittest.TestCase):
    def setUp(self):
        self.single_x = torch.randn(1024)
        self.single_y = torch.randn(1024)

        self.batch_x = torch.randn(32, 1024)
        self.batch_y = torch.randn(32, 1024)

        self.kernel = ExplicitKernel()

    def test_single_values(self):
        kernel = ExplicitKernel()
        self.assertEqual(
            kernel(self.single_x, self.single_y).shape, torch.Size([]))

    def test_single_batch(self):
        kernel = ExplicitKernel()
        self.assertEqual(
            kernel(self.single_x, self.batch_y).shape, torch.Size([32]))

    def test_batch_single(self):
        kernel = ExplicitKernel()
        self.assertEqual(
            kernel(self.batch_x, self.single_y).shape, torch.Size([32]))

    def test_batch_values(self):
        kernel = ExplicitKernel()
        self.assertEqual(
            kernel(self.batch_x, self.batch_y).shape, torch.Size([32, 32]))


class TestKernelDistance(unittest.TestCase):
    def setUp(self):
        self.single_x = torch.randn(1024)
        self.single_y = torch.randn(1024)

        self.batch_x = torch.randn(32, 1024)
        self.batch_y = torch.randn(32, 1024)

        self.kernel = ExplicitKernel()

    def test_single_values(self):
        distance = KernelDistance(self.kernel)
        self.assertEqual(
            distance(self.single_x, self.single_y).shape, torch.Size([]))

    def test_single_batch(self):
        distance = KernelDistance(self.kernel)
        self.assertEqual(
            distance(self.single_x, self.batch_y).shape, torch.Size([32]))

    def test_batch_single(self):
        distance = KernelDistance(self.kernel)
        self.assertEqual(
            distance(self.batch_x, self.single_y).shape, torch.Size([32]))

    def test_batch_values(self):
        distance = KernelDistance(self.kernel)
        self.assertEqual(
            distance(self.batch_x, self.batch_y).shape, torch.Size([32, 32]))