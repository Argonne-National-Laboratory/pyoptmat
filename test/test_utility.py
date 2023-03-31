from pyoptmat import utility

import torch

import unittest
import numpy as np
import numpy.random as ra
import scipy.interpolate as inter

torch.set_default_tensor_type(torch.DoubleTensor)


class TestArbitraryBatchTimeSeriesInterpolator(unittest.TestCase):
    def setUp(self):
        self.ntime = 100
        self.nbatch = 50
        self.times = np.empty((self.ntime, self.nbatch))
        self.values = np.empty((self.ntime, self.nbatch))
        for i in range(self.nbatch):
            tmax = ra.random()
            self.times[:, i] = np.linspace(0, tmax, self.ntime)
            self.values[:, i] = ra.random((self.ntime,))

        self.ref_obj = utility.BatchTimeSeriesInterpolator(
            torch.tensor(self.times), torch.tensor(self.values)
        )
        self.obj = utility.ArbitraryBatchTimeSeriesInterpolator(
            torch.tensor(self.times), torch.tensor(self.values)
        )

    def test_interpolate(self):
        y = 10
        X = torch.zeros((y, self.nbatch))
        for i in range(self.nbatch):
            tmin, tmax = sorted(
                list(ra.uniform(0, np.max(self.times[:, i]), size=(2,)))
            )
            X[:, i] = torch.linspace(tmin, tmax, y)

        # Answer done one at a time
        Y1 = torch.zeros((y, self.nbatch))
        for i in range(y):
            Y1[i] = self.ref_obj(X[i])

        # With the new object
        Y2 = self.obj(X)

        print(Y2.shape)

        self.assertTrue(torch.allclose(Y1, Y2))


class TestInterpolateBatchTimesObject(unittest.TestCase):
    def setUp(self):
        self.ntime = 100
        self.nbatch = 50
        self.times = np.empty((self.ntime, self.nbatch))
        self.values = np.empty((self.ntime, self.nbatch))
        for i in range(self.nbatch):
            tmax = ra.random()
            self.times[:, i] = np.linspace(0, tmax, self.ntime)
            self.values[:, i] = ra.random((self.ntime,))

        self.obj = utility.BatchTimeSeriesInterpolator(
            torch.tensor(self.times), torch.tensor(self.values)
        )

    def test_interpolate_random(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = ra.uniform(0, self.times[-1, i])
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = self.obj(torch.tensor(targets))
        self.assertTrue(np.allclose(Y, correct))

    def test_interpolate_zero(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = 0
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = self.obj(torch.tensor(targets))

        self.assertTrue(np.allclose(Y, correct))

    def test_interpolate_end(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = self.times[-1, i]
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = self.obj(torch.tensor(targets))

        self.assertTrue(np.allclose(Y, correct))


class TestCheaterInterpolateBatchTimesObject(unittest.TestCase):
    def setUp(self):
        self.ntime = 100
        self.nbatch = 50
        self.times = np.empty((self.ntime, self.nbatch))
        self.values = np.empty((self.ntime, self.nbatch))
        for i in range(self.nbatch):
            tmax = ra.random()
            self.times[:, i] = np.linspace(0, tmax, self.ntime)
            self.values[:, i] = ra.random((self.ntime,))

        self.idx = 73
        self.obj = utility.CheaterBatchTimeSeriesInterpolator(
            torch.tensor(self.times), torch.tensor(self.values)
        )

    def test_interpolate_random(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = self.times[self.idx, i]
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = self.obj(torch.tensor(targets))

        self.assertTrue(np.allclose(Y, correct))


class TestInterpolateBatchTimes(unittest.TestCase):
    def setUp(self):
        self.ntime = 100
        self.nbatch = 50
        self.times = np.empty((self.ntime, self.nbatch))
        self.values = np.empty((self.ntime, self.nbatch))
        for i in range(self.nbatch):
            tmax = ra.random()
            self.times[:, i] = np.linspace(0, tmax, self.ntime)
            self.values[:, i] = ra.random((self.ntime,))

    def test_interpolate_random(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = ra.uniform(0, self.times[-1, i])
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = utility.timeseries_interpolate_batch_times(
            torch.tensor(self.times), torch.tensor(self.values), torch.tensor(targets)
        )

        self.assertTrue(np.allclose(Y, correct))

    def test_interpolate_zero(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = 0
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = utility.timeseries_interpolate_batch_times(
            torch.tensor(self.times), torch.tensor(self.values), torch.tensor(targets)
        )

        self.assertTrue(np.allclose(Y, correct))

    def test_interpolate_end(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        for i in range(self.nbatch):
            targets[i] = self.times[-1, i]
            correct[i] = inter.interp1d(self.times[:, i], self.values[:, i])(targets[i])

        Y = utility.timeseries_interpolate_batch_times(
            torch.tensor(self.times), torch.tensor(self.values), torch.tensor(targets)
        )

        self.assertTrue(np.allclose(Y, correct))


class TestInterpolateSingleTimes(unittest.TestCase):
    def setUp(self):
        self.ntime = 100
        self.nbatch = 50
        self.times = np.linspace(0, ra.random(), self.ntime)
        self.values = np.empty((self.ntime, self.nbatch))
        for i in range(self.nbatch):
            self.values[:, i] = ra.random((self.ntime,))

    def test_interpolate_random(self):
        correct = np.empty((self.nbatch,))
        targets = np.empty((self.nbatch,))
        t = ra.uniform(0, self.times[-1])
        for i in range(self.nbatch):
            correct[i] = inter.interp1d(self.times, self.values[:, i])(t)

        Y = utility.timeseries_interpolate_single_times(
            torch.tensor(self.times), torch.tensor(self.values), torch.tensor(t)
        )

        self.assertTrue(np.allclose(Y, correct))

    def test_interpolate_zero(self):
        correct = np.empty((self.nbatch,))
        t = 0
        for i in range(self.nbatch):
            correct[i] = inter.interp1d(self.times, self.values[:, i])(t)

        Y = utility.timeseries_interpolate_single_times(
            torch.tensor(self.times), torch.tensor(self.values), torch.tensor(t)
        )

        self.assertTrue(np.allclose(Y, correct))

    def test_interpolate_end(self):
        correct = np.empty((self.nbatch,))
        t = self.times[-1]
        for i in range(self.nbatch):
            correct[i] = inter.interp1d(self.times, self.values[:, i])(t)

        Y = utility.timeseries_interpolate_single_times(
            torch.tensor(self.times), torch.tensor(self.values), torch.tensor(t)
        )

        self.assertTrue(np.allclose(Y, correct))
