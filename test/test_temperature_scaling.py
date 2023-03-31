import unittest

import numpy as np
import scipy.interpolate as inter
import torch

from pyoptmat import temperature


class TestConstantParameter(unittest.TestCase):
    def test_value(self):
        pshould = torch.tensor([1.0, 2.0])
        obj = temperature.ConstantParameter(pshould)
        pval = obj(torch.tensor(1.0))

        self.assertTrue(np.allclose(pshould.numpy(), pval.numpy()))

    def test_value_batch(self):
        pshould = torch.tensor([1.0, 2.0])
        obj = temperature.ConstantParameter(pshould)
        pval = obj(torch.linspace(1.0, 10.0, 10))

        self.assertTrue(np.allclose(np.tile(pshould[None, :], (10, 1)), pval.numpy()))

    def test_value_batch_batch(self):
        pshould = torch.tensor([1.0, 2.0])
        obj = temperature.ConstantParameter(pshould)
        pval = obj(torch.linspace(1.0, 10.0, 10).unsqueeze(0).expand(4, 10))

        self.assertTrue(
            np.allclose(np.tile(pshould[None, None, :], (4, 10, 1)), pval.numpy())
        )


class TestArrheniusScaling(unittest.TestCase):
    def test_value(self):
        A = 1.2
        Q = 100.0

        T = 100.0

        obj = temperature.ArrheniusScaling(torch.tensor(A), torch.tensor(Q))
        y1 = obj.value(torch.tensor(T))
        y2 = A * np.exp(-Q / T)

        self.assertTrue(np.allclose(y1.numpy(), y2))

    def test_value_batch(self):
        A = torch.linspace(1.2, 1.3, 50)
        Q = torch.linspace(100.0, 120, 50)

        T = torch.linspace(100.0, 200, 50)

        obj = temperature.ArrheniusScaling(A, Q)
        y1 = obj.value(T)
        y2 = A.numpy() * np.exp(-Q.numpy() / T.numpy())

        self.assertEqual(y1.shape, (50,))
        self.assertEqual(y2.shape, (50,))

        self.assertTrue(np.allclose(y1.numpy(), y2))

    def test_value_batch_batch(self):
        A = torch.linspace(1.2, 1.3, 50)
        Q = torch.linspace(100.0, 120, 50)

        T = torch.linspace(100.0, 200, 50)

        obj = temperature.ArrheniusScaling(A, Q)
        y1 = obj.value(T.unsqueeze(0).expand(7, 50))
        y2 = np.tile((A.numpy() * np.exp(-Q.numpy() / T.numpy()))[None, ...], (7, 1))

        self.assertEqual(y1.shape, (7, 50))
        self.assertEqual(y2.shape, (7, 50))

        self.assertTrue(np.allclose(y1.numpy(), y2))


class TestPolynomialScaling(unittest.TestCase):
    def test_value(self):
        coefs = torch.tensor([1.1, 2.5, 3.0])
        x = torch.ones((100,)) * 1.51
        obj = temperature.PolynomialScaling(coefs)
        y1 = obj.value(x)

        y2 = np.polyval(coefs.numpy(), x)

        self.assertEqual(y1.shape, (100,))
        self.assertEqual(y2.shape, (100,))

        self.assertTrue(np.allclose(y1.numpy(), y2))

    def test_value_batch(self):
        coefs = torch.tensor([[1.1] * 100, [2.5] * 100, [3.0] * 100])

        x = torch.ones((100,)) * 1.51
        obj = temperature.PolynomialScaling(coefs)
        y1 = obj.value(x)

        y2 = np.polyval(coefs.numpy(), x)

        self.assertEqual(y1.shape, (100,))
        self.assertEqual(y2.shape, (100,))

        self.assertTrue(np.allclose(y1.numpy(), y2))

    def test_value_batch_batch(self):
        coefs = torch.tensor([[1.1] * 100, [2.5] * 100, [3.0] * 100])

        x = torch.ones((100,)) * 1.51
        obj = temperature.PolynomialScaling(coefs)
        y1 = obj.value(x.unsqueeze(0).expand(10, 100))

        y2 = np.tile(np.polyval(coefs.numpy(), x)[None, ...], (10, 1))

        self.assertEqual(y1.shape, (10, 100))
        self.assertEqual(y2.shape, (10, 100))

        self.assertTrue(np.allclose(y1.numpy(), y2))

    def test_value_constant(self):
        coefs = torch.tensor([2.51])
        x = torch.ones((100,)) * 1.51
        obj = temperature.PolynomialScaling(coefs)
        y1 = obj.value(x)

        y2 = np.polyval(coefs.numpy(), x)

        self.assertTrue(np.allclose(y1.numpy(), y2))


class TestPiecewiseScaling(unittest.TestCase):
    def test_value(self):
        points = torch.tensor([0.0, 10.0, 20.0, 30.0])
        values = torch.tensor([1.0, 2.0, 4.0, -1.0])

        x = torch.linspace(0.1, 29.9, 5)[torch.randperm(5)]
        obj = temperature.PiecewiseScaling(points, values)

        y1 = obj.value(x)

        ifn = inter.interp1d(points.numpy(), values.numpy())
        y2 = ifn(x.numpy())

        self.assertTrue(np.allclose(y1.numpy(), y2))

    def test_value_batch(self):
        nbatch = 50
        points = torch.tensor([0.0, 10.0, 20.0, 30.0])

        values = torch.tensor(
            np.array(
                [
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                ]
            )
        ).T

        obj = temperature.PiecewiseScaling(points, values)

        x = torch.linspace(0.1, 29.9, nbatch)[torch.randperm(nbatch)]

        y1 = obj.value(x)

        y2 = np.zeros(y1.numpy().shape)

        for i in range(nbatch):
            ifn = inter.interp1d(points.numpy(), values[i].numpy())
            y2[i] = ifn(x[i].numpy())

        self.assertEqual(y1.shape, (50,))
        self.assertEqual(y2.shape, (50,))

        self.assertTrue(np.allclose(y1, y2))

    def test_value_batch_batch(self):
        nbatch = 50
        points = torch.tensor([0.0, 10.0, 20.0, 30.0])

        values = torch.tensor(
            np.array(
                [
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                ]
            )
        ).T

        obj = temperature.PiecewiseScaling(points, values)

        xt = torch.rand(7,nbatch) * 28.8 + 0.1

        y1 = obj.value(xt)

        y2 = np.zeros(y1.numpy().shape)
        
        for j in range(7):
            for i in range(nbatch):
                ifn = inter.interp1d(points.numpy(), values[i].numpy())
                y2[j, i] = ifn(xt[j,i].numpy())

        self.assertEqual(y1.shape, (7, 50))
        self.assertEqual(y2.shape, (7, 50))

        self.assertTrue(np.allclose(y1, y2))

    def test_value_batch_batch_lb(self):
        nbatch = 50
        points = torch.tensor([0.0, 10.0, 20.0, 30.0])

        values = torch.tensor(
            np.array(
                [
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                    np.random.rand(nbatch),
                ]
            )
        ).T

        obj = temperature.PiecewiseScaling(points, values)

        xt = torch.rand(7,nbatch) * 28.8 + 0.1
        xt[0,0] = 0.0

        y1 = obj.value(xt)

        y2 = np.zeros(y1.numpy().shape)
        
        for j in range(7):
            for i in range(nbatch):
                ifn = inter.interp1d(points.numpy(), values[i].numpy())
                y2[j, i] = ifn(xt[j,i].numpy())

        self.assertEqual(y1.shape, (7, 50))
        self.assertEqual(y2.shape, (7, 50))

        self.assertTrue(np.allclose(y1, y2))


class TestShearModulusScaling(unittest.TestCase):
    def setUp(self):
        self.mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )

    def test_value(self):
        self.A = 0.01

        Ts = torch.linspace(25, 950, 50) + 273.15

        obj = temperature.ShearModulusScaling(self.A, self.mu)

        v1 = obj.value(Ts)

        v2 = self.A * self.mu(Ts)

        self.assertTrue(np.allclose(v1.numpy(), v2))

    def test_value_batch(self):
        nbatch = 50
        self.A = torch.linspace(0.01, 0.1, nbatch)

        Ts = torch.linspace(25, 950, nbatch) + 273.15

        obj = temperature.ShearModulusScaling(self.A, self.mu)

        v1 = obj.value(Ts)

        v2 = self.A * self.mu(Ts)

        self.assertEqual(v1.shape, (50,))
        self.assertEqual(v2.shape, (50,))

        self.assertTrue(np.allclose(v1.numpy(), v2))

    def test_value_batch_batch(self):
        nbatch = 50
        self.A = torch.linspace(0.01, 0.1, nbatch)

        Ts = torch.linspace(25, 950, nbatch) + 273.15

        obj = temperature.ShearModulusScaling(self.A, self.mu)

        v1 = obj.value(Ts.unsqueeze(0).expand((4, 50)))

        v2 = np.tile((self.A * self.mu(Ts))[None, ...], (4, 1))

        self.assertEqual(v1.shape, (4, 50))
        self.assertEqual(v2.shape, (4, 50))

        self.assertTrue(np.allclose(v1.numpy(), v2))


class TestShearModulusScalingExp(unittest.TestCase):
    def setUp(self):
        self.mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )

    def test_value(self):
        self.A = -2.1

        Ts = torch.linspace(25, 950, 50) + 273.15

        obj = temperature.ShearModulusScalingExp(torch.tensor(self.A), self.mu)

        v1 = obj.value(Ts)

        v2 = np.exp(self.A) * self.mu(Ts)

        self.assertTrue(np.allclose(v1.numpy(), v2))

    def test_value_batch(self):
        nbatch = 50
        self.A = torch.linspace(-2.1, -1.0, nbatch)

        Ts = torch.linspace(25, 950, nbatch) + 273.15

        obj = temperature.ShearModulusScalingExp(self.A, self.mu)

        v1 = obj.value(Ts)

        v2 = np.exp(self.A) * self.mu(Ts)

        self.assertEqual(v1.shape, (50,))
        self.assertEqual(v2.shape, (50,))

        self.assertTrue(np.allclose(v1.numpy(), v2))

    def test_value_batch_batch(self):
        nbatch = 50
        self.A = torch.linspace(-2.1, -1.0, nbatch)

        Ts = torch.linspace(25, 950, nbatch) + 273.15

        obj = temperature.ShearModulusScalingExp(self.A, self.mu)

        v1 = obj.value(Ts.unsqueeze(0).expand((4, 50)))

        v2 = np.tile((np.exp(self.A) * self.mu(Ts))[None, ...], (4, 1))

        self.assertEqual(v1.shape, (4, 50))
        self.assertEqual(v2.shape, (4, 50))

        self.assertTrue(np.allclose(v1.numpy(), v2))


class TestMTSScaling(unittest.TestCase):
    def setUp(self):
        self.mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        self.b = 2.474e-7
        self.k = 1.38064e-20

    def test_value(self):
        tau0 = 1000.0
        g0 = 0.5
        p = 2.0 / 3.0
        q = 1.0 / 3.0

        Ts = torch.linspace(25, 950, 50) + 273.15

        obj = temperature.MTSScaling(tau0, g0, q, p, self.k, self.b, self.mu)

        v1 = obj.value(Ts)

        v2 = tau0 * (
            1.0 - (self.k * Ts / (self.mu(Ts) * self.b**3.0 * g0)) ** (1 / q)
        ) ** (1 / p)

        self.assertTrue(np.allclose(v1, v2))

    def test_batch(self):
        nbatch = 50

        tau0 = torch.linspace(100.0, 1000.0, nbatch)
        g0 = torch.linspace(0.5, 1.0, nbatch)
        p = torch.linspace(2.0 / 3.0, 1.0, nbatch)
        q = 1.0 / 3.0

        Ts = torch.linspace(25, 950, 50) + 273.15

        obj = temperature.MTSScaling(tau0, g0, q, p, self.k, self.b, self.mu)

        v1 = obj.value(Ts)

        v2 = tau0 * (
            1.0 - (self.k * Ts / (self.mu(Ts) * self.b**3.0 * g0)) ** (1 / q)
        ) ** (1 / p)

        self.assertEqual(v1.shape, (50,))
        self.assertEqual(v2.shape, (50,))

        self.assertTrue(np.allclose(v1, v2))

    def test_batch_batch(self):
        nbatch = 50

        tau0 = torch.linspace(100.0, 1000.0, nbatch)
        g0 = torch.linspace(0.5, 1.0, nbatch)
        p = torch.linspace(2.0 / 3.0, 1.0, nbatch)
        q = 1.0 / 3.0

        Ts = torch.linspace(25, 950, 50) + 273.15

        obj = temperature.MTSScaling(tau0, g0, q, p, self.k, self.b, self.mu)

        v1 = obj.value(Ts.unsqueeze(0).expand((6, 50)))

        v2 = tau0 * (
            1.0 - (self.k * Ts / (self.mu(Ts) * self.b**3.0 * g0)) ** (1 / q)
        ) ** (1 / p)
        v2 = np.tile(v2[None, ...], (6, 1))

        self.assertEqual(v1.shape, (6, 50))
        self.assertEqual(v2.shape, (6, 50))

        self.assertTrue(np.allclose(v1, v2))


class TestKMRateSensitivityScaling(unittest.TestCase):
    def test_value(self):
        A = -8.679
        mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        b = 2.474e-7
        k = 1.38064e-20

        Ts = torch.linspace(25, 950.0, 50) + 273.15

        obj = temperature.KMRateSensitivityScaling(A, mu, b, k)
        v1 = obj.value(Ts)

        mu_values = np.array([mu.value(T).numpy() for T in Ts])

        v2 = -mu_values * b**3.0 / (k * Ts * A)

        v2 = np.minimum(v2, 20.0)

        self.assertTrue(np.allclose(v1.numpy(), v2))

    def test_value_batch(self):
        A = torch.linspace(-8.679 - 1, -8.679 + 1, 50)
        mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        b = 2.474e-7
        k = 1.38064e-20

        Ts = torch.linspace(25, 950.0, 50) + 273.15

        obj = temperature.KMRateSensitivityScaling(A, mu, b, k)
        v1 = obj.value(Ts)

        mu_values = np.array([mu.value(T).numpy() for T in Ts])

        v2 = -mu_values * b**3.0 / (k * Ts * A.numpy())
        v2 = np.minimum(v2, 20.0)

        self.assertEqual(v1.shape, (50,))
        self.assertEqual(v2.shape, (50,))

        self.assertTrue(np.allclose(v1.numpy(), v2))

    def test_value_batch_batch(self):
        A = torch.linspace(-8.679 - 1, -8.679 + 1, 50)
        mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        b = 2.474e-7
        k = 1.38064e-20

        Ts = torch.linspace(25, 950.0, 50) + 273.15

        obj = temperature.KMRateSensitivityScaling(A, mu, b, k)
        v1 = obj.value(Ts.unsqueeze(0).expand(6, 50))

        mu_values = np.array([mu.value(T).numpy() for T in Ts])

        v2 = -mu_values * b**3.0 / (k * Ts * A.numpy())
        v2 = np.minimum(v2, 20.0)
        v2 = np.tile(v2[None, ...], (6, 1))

        self.assertEqual(v1.shape, (6, 50))
        self.assertEqual(v2.shape, (6, 50))

        self.assertTrue(np.allclose(v1.numpy(), v2))


class TestKMViscosityScaling(unittest.TestCase):
    def test_value(self):
        A = -8.679
        B = -0.744
        mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        b = 2.474e-7
        k = 1.38064e-20
        eps0 = 1e10

        Ts = torch.linspace(25, 950.0, 50) + 273.15
        mu_values = np.array([mu.value(T).numpy() for T in Ts])

        obj = temperature.KMViscosityScaling(A, torch.tensor(B), mu, eps0, b, k)

        v1 = obj.value(Ts)
        v2 = (
            np.exp(B)
            * mu_values
            * eps0 ** (k * Ts.numpy() * A / (mu_values * b**3.0))
        )

        self.assertTrue(np.allclose(v1, v2))

    def test_value_batch(self):
        A = torch.linspace(-8.679 - 1, -8.679 + 1, 50)
        B = torch.linspace(-0.744, -0.80, 50)
        mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        b = 2.474e-7
        k = 1.38064e-20
        eps0 = 1e10

        Ts = torch.linspace(25, 950.0, 50) + 273.15
        mu_values = np.array([mu.value(T).numpy() for T in Ts])

        obj = temperature.KMViscosityScaling(A, B, mu, eps0, b, k)

        v1 = obj.value(Ts)
        v2 = (
            np.exp(B.numpy())
            * mu_values
            * eps0 ** (k * Ts.numpy() * A.numpy() / (mu_values * b**3.0))
        )

        self.assertTrue(v1.shape, (50,))
        self.assertTrue(v2.shape, (50,))

        self.assertTrue(np.allclose(v1, v2))

    def test_value_batch_batch(self):
        A = torch.linspace(-8.679 - 1, -8.679 + 1, 50)
        B = torch.linspace(-0.744, -0.80, 50)
        mu = temperature.PolynomialScaling(
            torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
        )
        b = 2.474e-7
        k = 1.38064e-20
        eps0 = 1e10

        Ts = torch.linspace(25, 950.0, 50) + 273.15
        mu_values = np.array([mu.value(T).numpy() for T in Ts])

        obj = temperature.KMViscosityScaling(A, B, mu, eps0, b, k)

        v1 = obj.value(Ts.unsqueeze(0).expand(8, 50))
        v2 = (
            np.exp(B.numpy())
            * mu_values
            * eps0 ** (k * Ts.numpy() * A.numpy() / (mu_values * b**3.0))
        )

        v2 = np.tile(v2[None, ...], (8, 1))

        self.assertTrue(v1.shape, (8, 50))
        self.assertTrue(v2.shape, (8, 50))

        self.assertTrue(np.allclose(v1, v2))
