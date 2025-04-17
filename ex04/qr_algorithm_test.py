from unittest.result import STDERR_LINE
import numpy as np
from numpy.testing import assert_allclose, verbose
from scipy.linalg import hilbert
import unittest
from sys import stderr

import qr_algorithm as submission

class QRTest(unittest.TestCase):
    def check_shape_common(self, A, B, Aname, Bname):
        self.assertTupleEqual(
            B.shape,
            A.shape,
            msg=f"Err: {Bname} has a different shape than {Aname}",
        )

    def check_tridiag_common(self, T, Tname):
        with self.subTest(m=T.shape[0], n=T.shape[1]):
            assert_allclose(
                T,
                T.T,
                atol=1e-15,
                err_msg=f"{Tname} is not symmetric",
                verbose=False,
            )
            assert_allclose(
                T,
                np.triu(T, -1),
                atol=1e-15,
                err_msg=f"{Tname} is not upper Hessenberg",
                verbose=False,
            )
            assert_allclose(
                T,
                np.tril(T, 1),
                atol=1e-15,
                err_msg=f"{Tname} is not lower Hessenberg",
                verbose=False,
            )

    def check_similar_common(self, A, B, Aname, Bname):
        with self.subTest(m=A.shape[0], n=A.shape[1]):
            self.check_shape_common(A, B, Aname, Bname)

            Λ_A = np.sort(np.linalg.eig(A)[0])
            Λ_B = np.sort(np.linalg.eig(B)[0])

            assert_allclose(
                Λ_B,
                Λ_A,
                atol=1e-14,
                err_msg=f"Err: {Bname} is not similar to {Aname}",
                verbose=False,
            )

    def check_tridiag(self, A):
        T = submission.tridiag(A.copy())
        self.check_tridiag_common(T, "T")
        self.check_similar_common(A, T, "A", "T")

    def check_QR_alg(self, shift, A):
        T = np.zeros(A.shape)
        for k in [-1,0,1]:
            T += np.diag(np.diag(A,k),k)

        if shift:
            Tnew, t = submission.QR_alg_shifted(T.copy())
        else:
            Tnew, t = submission.QR_alg(T.copy())
        self.check_tridiag_common(Tnew, "Tnew")
        self.check_similar_common(T, Tnew, "T", "Tnew")

        with self.subTest(shift=shift):
            assert_allclose(
                0,
                np.abs(Tnew[-1, -2]),
                atol=1e-12,
                err_msg=f"Err: T_(m, m-1) is not sufficiently close to zero",
                verbose=False,
            )


    def check_QR_driver(self, shift, A):
        Λ, _ = submission.QR_alg_driver(A, shift)
        Λ_ref, _ = np.linalg.eig(A)

        Λ = np.sort(np.array(Λ))
        Λ_ref = np.sort(Λ_ref.real)

        with self.subTest(m=A.shape[0], n=A.shape[1], shift=shift):
            assert_allclose(
                Λ,
                Λ_ref,
                atol=1e-14,
                err_msg=f"Err: QR_driver does not return correct eigenvalues",
                verbose=False,
            )


    def test_tridiag(self):
        A = np.random.rand(4, 4) - 0.5
        A = A @ A.T
        self.check_tridiag(A)
        A = np.random.rand(10, 10) - 0.5
        A += A.T
        self.check_tridiag(A)

    def test_qr_alg(self):
        A = hilbert(5)
        self.check_QR_alg(False, A)

    def test_qr_alg_shifted(self):
        A = hilbert(5)
        self.check_QR_alg(True, A)

    def test_driver(self):
        A = hilbert(5)
        self.check_QR_driver(False, A)
        self.check_QR_driver(True, A)


if __name__ == "__main__":
    np.random.seed(4)
    unittest.main(verbosity=2)
