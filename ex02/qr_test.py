import numpy as np
from numpy.testing import assert_allclose
import unittest

import qr as submission


class QRTest(unittest.TestCase):
    def check_shape_common(self, A, B, Aname, Bname, fname):
        self.assertTupleEqual(
            B.shape,
            A.shape,
            msg=f"{fname}: {Bname} has a different shape than {Aname}",
        )

    def check_house(self, A):
        with self.subTest(m=A.shape[0], n=A.shape[1]):
            W, R = submission.implicit_qr(A.copy())
            self.check_shape_common(A, R, "A", "R", "implicit_qr")
            self.check_shape_common(A, W, "A", "W", "implicit_qr")

            assert_allclose(
                R,
                np.triu(R),
                atol=1e-15,
                err_msg="implicit_qr: R is not upper triangular",
                verbose=False,
            )
            assert_allclose(
                np.diag(np.conjugate(W.T) @ W),
                np.ones(W.shape[1], dtype=W.dtype),
                atol=1e-15,
                err_msg="implicit_qr: The columns of W are not normalized",
                verbose=False,
            )

    def check_form_q(self, A):
        with self.subTest(m=A.shape[0], n=A.shape[1]):
            W, R = submission.implicit_qr(A.copy())
            Q = submission.form_q(W)
            assert_allclose(
                Q.T.conjugate() @ Q,
                np.eye(Q.shape[0]),
                atol=1e-14,
                err_msg="form_q: Q is not unitary",
                verbose=False,
            )
            assert_allclose(
                Q @ R,
                A,
                atol=1e-14,
                err_msg="form_q: QR is not equal to A",
                verbose=False,
            )

    def test_qr_real(self):
        A = np.random.rand(5, 3) - 0.5
        self.check_house(A)
        self.check_form_q(A)

    def test_qr_complex(self):
        A = np.random.rand(3, 3) + np.random.rand(3, 3)*1j
        self.check_house(A)
        self.check_form_q(A)

    def test_qr_int(self):
        A = np.array([[1,2],[3,4],[5,6]])
        self.check_house(A)
        self.check_form_q(A)

if __name__ == "__main__":
    np.random.seed(4)
    unittest.main(verbosity=2)
