import numpy as np
from numpy.testing import assert_allclose
import unittest

import qr_musterlsg as submission

class QRTest(unittest.TestCase):
    def check_shape_common(self, A, B, Aname, Bname, fname):
        self.assertTupleEqual(
            B.shape,
            A.shape,
            msg=f"{fname}: {Bname} has a different shape than {Aname}",
        )

    def check_givens(self, A):
        with self.subTest(m=A.shape[0], n=A.shape[1]):
            G, R = submission.givens_qr(A.copy())
            self.check_shape_common(A, R, "A", "R", "givens_qr")
            self.assertTupleEqual(
                G.shape,
                (A.shape[1],2),
                msg=f"givens_qr: G has the wrong format. Should be m+1 x 2",
            )

            assert_allclose(
                R,
                np.triu(R),
                atol=1e-15,
                err_msg="givens_qr: R is not upper triangular",
                verbose=False,
            )

            assert_allclose(
                np.diag(G @ G.T.conj()),
                np.ones(G.shape[0], dtype=G.dtype),
                atol=1e-15,
                err_msg="givens_qr: The rows of G are not normalized, i.e., aren't valid cos(φ),sin(φ) pairs.",
                verbose=False,
            )

    def check_form_q(self, A):
        with self.subTest(m=A.shape[0], n=A.shape[1]):
            G, R = submission.givens_qr(A.copy())
            Q = submission.form_q(G)
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
                err_msg="givens_qr -> form_q: QR is not equal to A",
                verbose=False,
            )

    def test_qr_real(self):
        A = np.triu(np.random.rand(5, 4) - 0.5, -1)
        self.check_givens(A)
        self.check_form_q(A)
        #print("real", A)

    def test_qr_complex(self):
        A = np.triu(np.random.rand(5, 4) + np.random.rand(5, 4)*1j, -1)
        self.check_givens(A)
        self.check_form_q(A)
        #print("complex", A)


    def test_qr_int(self):
        A = np.array([[1,2,3],[4,5,6],[0,7,8],[0,0,9]])
        self.check_givens(A)
        self.check_form_q(A)
        #print("int", A)

if __name__ == "__main__":
    np.random.seed(4)
    unittest.main(verbosity=2)
