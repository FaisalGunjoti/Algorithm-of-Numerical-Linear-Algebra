import enum
import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as pl

# from krylov_musterlsg import cg, gmres, gmres_givens
from krylov import cg, gmres, gmres_givens


def create_A(k, ω):
    np.random.seed(17)
    D = np.diag(sum([[i] * i for i in range(1, k + 1)], []))
    m = D.shape[0]
    M = np.random.rand(m, m) - 0.5
    return D + ω * M

def preconditioner(A, alt=False):
    if alt:
        return np.triu(A)
    else:
        return np.diag(np.diag(A))


def magic(A, b, tol=1e-12):
    P = preconditioner(A)
    #! you should normally not do it like this!
    b_tilde = A.T @ solve_triangular(P.T, solve_triangular(P, b), lower=True)
    A_tilde = A.T @ solve_triangular(P.T, solve_triangular(P, A), lower=True)
    return cg(A_tilde, b_tilde, tol=tol)


def solve_benchmark(A, P1, P2, b):
    conv = {}
    sol = {}

    sol["gmres"], conv["gmres"] = gmres(A.copy(), b.copy(), tol=1e-12)
    sol["gmres_G"], conv["gmres_G"] = gmres_givens(A.copy(), b.copy(), tol=1e-12)
    sol["gmres_p1"], conv["gmres_p1"] = gmres(A.copy(), b.copy(), P=P1.copy(), tol=1e-12)
    sol["gmres_p2"], conv["gmres_p2"] = gmres(A.copy(), b.copy(), P=P2.copy(), tol=1e-12)
    sol["magic"], conv["magic"] = magic(A.copy(), b.copy(), tol=1e-12)
    sol["cg"], conv["cg"] = cg(A.copy(), b.copy(), tol=1e-12)

    print(f"{'-'*22}")
    print(f"{' '*10} ||r||/||b||")
    for alg, x in sol.items():
        r_b = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        print(f"{alg:{10}} {r_b:{11}.{3}}")
    print(f"{'-'*22}")

    return conv


if __name__ == "__main__":
    Ω = [0, 0.1, 1]
    fig1, ax1 = pl.subplots(len(Ω), 3)
    fig2, ax2 = pl.subplots(len(Ω), 1)
    fig1.tight_layout()
    fig2.tight_layout()
    for s, ω in enumerate(Ω):
        A = create_A(30, ω)
        P1 = preconditioner(A)
        P2 = preconditioner(A, True)
        b = np.ones(A.shape[0])

        Λ0 = np.linalg.eigvals(A)
        Λ1 = np.linalg.eigvals(solve_triangular(P1,A))
        Λ2 = np.linalg.eigvals(solve_triangular(P2,A))

        conv = solve_benchmark(A, P1, P2, b)

        for name, r_b in conv.items():
            ax2[s].semilogy(r_b, ".-", label=name)
            ax2[s].legend()
            ax2[s].set_xlabel("iterations")
            ax2[s].set_ylabel("||r||/||b||")
            ax2[s].set_title(f"solve  (D+ωM)x = b  with ω={ω}")
            for i,(name,Λ) in enumerate(zip(['A', 'inv(P1)A', 'inv(P2)A'],[Λ0, Λ1, Λ2])):
                ax1[s,i].scatter(np.real(Λ), np.imag(Λ))
                ax1[s,i].set_title(f"Eigenvalues of {name}")
                ax1[s,i].set_xlabel("Re")
                ax1[s,i].set_ylabel("Im")
    pl.show()
