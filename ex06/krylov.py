import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import qr

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    r, p = b.copy(), b.copy()
    r_b = []
    for _ in range(m):
        A_p = A @ p
        alpha = np.dot(r, r) / np.dot(p, A_p)
        x = x + (alpha * p)
        r_next = r - alpha * A_p
        residual_norm = np.linalg.norm(r_next) / np.linalg.norm(b)
        r_b.append(residual_norm)
        
        if residual_norm < tol:
            break
    
        beta = np.dot(r_next, r_next) / np.dot(r, r)
        p = r_next + beta * p
        r = r_next
    return x, r_b
    # todo


def arnoldi_n(A, Q, P):
    # n-th step of arnoldi
    m, n = Q.shape
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)
    
    q = Q[:, n - 1]
    v = np.dot(A, q)
    
    for j in range(n):
        h[j] = np.dot(Q[:, j], v)
        v = v -  (h[j] * Q[:, j])
        
    h[n] = np.linalg.norm(v, 2)
    if h[n] == 0:
        raise ValueError("Zero division error. Check input matrix !!")
    
    q = v / h[n]
    
    # todo

    return h, q


def gmres(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    
    
    n = 100  
    H = np.zeros((n + 1, n))
    V = np.zeros((n + 1, m))
    r_0 = b - solve_triangular(P, A @ x, lower=False)
    beta = np.linalg.norm(r_0)
    V[0] = r_0 / beta

    for j in range(n):
        w_value = solve_triangular(P, A @ V[j], lower=False)

        H[: j + 1, j], w_value = np.dot(V[: j + 1], w_value), w_value - V[: j + 1].T @ np.dot(V[: j + 1], w_value)
        H[j + 1, j] = np.linalg.norm(w_value)    
        
        
        H[j + 1, j] = np.linalg.norm(w_value)
        V[j + 1] = w_value / H[j + 1, j]

        e_1 = np.zeros(j + 2)
        e_1[0] = beta
        Q, R = qr(H[:j + 2, :j + 1], mode='reduced')
        y_value = solve_triangular(R, Q.T @ e_1)[:j + 1]

        x_new = x + V[:j + 1].T @ y_value

        residual_new = np.linalg.norm(solve_triangular(P, A @ x_new, lower=False) - b)
        r_b.append(residual_new / np.linalg.norm(b))
        if residual_new < tol:
            return x_new, r_b
    
    # todo

    return x, r_b


def gmres_givens(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    
    Q, H = np.zeros([m, m], dtype=A.dtype), np.zeros([m + 1, m], dtype=A.dtype)
    Q[:, 0] = b / np.linalg.norm(b, 2)
    r = b
    r_b.append(np.linalg.norm(r, 2) / np.linalg.norm(b, 2))
     
    n, r_b = 100, [1] 
    H = np.zeros((n + 1, n))
    V = np.zeros((n + 1, m))
    r_0 = b - solve_triangular(P, A @ x, lower=False)
    beta = np.linalg.norm(r_0)
    V[0] = r_0 / beta

    for j in range(n):
        d = np.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
        cs, sn = (1.0, 0.0) if np.abs(d) < 1e-12 else (H[j, j] / d, H[j + 1, j] / d)        
        H[j, j], H[j + 1, j] = cs * H[j, j] - sn * H[j + 1, j], sn * H[j, j] + cs * H[j + 1, j]
        V[j, j], V[j + 1, j] = cs * V[j, j] - sn * V[j + 1, j], sn * V[j, j] + cs * V[j + 1, j]
        e = np.zeros(j + 2)
        e[0] = beta

        try:
            y = solve_triangular(H[:j + 1, :j + 1], e[:j + 1])
        except np.linalg.LinAlgError:
            print(f"Skipping iteration {j} because of Singularity")
            continue

        x_new = x + V[:j + 1].T @ y
        residual_new = np.linalg.norm(solve_triangular(P, A @ x_new, lower=False) - b)
        r_b.append(residual_new / np.linalg.norm(b))
        if residual_new < tol:
            return x_new, r_b
    
    # todo
    return x, r_b
