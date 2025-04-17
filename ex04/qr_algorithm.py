import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl

def sign_f(x1):
    if x1 > 0:
        return 1
    else:
        return -1

def tridiag(A):
    # todo
    m, n = np.shape(A)
    if n != m:
        return ("Error, input matrix must be symmetric") 
    
    for i in range(m - 2):
        xk = np.copy(A[i+1:, i]) 
        xk_norm = np.linalg.norm(xk, ord=2) 
        sign = sign_f(xk[0]) 
        ek = np.zeros((m-i-1 ,1 )) 
        ek[0] = 1 
        
        vk = ek * sign * xk_norm + xk.reshape(-1,1) 
        vk /= (np.linalg.norm(vk, ord=2)) 
        
        A1 = np.dot(vk.T, A[i+1:, i:])
        A[i+1:, i:] -= 2 * np.dot(vk, A1)
        
        
        A2 = np.dot(A[:,i+1:] ,vk)
        A[:, i+1:] -= 2 * np.dot(A2, vk.T) 
        
        A = (A + A.T)/2 
        A[np.abs(A) < 1e-12] = 0
        
    return A


def QR_alg(T):
    t = []
    m = T.shape[0]
    t.append(np.abs(T[m - 1, m - 2]))
    
    while (np.abs(T[m - 1 ,m - 2]) >= 1e-12): 
        Q, R = np.linalg.qr(T)
        T = np.dot(R, Q)
        t.append(np.abs(T[m - 1, m - 2]))
        
        T[np.abs(T) < 1e-12] = 0
        T = (T + T.T) / 2
    return (T, t)


def wilkinson_shift(T):
    μ = 0
    # todo
    m = T.shape[0]
    B = np.copy(T[m - 2: ,m - 2:])
    delta_0 = (B[0,0] - B[1,1]) * 0.5
    μ = B[1,1] - (sign_f(delta_0) * B[0,1]**2) / (np.abs(delta_0) + np.sqrt(delta_0**2 + B[0,1]**2))
    return μ


def QR_alg_shifted(T):
    t = []
    # todo
    m = T.shape[0]
    t.append(np.abs(T[m - 1, m - 2]))
    
    while (np.abs(T[m - 1, m - 2]) >= 1e-12):
        Λ = wilkinson_shift(T)
        TΛ = T - Λ * np.identity(m)
        Q,R = np.linalg.qr(TΛ)
        
        T = np.dot(R, Q) + Λ * np.identity(m)
        t.append(np.abs(T[m - 1, m - 2]))
        T[np.abs(T) < 1e-12] = 0 
        T = (T + T.T) / 2 
    return (T, t)


def QR_alg_driver(A, shift):
    all_t = []
    Λ = []
    T = tridiag(A)
    m = T.shape[0]
    
    while m > 0:
        if m == 1:
            Λ.append(T[0, 0])
            break
        else:
            if shift == 1:
                T, t = QR_alg_shifted(T)
            else:
                T, t = QR_alg(T)

            Λ.append(T[-1, -1])
            all_t.extend(t)

            m -= 1
            T = T[:m, :m]

    return (Λ, all_t)


if __name__ == "__main__":

    matrices = {
        "hilbert": hilbert(4),
        "diag(1,2,3,4)+ones": np.diag([1, 2, 3, 4]) + np.ones((4, 4)),
        "diag(5,6,7,8)+ones": np.diag([5, 6, 7, 8]) + np.ones((4, 4)),
    }

    fig, ax = pl.subplots(len(matrices.keys()), 2, figsize=(10, 10))

    for i, (mat, A) in enumerate(matrices.items()):
        print(f"A = {mat}")
        Λ,_ = np.linalg.eig(A)
        print(f"Λ = {np.sort(Λ)}\n")
        for j, shift in enumerate([True, False]):
            Λ, conv = QR_alg_driver(A.copy(), shift)
            ax[i, j].semilogy(range(len(conv)), conv, ".-")
            ax[i, j].set_title(f"A = {mat}, shift = {shift}")

    pl.show()