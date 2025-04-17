import numpy as np

def givenRotation(a, b):
    if b == 0:
        c = 1
        s = 0
    else:
        r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)  
        c = np.conj(a) / r
        s = - np.conj(b) / r
        
    return c, s    

def givens_qr(A):
    
    A = A.astype(np.complex128)  
    _, m = A.shape
    
    R = A.copy()
    G = np.zeros((m, 2), dtype=np.complex128)  
    
    for j in range(m):
        a = R[j, j]
        b = R[j + 1, j]
        c, s = givenRotation(a, b)
        G[j, :] = [c, s]  
        
        for k in range(j, m):
            temp = c * R[j, k] - s * R[j + 1, k]
            R[j + 1, k] = s * R[j, k] + c * R[j + 1, k]
            R[j, k] = temp
        
        R[j + 1, j] = 0
        
    return G, R


def form_q(G):
    m, _ = G.shape
    m_plus_1 = m + 1
    Q = np.eye(m + 1, dtype=np.complex128)  
    
    for j in range(m):
        c, s = G[j, 0], G[j, 1]  
        G_j = np.eye(m_plus_1, dtype=np.complex128)
        G_j[j, j] = c
        G_j[j, j + 1] = -np.conj(s)
        G_j[j + 1, j] = s
        G_j[j + 1, j + 1] = np.conj(c)
        Q = Q @ G_j.T  
    
    return Q
