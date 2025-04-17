'''import numpy as np

def givens_rotation(a, b):
    if b == 0:
        c = 1
        s = 0
    else:
        r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
        c = a / r
        s = -b / r
    return c, s

def apply_givens(c, s, x, y):
    return c * x - s * y, s * x + c * y

def givens_qr(H):
    H = np.array(H, dtype=np.complex128)
    m_plus_1, m = H.shape
        
    R = H.copy()
    G = []

    for i in range(m):
        a = R[i, i]
        b = R[i+1, i]
        c, s = givens_rotation(a, b)

        G.append((c, s))

        for j in range(i, m):
            R[i, j], R[i+1, j] = apply_givens(c, s, R[i, j], R[i+1, j])

        R[i+1, i] = 0

    G = np.array(G)
    return G, R

def form_q(G):
    
    m = G.shape[0]  
    Q = np.eye(m + 1, dtype=np.complex128)  

    for i in range(m):
        c, s = G[i]
        
        G_mat = np.eye(m + 1, dtype=np.complex128)
        G_mat[i, i] = c
        G_mat[i, i + 1] = -s
        G_mat[i + 1, i] = s
        G_mat[i + 1, i + 1] = c
        
        Q = Q @ G_mat.T

    return Q
'''

import numpy as np

def givens_rotation(a, b):
    """
    Compute the Givens rotation matrix elements c and s for a and b.
    """
    if b == 0:
        c = 1
        s = 0
    else:
        r = np.hypot(a, b)
        c = a / r
        s = -b / r
    return c, s

def givens_qr(H):
    """
    Perform QR factorization of an upper Hessenberg matrix H using Givens rotations.
    
    Parameters:
    H (numpy.ndarray): An (m+1) x m upper Hessenberg matrix.
    
    Returns:
    R (numpy.ndarray): An (m+1) x m upper triangular matrix.
    G (numpy.ndarray): An m x 2 matrix where each row contains the (c, s) values of the Givens rotations.
    """
    m_plus_1, m = H.shape
    assert m_plus_1 == m + 1, "H must be an (m+1) x m matrix."
    
    R = H.copy()
    G = np.zeros((m, 2), dtype=complex)
    
    for j in range(m):
        a = R[j, j]
        b = R[j + 1, j]
        c, s = givens_rotation(a, b)
        G[j, :] = [c, s]
        
        # Apply the Givens rotation to zero out R[j + 1, j]
        for k in range(j, m):
            temp = c * R[j, k] - s * R[j + 1, k]
            R[j + 1, k] = s * R[j, k] + c * R[j + 1, k]
            R[j, k] = temp
        
        # Zero out the element below the diagonal
        R[j + 1, j] = 0
    
    return R, G

def form_q(G):
    """
    Construct the unitary matrix Q from the Givens rotations.
    
    Parameters:
    G (numpy.ndarray): An m x 2 matrix where each row contains the (c, s) values of the Givens rotations.
    
    Returns:
    Q (numpy.ndarray): An (m+1) x (m+1) unitary matrix.
    """
    m, _ = G.shape
    m_plus_1 = m + 1
    Q = np.eye(m_plus_1, dtype=complex)
    
    for j in range(m):
        c, s = G[j, :]
        G_j = np.eye(m_plus_1, dtype=complex)
        G_j[j, j] = c
        G_j[j, j + 1] = -np.conj(s)
        G_j[j + 1, j] = s
        G_j[j + 1, j + 1] = np.conj(c)
        Q = Q @ G_j.T
    
    return Q
