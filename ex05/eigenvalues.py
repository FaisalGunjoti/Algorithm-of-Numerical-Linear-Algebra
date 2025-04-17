import numpy as np


def gershgorin(A):
    λ_min, λ_max = 0,0

    # todo
    m = A.shape[0]
    for i in range(m):
        r = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        c = A[i, i]
        λ_min = min(λ_min, c - r)
        λ_max = max(λ_max, c + r)
        
    return λ_min, λ_max

def power(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    # todo        
    while(True):
        w = A @ v
        v = w / np.linalg.norm(w, 2)
        λ = (v.T) @ (A @ v)
        
        error = np.linalg.norm((A @ v) - (λ * v), np.inf)
        err.append(error)
        
        if error <= 10e-13:
            break
           
    return v, λ, err


def inverse(A, v0, μ):
    v = v0.copy()
    λ = 0
    err = []

    # todo
    i = np.eye(A.shape[0])        
    while(True):
        w = np.linalg.solve(A - μ * i, v)
        v = w / np.linalg.norm(w,2)
        λ = (v.T) @ (A @ v)
        
        error = np.linalg.norm((A @ v) - (λ * v), np.inf)
        err.append(error)
        
        if error <= 10e-13:
            break
        
    return v, λ, err


def rayleigh(A, v0):
    v = v0.copy()
    λ = 0
    err = []
    # todo
      
    λ = (v.T) @ (A @ v)
    i = np.eye(A.shape[0])
    while(True):
        w = np.linalg.solve(A - λ * i, v)
        v = w / np.linalg.norm(w,2)
        λ = (v.T) @ (A @ v)
        
        error = np.linalg.norm((A @ v) - (λ * v), np.inf)
        err.append(error)
        
        if error <= 10e-13:
            break     

    return v, λ, err


def randomInput(m):
    #! DO NOT CHANGE THIS FUNCTION !#
    A = np.random.rand(m, m) - 0.5
    A += A.T  # make matrix symmetric
    v0 = np.random.rand(m) - 0.5
    v0 = v0 / np.linalg.norm(v0) # normalize vector
    return A, v0


if __name__ == '__main__':
    pass
    # todo
    A, v0 = randomInput(5)
    A = np.array([[14, 0, 1],
                  [-3, 2, -2],
                  [5, -3, 3]])
    v0 = np.array([1, 1, 1])
    μ = 10
    print("A = ", A)
    print("V =", v0)
    print("µ =", μ)

    λ_min, λ_max = gershgorin(A)
    print("\nGershgorin circle :\nλ_min = {}, \tλ_max = {}".format(λ_min, λ_max))

    print("\nPower Iteration Method :")
    v, λ, err = power(A, v0)
    print("λ = {}, \terr = {}".format(λ, err[-1]))

    print("\nInverse Iteration Method :")
    v, λ, err = inverse(A, v0, μ)
    print("λ = {}, \terr = {}".format(λ, err[-1]))

    print("\nRayleigh Quotient Iteration Method :")
    v, λ, err = rayleigh(A, v0)
    print("λ = {}, \terr = {}".format(λ, err[-1]))
