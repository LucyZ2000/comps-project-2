import numpy as np

def position(Q):
    return Q[:,-1]

def orientation(Q):
    return list(Q[:,:-1].T)

def geodesic(p, v, t):
    if not np.isclose(p @ p, 1):
        raise ValueError("Position must lie on sphere")
    r = np.linalg.norm(v)
    if np.isclose(r, 0):
        raise ValueError("Direction must be non-zero")
    if not np.isclose(p @ v, 0):
        raise ValueError("Direction must be tangent to sphere at position")
    return np.cos(t)*p + np.sin(t)*v/r

def distance(p, q):
   return np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
