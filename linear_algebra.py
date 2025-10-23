import numpy as np

def rotation_matrix_4d(axis1, axis2, theta):
    R = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    R[axis1, axis1] = c
    R[axis2, axis2] = c
    R[axis1, axis2] = -s
    R[axis2, axis1] = s
    return R

#generates 3 orthonormal vectors tangent to S3 at point p using gram schmidt

def complete_tangent_basis(p):
    basis = []
    for i in range(4):
        v = np.zeros(4)
        v[i] = 1.0
        v -= np.dot(v, p) * p  
        if np.linalg.norm(v) > 1e-6:
            v /= np.linalg.norm(v)
            basis.append(v)
        if len(basis) == 3:
            break
    return np.array(basis)