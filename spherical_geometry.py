import numpy as np
from linear_algebra import rotation_matrix_4d


def move_forward(pos, forward_dir, step_size):
    forward_dir = forward_dir - np.dot(forward_dir, pos) * pos #projects foward_dir onto tangent plane of sphere at pos
    forward_dir /= np.linalg.norm(forward_dir) # normalize
    
    new_pos = geodesic(pos, forward_dir, step_size)
    
    new_forward = forward_dir - np.dot(forward_dir, new_pos) * new_pos 
    new_forward /= np.linalg.norm(new_forward) 
    
    return new_pos, new_forward

def move_backward(pos, forward_dir, step_size):
    return move_forward(pos, -forward_dir, step_size)

def move_right(pos, right_dir, step_size):
    right_dir = right_dir - np.dot(right_dir, pos) * pos 
    right_dir /= np.linalg.norm(right_dir) 
    
    new_pos = geodesic(pos, right_dir, step_size)
    
    new_right = right_dir - np.dot(right_dir, new_pos) * new_pos 
    new_right /= np.linalg.norm(new_right) 
    
    return new_pos, new_right

def move_left(pos, right_dir, step_size):
    return move_right(pos, -right_dir, step_size)

def tangent_to_light(p, q, obj_type="sphere"):
    cos_theta = np.dot(p, q)
    
    if obj_type == "sphere" and (np.isclose(cos_theta, 1.0) or np.isclose(cos_theta, -1.0)):
        raise ValueError(f"{p,q}: p and q cannot be identical or antipodal for a sphere")
    
    if np.isclose(cos_theta, 1.0) or np.isclose(cos_theta, -1.0):
        v = np.zeros_like(p)
        v[0] = 1.0
        v -= np.dot(v, p) * p
        if np.linalg.norm(v) < 1e-6:
            v[1] = 1.0
            v -= np.dot(v, p) * p
        return v / np.linalg.norm(v)
    
    sin_theta = np.sqrt(1 - cos_theta**2)
    v = (q - cos_theta * p) / sin_theta
    return v / np.linalg.norm(v)


"""
def tangent_to_light(p, q):
    cos_theta = np.dot(p, q)
    if np.isclose(cos_theta, 1.0) or np.isclose(cos_theta, -1.0):
        raise ValueError(f"{p,q}:p and q cannot be identical or antipodal")
    sin_theta = np.sqrt(1 - cos_theta**2)
    v = (q - cos_theta * p) / sin_theta
    return v / np.linalg.norm(v)
"""

def phong_shading(p, N, cam_pos, light_pos, object_color,
                  k_ambient=0.1, k_diffuse=0.6, k_specular=0.3, shininess=32,
                  light_color=np.array([1.0,1.0,1.0]), light_intensity=1.0,
                  L=None, V=None):
    if L is None:
        L = tangent_to_light(p, light_pos)
    if V is None:
        V = tangent_to_light(p, cam_pos)
    
    L2 = -L
    V2 = -V
    
    diff = max(np.dot(N, L), 0.0)
    R = 2 * np.dot(N, L) * N - L
    spec = max(np.dot(R, V), 0.0) ** shininess
    
    diff2 = max(np.dot(N, L2), 0.0)
    R2 = 2 * np.dot(N, L2) * N - L2
    spec2 = max(np.dot(R2, V2), 0.0) ** shininess
    
    color = (k_ambient * object_color +
             light_intensity * light_color * ((k_diffuse * diff * object_color +
                                              k_specular * spec * light_color) +
                                             (k_diffuse * diff2 * object_color +
                                              k_specular * spec2 * light_color)))
    color = np.clip(color, 0, 255)
    return color.astype(np.uint8)


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

