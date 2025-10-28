import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def tangent_to_light(p, q):
    cos_theta = np.dot(p, q)
    if np.isclose(cos_theta, 1.0) or np.isclose(cos_theta, -1.0):
        raise ValueError("p and q cannot be identical or antipodal")
    sin_theta = np.sqrt(1 - cos_theta**2)
    v = (q - cos_theta * p) / sin_theta
    return v / np.linalg.norm(v)

def tangent_to_light_2(p, q):
    cos_theta = np.dot(p, q)
    theta_2 = 2 * np.pi - np.arccos(cos_theta)
    cos_theta_2 = np.cos(theta_2)
    sin_theta_2 = np.sqrt(1 - cos_theta_2**2)
    v = (q - cos_theta_2 * p) / sin_theta_2
    return v / np.linalg.norm(v)

# Example points on the sphere
p = np.array([0, 0, 1])   # North pole
q = np.array([0, 1, 0])   # Point on equator

# Compute tangent directions
v1 = tangent_to_light(p, q)
v2 = tangent_to_light_2(p, q)

# Create a sphere for visualization
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='lightgray', alpha=0.3)

# Plot points and tangent vectors
ax.quiver(*p, *v1, color='blue', length=0.5, normalize=True, label='short way')
ax.quiver(*p, *v2, color='red', length=0.5, normalize=True, label='long way')
ax.scatter(*p, color='black', s=60, label='p')
ax.scatter(*q, color='green', s=60, label='q')

ax.set_box_aspect([1,1,1])
ax.legend()
ax.set_title("Tangent directions from p to q on the sphere")
plt.show()
