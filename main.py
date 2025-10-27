import numpy as np
from hittable import Sphere, Cylinder, Half_space
from PIL import Image
from spherical_geometry import geodesic, position, orientation
from scipy.linalg import expm

width, height = 400, 400
aspect_ratio = width / height

def move_forward(pos, forward_dir, step_size):
    forward_dir = forward_dir - np.dot(forward_dir, pos) * pos #projects foward_dir onto tangent plane of sphere at pos
    forward_dir /= np.linalg.norm(forward_dir) # normalize
    
    new_pos = geodesic(pos, forward_dir, step_size)
    
    new_forward = forward_dir - np.dot(forward_dir, new_pos) * new_pos 
    new_forward /= np.linalg.norm(new_forward) 
    
    return new_pos, new_forward

def rotation_matrix_4d(axis1, axis2, theta):
    R = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    R[axis1, axis1] = c
    R[axis2, axis2] = c
    R[axis1, axis2] = -s
    R[axis2, axis1] = s
    return R

Q = np.eye(4)

theta = 0.1
R_cam = rotation_matrix_4d(2, 1, theta) @ rotation_matrix_4d(0, 3, theta)
Q = R_cam @ Q

cam_pos = position(Q)
right_cam, up_cam, back_cam = orientation(Q)
forward_cam = -back_cam

origin = np.array([0, 0, 0, 1.0])
forward_obj = np.array([0, 0, 1, 0])
right_obj   = np.array([1, 0, 0, 0])

sphere1_center = geodesic(origin, forward_obj, 0.7)
sphere2_center = geodesic(origin, forward_obj + 0.47*right_obj, 0.7)
sphere3_center = geodesic(origin, forward_obj - 0.7*right_obj, 0.7)

objects = [
    Sphere(center=sphere1_center, radius=0.1, color=[255, 0, 0]),
    Sphere(center=sphere2_center, radius=0.1, color=[0, 255, 0]),
    Sphere(center=sphere3_center, radius=0.1, color=[0, 0, 255])
]

ambient = np.array([15, 15, 0], dtype=np.uint8)

def tangent_to_light(p, q):
    cos_theta = np.dot(p, q)
    if np.isclose(cos_theta, 1.0) or np.isclose(cos_theta, -1.0):
        raise ValueError("p and q cannot be identical or antipodal")
    sin_theta = np.sqrt(1 - cos_theta**2)
    v = (q - cos_theta * p) / sin_theta
    return v / np.linalg.norm(v)

def phong_shading(p, N, cam_pos, light_pos, object_color,
                  k_ambient=0.1, k_diffuse=0.6, k_specular=0.3, shininess=32,
                  light_color=np.array([1.0,1.0,1.0]), light_intensity=1.0):

    L = tangent_to_light(p, light_pos)
    V = tangent_to_light(p, cam_pos)
    
    diff = max(np.dot(N, L), 0.0)
    R = 2 * np.dot(N, L) * N - L
    spec = max(np.dot(R, V), 0.0) ** shininess
    
    color = (k_ambient * object_color +
             light_intensity * light_color * (k_diffuse * diff * object_color +
                                              k_specular * spec * light_color))
    color = np.clip(color, 0, 255)
    return color.astype(np.uint8)

image = np.zeros((height, width, 3), dtype=np.uint8)
light_pos = np.array([6.0, 6.0, -7.0, 5.0])
light_pos /= np.linalg.norm(light_pos)

image = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    npc_y = (y + 0.5) / height
    screen_y = 1 - 2 * npc_y
    if aspect_ratio <= 1:
        screen_y /= aspect_ratio

    for x in range(width):
        npc_x = (x + 0.5) / width
        screen_x = 2 * npc_x - 1
        if aspect_ratio > 1:
            screen_x *= aspect_ratio

        ray_dir = screen_x * right_cam + screen_y * up_cam + forward_cam
        ray_dir -= np.dot(ray_dir, cam_pos) * cam_pos
        ray_dir /= np.linalg.norm(ray_dir)

        t = 0.0
        max_t = 2 * np.pi
        eps = 1e-3
        max_steps = 128
        hit_color = None
        hit_obj = None

        for _ in range(max_steps):
            p = geodesic(cam_pos, ray_dir, t)

            min_d = np.inf
            hit_obj = None
            for obj in objects:
                d = obj.sdf(p)
                if d < min_d:
                    min_d = d
                    hit_obj = obj

            if min_d < eps:
                N = hit_obj.normal(p)
                cos_theta = np.dot(p, light_pos)
                if np.isclose(abs(cos_theta), 1.0):
                    offset = 1e-3 * np.random.randn(4)
                    light_p = light_pos + offset
                    light_p /= np.linalg.norm(light_p)
                else:
                    light_p = light_pos

                hit_color = phong_shading(p, N, cam_pos,
                                          light_pos=light_p,
                                          object_color=hit_obj.color)
                break

            t += min_d
            if t > max_t:
                break

        if hit_color is not None:
            image[y, x] = hit_color
        else:
            t_sky = 0.5 * (ray_dir[1] + 1)
            sky = (1 - t_sky) * np.array([180, 200, 255]) + t_sky * np.array([60, 120, 255])
            image[y, x] = sky.astype(np.uint8)


img = Image.fromarray(image, mode='RGB')
img.save("raymarch_s3_rotated.png")
print("Image saved as raymarch_s3_rotated.png")
