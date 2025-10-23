import numpy as np
from hittable import Sphere, Cylinder, Half_space
from PIL import Image

width, height = 400, 400
aspect_ratio = width / height

camera_position = np.array([1.0, 0.0, 0.0, 0.0])

def tangent_basis(camera_pos):
    basis = []
    for i in range(4):
        v = np.zeros(4)
        v[i] = 1.0
        v -= np.dot(v, camera_pos) * camera_pos
        if np.linalg.norm(v) > 1e-6:
            v /= np.linalg.norm(v)
            basis.append(v)
        if len(basis) == 3:
            break
    return np.array(basis)

basis = tangent_basis(camera_position)  


objects = [
    Sphere(center=[-1, -1, -3, 0], radius=1, color=[255, 0, 0]),
    Sphere(center=[1, 1, -3, 0], radius=1, color=[255, 255, 0]),
    Cylinder(radius=0.5, color=[0, 255, 255]),
    Half_space(color=[200, 200, 200])
]

ambient = np.array([15, 15, 0])


image = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        npc_x, npc_y = (x + 0.5) / width, (y + 0.5) / height
        screen_x, screen_y = 2 * npc_x - 1, 1 - 2 * npc_y
        if aspect_ratio > 1:
            screen_x *= aspect_ratio
        else:
            screen_y /= aspect_ratio

        w = -1.0  # forward along basis[2]
        tangent_vector = screen_x * basis[0] + screen_y * basis[1] + w * basis[2]
        tangent_vector /= np.linalg.norm(tangent_vector)

        t = 0.0
        max_t = np.pi  # maximum angular distance
        eps = 1e-3
        max_steps = 128
        hit_color = None
        hit_obj = None

        for _ in range(max_steps):
            p = np.cos(t) * camera_position + np.sin(t) * tangent_vector
            p /= np.linalg.norm(p)

            min_d = np.inf
            for obj in objects:
                d = obj.sdf(p)
                if d < min_d:
                    min_d = d
                    hit_obj = obj

            if min_d < eps:
                distance = min_d
                color = hit_obj.color
                break
            
            t += min_d
            if t > max_t:
                break


        if hit_color is not None:
            n = hit_obj.normal(p)
            shaded = hit_color  
            image[y, x] = np.clip(shaded, 0, 255).astype(np.uint8)
        else:
            t_sky = 0.5 * (tangent_vector[1] + 1)
            sky = (1 - t_sky) * np.array([180, 200, 255]) + t_sky * np.array([60, 120, 255])
            image[y, x] = sky.astype(np.uint8)

img = Image.fromarray(image, mode='RGB')
img.save("raymarch_s3.png")
print("Image saved as raymarch_s3.png")