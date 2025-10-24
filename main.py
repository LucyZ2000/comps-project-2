import numpy as np
from hittable import Sphere, Cylinder, Half_space
from PIL import Image
from linear_algebra import rotation_matrix_4d
from spherical_geometry import geodesic, position, orientation

width, height = 400, 400
aspect_ratio = width / height

Q = np.eye(4)  
cam_pos = position(Q)
right, up, back = orientation(Q)

theta = np.pi / 16
R = rotation_matrix_4d(0, 3, theta)
Q = R @ Q
cam_pos = position(Q)
right, up, back = orientation(Q)

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

        ray_dir = screen_x * right + screen_y * up - back
        ray_dir /= np.linalg.norm(ray_dir)

        t = 0.0
        max_t = np.pi  # maximum angular distance
        eps = 1e-3
        max_steps = 128
        hit_color = None
        hit_obj = None

        for _ in range(max_steps):
            p = geodesic(cam_pos, ray_dir, t)

            min_d = np.inf
            for obj in objects:
                d = obj.sdf(p)
                if d < min_d:
                    min_d = d
                    hit_obj = obj

            if min_d < eps:
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
            t_sky = 0.5 * (ray_dir[1] + 1)
            sky = (1 - t_sky) * np.array([180, 200, 255]) + t_sky * np.array([60, 120, 255])
            image[y, x] = sky.astype(np.uint8)

img = Image.fromarray(image, mode='RGB')
img.save("raymarch_s3.png")
print("Image saved as raymarch_s3.png")