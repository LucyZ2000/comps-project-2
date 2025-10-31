import numpy as np
from hittable import Sphere, Cylinder, Half_space
from PIL import Image
from linear_algebra import rotation_matrix_4d
from spherical_geometry import geodesic, position, orientation, tangent_to_light, move_forward, move_right, phong_shading

width, height = 400, 400
aspect_ratio = width / height

Q = np.eye(4)

theta = 0.1
R_cam = rotation_matrix_4d(0, 1, theta) 
Q = R_cam @ Q

"""
can combine camera movements by adding
@ rotation_matrix_4d(__, __, theta)
"""


cam_pos = position(Q)
right_cam, up_cam, back_cam = orientation(Q)
forward_cam = -back_cam

origin = np.array([0, 0, 0, 1.0])
forward_obj = np.array([0, 0, 1, 0])

step_size = 0.5
cam_pos, right_cam = move_right(cam_pos, right_cam, step_size)
#cam_pos, up_cam = move_right(cam_pos, up_cam, step_size)

right_obj   = np.array([1, 0, 0, 0])
up_obj = np.array([0, 1, 0, 0])

sphere1_center = geodesic(origin, forward_obj + 0.5*up_obj, 0.7)
sphere2_center = geodesic(origin, forward_obj + 0.4*right_obj, 0.7)
sphere3_center = geodesic(origin, forward_obj - 0.7*right_obj+0.5*up_obj, 0.7)
cylinder_center = geodesic(origin, 0.5*forward_obj - 1.5*right_obj + -0.5*up_obj, 0.7)

objects = [
    Sphere(center=sphere1_center, radius=0.1, color=[255, 0, 0]),
    Sphere(center=sphere2_center, radius=0.1, color=[0, 255, 0]),
    Sphere(center=sphere3_center, radius=0.1, color=[0, 0, 255]),
    Cylinder(center=cylinder_center, radius=0.1, color=[255, 255, 0]),
]


image = np.zeros((height, width, 3), dtype=np.uint8)
light_pos = np.array([10.0, 10.0, 10.0, 1.0])
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
        max_steps = 200
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
    
                if isinstance(hit_obj, Cylinder):
                    L = tangent_to_light(p, light_pos, obj_type="cylinder")
                    V = tangent_to_light(p, cam_pos, obj_type="cylinder")
                else:
                    L = tangent_to_light(p, light_pos, obj_type="sphere")
                    V = tangent_to_light(p, cam_pos, obj_type="sphere")
    
                hit_color = phong_shading(p, N, cam_pos, light_pos=light_pos,
                              object_color=hit_obj.color, L=L, V=V)
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
