import numpy as np
from hittable import Sphere, Cylinder, Half_space
from PIL import Image
from linear_algebra import rotation_matrix_4d
from spherical_geometry import geodesic, position, orientation, move_forward, move_right, phong_shading

def main():
    width, height = 400, 400
    aspect_ratio = width / height

    Q = np.eye(4)

    # rotation
    theta = 0.1
    R_cam = rotation_matrix_4d(2, 1, theta) @ rotation_matrix_4d(0, 3, theta)
    Q = R_cam @ Q

    cam_pos = position(Q)
    right_cam, up_cam, back_cam = orientation(Q)
    forward_cam = -back_cam

    origin = np.array([0, 0, 0, 1.0])
    forward_obj = np.array([0, 0, 1, 0])

    step_size = 0.5
    
    # move forward 
    # origin, forward_obj = move_forward(origin, forward_obj, step_size)

    right_obj   = np.array([1, 0, 0, 0])
    
    # move right
    # origin, right_obj = move_right(origin, right_obj, step_size)
    
    sphere1_center = geodesic(origin, forward_obj, 0.7)
    sphere2_center = geodesic(origin, forward_obj + 0.47*right_obj, 0.7)
    sphere3_center = geodesic(origin, forward_obj - 0.7*right_obj, 0.7)
    # cylinder_center = geodesic(origin, forward_obj, 0.7)

    objects = [
        Sphere(center=sphere1_center, radius=0.1, color=[255, 0, 0]),
        Sphere(center=sphere2_center, radius=0.1, color=[0, 255, 0]),
        Sphere(center=sphere3_center, radius=0.1, color=[0, 0, 255]),
        # Cylinder(center=cylinder_center, radius=0.2, color=[255, 255, 0]),
    ]

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
    img.save("images/raymarch_s3_rotated_phong_2_cylinder.png")
    print("Image saved as raymarch_s3_rotated_forward1_right.png")

if __name__ == "__main__":
    main()