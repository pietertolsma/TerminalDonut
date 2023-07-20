import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from sympy import symbols, solve, S
import multiprocessing as mp

class Torus():
    def __init__(self, radius, thickness):
        self.r = radius
        self.R = thickness

    def intersection(self, ray):
        # ray is a (origin, direction) pair

        ox, oy, oz = ray[0]
        dx, dy, dz = ray[1]

        t = symbols('t', real=True)

        dotprod_d = dx**2 + dy**2 + dz**2
        dotprod_o = ox**2 + oy**2 + oz**2
        dotprod_mix = ox*dx + oy*dy + oz*dz

        c4 = (dotprod_d)**2
        c3 = 4*(dotprod_d)*(dotprod_mix)
        c2 = 2*(dotprod_d)*(dotprod_o-(self.r**2+self.R**2)) + 4*(dotprod_mix)**2 + 4*self.R**2*dy**2
        c1 = 4*(dotprod_o - (self.r**2 + self.R**2))*(dotprod_mix) + 8*self.R**2*oy*dy
        c0 = (dotprod_o - (self.r**2 + self.R**2))**2 - 4*self.R**2*(self.r**2-oy**2)

        t_res = solve(c4*t**4 + c3*t**3 + c2*t**2 + c1*t + c0, t, domain=S.Reals)
        t_res = np.array(t_res)
       # print(t_res)
        if len(t_res[t_res > 0]) == 0:
            # No intersection found
            return None
        
        t = t_res[t_res > 0].min()
        intersection = ray[0] + t*ray[1]

        return intersection
    
class Camera():
    def __init__(self, width, height, fov):
        self.width = width
        self.height = height
        self.fov = fov

        self.intr = np.array([[width/(2*np.tan(fov/2)), 0, width/2],
                                [0, height/(2*np.tan(fov/2)), height/2],
                                [0, 0, 1]])
        
        self.intr_inv = np.linalg.inv(self.intr)

        self.tf_mat = np.identity(4)

    def translate(self, center):
        self.tf_mat[:3, 3] = center

    def rotate(self, xyz_angles):
        rot = np.identity(4)
        rot[:3, :3] = R.from_euler('xyz', xyz_angles, degrees=True).as_matrix()
        self.tf_mat = rot @ self.tf_mat

    def get_rays(self):
        # returns a list of (origin, direction) pairs
        rays = np.zeros((self.width, self.height, 6))
        for x in range(self.width):
            for y in range(self.height):
                world_coords = self.intr_inv @ np.array([x, y, 1])
                origin = (self.tf_mat @ np.concatenate([world_coords, [1]]))[:3]
                direction = origin - self.tf_mat[:3, 3]
                rays[x, y, :] = np.concatenate((origin, direction))

        return rays


class Renderer():
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.camera = Camera(width, height, 1)
        self.camera.translate([0,0,-15])
        self.camera.rotate([0,0,0])
        self.rays = self.camera.get_rays()

        self.donut = Torus(5, 2)

    def _ray_trace_pixel(self, i):
        x, y = i // self.width, i % self.width
        ray = self.rays[x, y, :].reshape((2, 3))
        intersection = self.donut.intersection(ray)

        symbols = ["@", "%", "#", "*", "+", "=", "-", ":", "."]

        if intersection is not None:
            z_val = intersection[2]
            index = abs(int(z_val/1.5))
            if index >= len(symbols):
                return symbols[-1]
            return symbols[index]
        else:
            return "."

    def get_frame(self):
        # initialize frame
        frame = np.full((self.height, self.width), ".") 
        with mp.Pool() as pool:
            results = pool.map(self._ray_trace_pixel, range(self.width * self.height))
        for i, result in enumerate(results):
            x, y = i // self.width, i % self.width
            frame[y][x] = result
        return frame

    def clear_screen(self):
        sys.stdout.write("\n"* self.height)
        sys.stdout.flush()

    def render_frame(self, frame):
        self.clear_screen()
        for line in frame:
            sys.stdout.write(" ".join(line) + "\n")
            sys.stdout.flush()


renderer = Renderer(40, 40)

def loop():
    frame = renderer.get_frame()
    renderer.render_frame(frame)

    renderer.camera.rotate([45,45,0])
    renderer.rays=renderer.camera.get_rays()

def start(framerate=30):
    while True:
        loop()
        # sleep
        time.sleep((1/framerate))

if __name__ == '__main__':
    start()