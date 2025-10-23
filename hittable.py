from abc import ABC, abstractmethod
import numpy as np

class Hittable(ABC):
    '''
    '''
    def __init__(self, color):
        self.color = np.array(color)
    
    def sdf(self, p):
        '''
        p: point in space
        '''
        pass
    
    def normal(self, p, eps=1e-4):
        '''
        TODO: change this to spherical
        computes surface normal at point p
        '''
        dx = np.array([eps, 0, 0, 0])
        dy = np.array([0, eps, 0, 0])
        dz = np.array([0, 0, eps, 0])
        dw = np.array([0, 0, 0, eps])
        nx = self.sdf(p + dx) - self.sdf(p - dx)
        ny = self.sdf(p + dy) - self.sdf(p - dy)
        nz = self.sdf(p + dz) - self.sdf(p - dz)
        nw = self.sdf(p + dw) - self.sdf(p - dw)
        
        n = np.array([nx, ny, nz, nw])
        n -= np.dot(n, p) * p #projects gradient onto the tangent space
        
        return n/np.linalg.norm(n)
        


class Sphere(Hittable):
    '''Sphere in Spherical space
    '''
    
    def __init__(self, center, radius, color):
        super().__init__(color)
        self.center = np.array(center) / np.linalg.norm(center) #normalizes center
        self.radius = radius
        
    def sdf(self, p):
        d = np.arccos(np.clip(np.dot(p, self.center), -1.0, 1.0)) # gives angular distance between p and spheres center
        return d - self.radius
    
    def normal(self, p, eps=1e-4):
        pass

class Cylinder(Hittable):
    def __init__(self, radius, color):
        super().__init__(color)
        self.radius = radius
    
    def sdf(self, p):
        r = np.sqrt(p[2]**2 + p[3]**2)
        return np.arccos(np.clip(r, -1.0, 1.0)) - self.radius

class Half_space(Hittable):

    def __init__(self, color):
        super().__init__(color)
    
    def sdf(self, p):
        return np.arcsin(p[2]) 