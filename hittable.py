from abc import ABC, abstractmethod
import numpy as np

class Hittable(ABC):
    '''
    '''
    def __init__(self, color):
        self.color = color
    
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
        nx = self.sdf(p + dx)[0] - self.sdf(p - dx)[0]
        ny = self.sdf(p + dy)[0] - self.sdf(p - dy)[0]
        nz = self.sdf(p + dz)[0] - self.sdf(p - dz)[0]
        nw = self.sdf(p + dw)[0] - self.sdf(p - dw)[0]
        
        n = np.array([nx, ny, nz, nw])
        return n/np.linalg.norm(n)
        


class Sphere(Hittable):
    '''Sphere in Spherical space
    '''
    def __init__(self, center, radius, color):
        super().__init__(color)
        self.center = np.array(center)
        self.radius = radius
        
    def sdf(self, radius, p):
        '''
        '''
        return np.arccos(p[3]) - radius
    
    def normal(self, p, eps=1e-4):
        pass

class Cylinder(Hittable):
    def __init__(self, center, color):
        super().__init__(color)
        self.center = np.array(center)
    
    def sdf(self, p, radius):
        r = np.sqrt(p[2]**2 + p[3]**2)
        return np.arccos(r) - radius, 

class Half_space(Hittable):
    
    def sdf(self, p):
        return np.arcsin(p[2]) 