import math
from dataclasses import dataclass

import numpy as np
from math import pi

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    
    # def __init__(self, x=0.0, y=0.0, theta=0.0):
    #     self.x = x
    #     self.y = y
    #     self.theta = theta

    def to_parent_transformation(self):
        """Return the homogeneous transformation matrix mapping from pose to parent frame"""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        x = self.x
        y = self.y
        transform = np.array([
            [c, -s, x],
            [s, c, y],
            [0, 0, 1.0]])
        return transform

    def from_parent_transformation(self):
        """Return the homogeneous transformation matrix mapping from parent frame to own frame"""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        x = self.x
        y = self.y
        transform = np.array([
            [c, s, -x * c - y * s],
            [-s, c, x * s - y * c],
            [0, 0, 1.0]])
        return transform

    def distance_from_pose(self, other_pose):
        """Return distance between this pose and another"""
        return math.sqrt((self.x-other_pose.x)**2 + (self.y-other_pose.y)**2)   
    
    def angle_between(self, other_pose):
        """Return normalised angle in radians between this pose and another
        """
        angle = self.theta - other_pose.theta
        return self.normalise_angle(angle)
        
    def normalise_angle(self, angle):
        """ normalise angle to be in (-pi, pi] """
        two_pi = 2*math.pi
        # reduce the angle  
        angle =  angle % two_pi 

        # force it to be the positive remainder, so that 0 <= angle < 360  
        angle = (angle + two_pi) % two_pi  

        # force into the minimum absolute value residue class, so that -180 < angle <= 180  
        if angle > math.pi:  
            angle -= two_pi
            
        return angle
 
    def __str__(self):
        return f'({self.x:.3f}, {self.y:.3f}) {self.theta * (180 / pi):.0f}\u00b0'
    