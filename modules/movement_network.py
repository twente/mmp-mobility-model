# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:01:46 2021

@author: Brian
"""

import numpy as np
import random
import math
from typing import Tuple

class MovementNetwork:
    def _init_config(self):
        self._config = dict()
        #Number of nodes
        self._config["N"] = 25
        #Size of simulation region (W x W)
        self._config["W"] = 10
        #Velocity (average move distance per time step)
        self._config["v"] = 1
        #Distance between two nodes in order for link to exist
        self._config["d"] = 1.5
        #Number of time steps
        self._config["T"] = 1000
        #Maximum absolute angle change in a single time step
        self._config["theta"] = 20
        
        #Maximum absolute angle offset when resetting angle for OOB handling
        self._config["phi"] = 30
        
        #Number of decimal places for rounding positions
        self._config["precision"] = 3
        #If N is not a square, take N random nodes from full lattice (True) or first N elements of lattice (False)
        self._config["sample"] = False
    
    #Init
    def __init__(self, config: dict = None):
        #Initialize config
        self._init_config()
        
        #Update config
        if config is not None:
            self.update_config(config)
        
        #Generate list of adjacency matrices as well (warning: uses a lot of memory for large N)
        self._KEEP_ADJ = True
    
    #Update config with new values
    def update_config(self, config: dict):
        self._config.update(config)
    
    #Initialize coordinates coordinates
    def _init_coords(self) -> np.ndarray:
        """
        Initialize coordinates of nodes
        Nodes are arranged in a square lattice inside region evenly spaced from each other and the borders
        If number of nodes is not a square, the smallest acceptable lattice is taken and will have some empty entries
        
        returns N-by-3 position array. Each row is a node and contains (x_pos, y_pos, angle).
        """
        
        #Get vars from config
        N = self._config["N"]
        W = self._config["W"]
        sample = self._config["sample"]
        precision = self._config["precision"]
        
        #Dimension of lattice
        n = int(np.ceil(np.sqrt(N)))
        
        #Max number of nodes that fit in lattice
        N_full = n*n
        
        #Step size of lattice, rounded to three decimals
        step = np.round(W/(n+1), precision)
        
        #Initialize coords array
        #float32 should be sufficiently accurate
        coords = np.zeros((N_full, 3), dtype="float32")
        
        #Iterate over rows of lattice and add coordinates
        for i in range(n):
            x_arr = step * (np.arange(n) + 1)
            y_arr = step * (i + 1) * np.ones(n)
            
            coords[i*n:(i+1)*n, 0] = x_arr
            coords[i*n:(i+1)*n, 1] = y_arr
        
        #Partial lattice
        if N < N_full:
            print("Warning: N is not a square number.")
            if sample:
                #Take random nodes from full lattice
                idx = random.sample(range(N_full), N)
                coords = coords[idx, :]
            else:
                #Take first N nodes from lattice
                coords = coords[0:N, :]
        
        #Generate angles in degrees. East is 0 degrees, increases counterclockwise (like unit circle).
        for i in range(N):
            #Don't include 360 since 0 = 360
            coords[i, 2] = random.randint(0, 359)
        
        return coords
    
    #Move nodes and calculate new positions and angles
    #returns N-by-3 array of updated positions after moving.
    def _move(self) -> None:
        #Get vars from config
        N = self._config["N"]
        v = self._config["v"]
        W = self._config["W"]
        theta = self._config["theta"]
        precision = self._config["precision"]
        
        #Initialize new positions by copying old positions
        last_position = self.POS[-1]
        new_position = np.copy(last_position)
        
        #Generate random angle offsets and add
        #Note: numpy's randint function is exclusive for the upper bound so we need to add 1
        angle_offset = np.random.randint(-theta, theta+1, size=N)
        new_position[:,2] = new_position[:,2] + angle_offset
        
        #Keep angle in range [0-360)
        new_position[:,2] = (new_position[:,2] + 360) % 360
        
        #Generate distance to move for each node
        #Uniformly distributed from 0 to 2v, so average move distance is v
        move_distance = np.random.rand(N) * 2 * v
        
        #Move nodes
        for i in range(N):
            old_X = last_position[i, 0]
            old_Y = last_position[i, 1]
            
            dist = move_distance[i]
            angle = new_position[i, 2]
            
            #New X, round to 3 figures
            new_X = old_X + dist * math.cos(math.radians(angle))
            new_X = round(new_X, precision)
            
            #New Y, round to 3 figures
            new_Y = old_Y + dist * math.sin(math.radians(angle))
            new_Y = round(new_Y, precision)
            
            #Update positions
            if new_X <= 0 or new_X >= W or new_Y <= 0 or new_Y >= W:
                #Handle OOB, set valid positions and angles
                new_X, new_Y, new_angle = self._oob_handler_naive(new_X, new_Y)
                new_position[i, 0] = new_X
                new_position[i, 1] = new_Y
                new_position[i, 2] = new_angle
            else:
                #If not OOB, positions are valid so we can just set the directly
                new_position[i, 0] = new_X
                new_position[i, 1] = new_Y
        
        self.POS.append(new_position)
    
    #Out of bounds handler for move function
    def _oob_handler_naive(self, old_X: float, old_Y: float) -> Tuple[float, float, int]:
        #Get vars from config
        W = self._config["W"]
        phi = self._config["phi"]
        precision = self._config["precision"]
        
        #Clamp X value to [0, W]
        new_X = np.clip(old_X, 0, W)
        new_X = round(new_X, precision)
        
        #Clamp Y value to [0, W]
        new_Y = np.clip(old_Y, 0, W)
        new_Y = round(new_Y, precision)
        
        #Calculate angle from center of region to position
        if new_X == W/2 and new_Y > W/2:
            #Set 90 and 270 degrees manually becase it causes divide by 0 in arctan.
            angle = 90
        elif new_X == W/2 and new_Y < W/2:
            angle = 270
        else:
            #Calculate between new_X and the center of the region
            angle = int(math.degrees(math.atan((new_Y - W/2)/((new_X - W/2)))))
            
            #Needs to add 180 for left half plane
            if new_X < W/2:
                angle = angle + 180
            
            #Keep in range 360
            angle = (angle + 360) % 360
        
        #Reverse angle so the new direction points to the center of the region
        new_angle = angle - 180
        #Keep in range 360
        new_angle = (new_angle + 360) % 360
        
        #Add a random angle offset
        new_angle = new_angle + random.randint(-phi, phi)
        
        #Keep in range 360
        new_angle = (new_angle + 360) % 360
        
        return new_X, new_Y, new_angle
    
    #Run
    def run(self):
        T = self._config["T"]
        
        self.POS = list()
        self.POS.append(self._init_coords())
        for i in range(T):
            self._move()
        
        #Strip angle from POS, we will probably never use it again
        #TODO: store angle separately
        for i in range(len(self.POS)):
            self.POS[i] = self.POS[i][:, [0, 1]]
        
        self._calc_contacts()
    
    #position: N-by-2 position array
    #returns N-by-N adjacency matrix, and contact list
    def _adjacency_matrix(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #Get vars from config
        N = self._config["N"]
        d = self._config["d"]
        
        #Calculation
        x_pos = position[:, 0] #column vector of x positions
        x_cols = np.tile(x_pos, (N, 1))  # horizontally stack, column is the same
        x_rows = x_cols.T # transpose, row is the same
        
        y_pos = position[:, 1]
        y_cols = np.tile(y_pos, (N, 1))
        y_rows = y_cols.T
        
        #Build matrix of distances
        distance = np.sqrt((x_cols - x_rows) ** 2 + (y_cols - y_rows) ** 2)
        
        A = distance
        # distance is smaller than dis then values change into 1 but the diagonal is 1
        A[A <= d] = 1
        A[A > d] = 0
        np.fill_diagonal(A, 0)  # change the diagonal value into 0
        
        #Adjacency matrix only contains 1 and 0, int8 is sufficient
        A = A.astype("int8")
        
        #Contact list
        C = np.array(np.where(A>0), dtype="int32").T
        
        return A, C
    
    #Calculate adjacency matrices and contact lists from positions
    def _calc_contacts(self):
        if self._KEEP_ADJ:
            self.ADJ = list()
        self.CON = list()
        
        for POS in self.POS:
            A, C = self._adjacency_matrix(POS)
            if self._KEEP_ADJ:
                self.ADJ.append(A)
            self.CON.append(C)
