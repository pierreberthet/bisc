#!/usr/bin/env python
"""

Collection of classes and functions for MEA stimulation

Conventions:
position = [um]
current = [nA]
voltage = [mV]

"""

import numpy as np
from numpy import linalg as la
import os.path


class Electrode:
    """Electrode: position and current (uses semi-infinite plane to compute field)"""
    def __init__(self, position=None, current=False):
        # convert to numpy array
        if type(position) == list:
            position = np.array(position)

        self.position = position
        self.sigma = 0.3
        self.max_field = 1000

        if current:
            self.current = current
        else:
            self.current = 0

    def set_current(self, current):
        self.current = current

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_max_field(self, max_field):
        self.max_field = max_field

    def field_contribution(self, point):
        if any(point != self.position):
            return self.current / (2*np.pi*self.sigma*la.norm(point-self.position))
        else:
            print "WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ", self.max_field
            return self.max_field
        # add return of x-y-z components of the field


class MEA:
    """MEA: collection of electrodes"""
    def __init__(self, electrodes=None):
        if electrodes:
            self.electrodes = electrodes
            if type(electrodes) in (np.ndarray, list):
                self.number_electrode = len(electrodes)
                # print "Create MEA with ", len(self.electrodes), " electrodes"
            elif isinstance(electrodes, Electrode):
                self.number_electrode = 1
                # print "Create MEA with 1 electrode"
            else:
                self.number_electrode = 0
                print "Wrong arguments"
        else:
            self.number_electrode = 0
            # print "Created empty MEA: add electrodes!"

    def set_electrodes(self, electrodes):
        self.electrodes = electrodes
        self.number_electrode = len(electrodes)

    def set_currents(self, currents_array):
        for ii in range(self.number_electrode):
            self.electrodes[ii].set_current(currents_array[ii])

    def set_random_currents(self, amp=None):
        if amp:
            currents = np.random.randn(self.number_electrode) * amp
        else:
            currents = np.random.randn(self.number_electrode) * 10

        for ii in range(self.number_electrode):
            self.electrodes[ii].set_current(currents[ii])

    def reset_currents(self, amp=None):
        currents = np.zeros(self.number_electrode)

        for ii in range(self.number_electrode):
            self.electrodes[ii].set_current(currents[ii])
    
    def get_currents(self):
        currents = np.zeros(self.number_electrode)
        for ii in range(self.number_electrode):
            currents[ii] = self.electrodes[ii].current
        return currents

    def get_electrode_array(self):
        return self.electrodes

    def compute_field(self, points):

        vp = []

        if points.ndim == 1:
            vp = 0
            if len(points) != 3:
                print "Error: expected 3d point"
                return
            else:
                for ii in range(self.number_electrode):
                    vp += self.electrodes[ii].field_contribution(points)

        elif points.ndim == 2:
            if points.shape[1] != 3:
                print "Error: expected 3d points"
                return
            else:

                vp = np.zeros(points.shape[0])
                for pp in range(0, len(vp)):
                    pf = 0
                    cur_point = points[pp]
                    for ii in range(self.number_electrode):
                        pf += self.electrodes[ii].field_contribution(cur_point)

                    vp[pp] = pf

        return vp

    def save_currents(self, filename):
        with open(filename, 'w') as f:
            for a in range(self.number_electrode):
                print >> f, "%g" % (self.get_currents()[a])

        print 'Currents saved successfully to file ', f.name

    def load_currents(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                currents = []
                for line in f:
                    currents.append(int(line))

                if len(currents) != self.number_electrode:
                    print 'Error: number of currents in file different than number of electrodes'
                else:
                    print 'Currents loaded successfully from file ', f.name
                    self.set_currents(currents)
        else:
            print 'File does not exist'


class SquareMEA(MEA):
    """Square MEA with N electrodes per side and a certain pitch"""
    def __init__(self, dim=None, pitch=None, x_plane=None):
        MEA.__init__(self)

        self.dim = dim
        if pitch:
            self.pitch = pitch
        else:
            self.pitch = 10
        if x_plane:
            self.x_plane = x_plane
        else:
            self.x_plane = 0

        # Create matrix of electrodes
        if (self.dim % 2 is 0):
            # print self.dim, 'even self.dim'
            sources_pos_y = range(-self.dim / 2 * self.pitch + self.pitch / 2, self.dim / 2 * self.pitch, self.pitch)
        else:
            sources_pos_y = range(-(self.dim / 2) * self.pitch, (self.dim / 2) * self.pitch + self.pitch, self.pitch)
        sources_pos_z = sources_pos_y[::-1]
        sources = []
        for ii in sources_pos_y:
            for jj in sources_pos_z:
                # list converted to np.array in Electrode constructor
                sources.append(Electrode([self.x_plane, ii, jj]))

        MEA.set_electrodes(self, sources)

    # override [] method
    def __getitem__(self, index):
        # return row of current matrix
        if index < self.dim:
            electrode_matrix = self.get_electrode_matrix()
            return electrode_matrix[index]
        else:
            print "Index out of bound"
            return None

    def get_electrodes_number(self):
        return self.dim**2

    def get_current_matrix(self):
        current_matrix = np.zeros((self.dim, self.dim))
        for yy in range(self.dim):
            for zz in range(self.dim):
                current_matrix[zz, yy] = MEA.get_currents(self)[self.dim * yy + zz]
        return current_matrix

    def get_electrode_matrix(self):
        electrode_matrix = [[0 for x in range(self.dim)] for y in range(self.dim)]
        for yy in range(0, self.dim):
            for zz in range(0, self.dim):
                electrode_matrix[zz][yy] = self.electrodes[self.dim * yy + zz]
        return electrode_matrix

    def set_current_matrix(self, currents):
        current_array = np.zeros((self.number_electrode))
        for yy in range(self.dim):
            for zz in range(self.dim):
                current_array[self.dim * yy + zz] = currents[zz, yy]
        MEA.set_currents(self, currents_array=current_array)


class GeometricNeuron:
    """GeometricNeuron is described with 3 parameters: soma_pos, dir_align, axon_length"""

    def __init__(self, soma_pos, align_dir, length = False, discrete_points = False):
        # Initialize neuron with 3d position and direction
        self.soma_pos = soma_pos
        if any(align_dir != np.array([0, 0, 0])):
            self.align_dir = align_dir/la.norm(align_dir)
        else:
            print "Error: axon must have a direction different than [0, 0, 0]"
            return
        if length:
            self.length = length
        else:
            self.length = 10
        if discrete_points:
            self.points = discrete_points
        else:
            self.points = 100

        # print "Created Geometric Neuron with soma at: ", self.soma_pos, " and direction ", self.align_dir

    def get_axon_points(self):
        steps = np.linspace(0, self.length, num=self.points)
        axon_points = np.array([self.soma_pos + st*self.align_dir for st in steps])
        return axon_points

    def get_axon_end(self):
        axon_end = self.soma_pos + self.length*self.align_dir
        return axon_end


