#!/usr/bin/env python
'''
Simulation of electrical stimulations on neurons.
Determine the threshold of current delivery needed to elicitate an AP on a neuron/axon at various depths.
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# from GeneticOptimization import MEAutility as MEA
from os.path import join
import utils
import neuron






# Parameters for the external field
sigma = 0.3
# source_xs = np.array([-50, -50, -15, -15, 15, 15, 50, 50])
# source_ys = np.array([-50, 50, -15, 15, 15, -15, -50, 50])
source_xs = np.array([-50, 0, 50, 0, 0])
source_ys = np.array([0, 50, 0, -50, 0])
source_zs = np.ones(len(source_xs))

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
stim_amp = 1.
n_stim_amp = -stim_amp / 4
# source_geometry = np.array([0, 0, 0, 0, stim_amp])  # monopole
source_geometry = np.array([-stim_amp, 0, stim_amp, 0, 0])  # dipole
# source_geometry = np.array([-stim_amp / 4, -stim_amp / 4, -stim_amp / 4, -stim_amp / 4, stim_amp])
# source_geometry = np.array([stim_amp, stim_amp, stim_amp, stim_amp, -stim_amp])

# source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])


 source_amps = source_geometry * amp
        ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
