#!/usr/bin/env python
'''
Implementation of Hallermann et al 2012 neuron model on a simple stick model
'''
import LFPy
import numpy as np
import os
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from GeneticOptimization import MEAutility as MEA
import surface_electrodes
from os.path import join
import utils 
import neuron

# plt.interactive(1)
plt.close('all')


################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################
folder = "morphologies/cell_hallermann_myelin"
#folder = "morphologies/cell_hallermann_unmyelin"
#folder = "morphologies/simple_axon_hallermann"
#folder = "morphologies/HallermannEtAl2012"
neuron.load_mechanisms(join(folder))

# Define cell parameters
cell_parameters = {          # various cell parameters,
    # 'morphology' : 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    #'morphology' : join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    'morphology' : join(folder,'cell_simple.hoc'), # simplified neuron model from HallermannEtAl2012
    #'morphology' : join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    #rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    # 'passive_parameters':dict(g_pas=1/30., e_pas=-65),
    'v_init' : -85.,    # initial crossmembrane potential
    # 'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : False,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 1000.,
    'dt' : 2.**-4,   # [ms] dt's should be in powers of 2 for both,
    'tstart' : -50.,    # start time of simulation, recorders start at t=0
    'tstop' : 100.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
    "extracellular": True,
    'custom_code': [join(folder, 'Cell parameters.hoc'),
                    join(folder, 'charge.hoc'),
                    join(folder, 'pruning.hoc')]
}

cell = LFPy.Cell(**cell_parameters)
#cell.set_rotation(x=np.pi/2)
#cell.set_rotation(y=np.pi/2)
#cell.set_rotation(z=np.pi/2)
n_tsteps = int(cell.tstop / cell.dt + 1)



print("number of segments: ", cell.totnsegs)

t = np.arange(n_tsteps) * cell.dt

pulse_start = 500
pulse_duration = 400

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start+pulse_duration)] = 1.

cortical_surface_height = np.max(cell.zend) +20 

# Parameters for the external field
sigma = 0.3
source_xs = np.array([-70, -70, -10, -10, 10, 10, 70, 70])
source_ys = np.array([-70, 70, -10, 10, 10, -10, -70, 70])
source_zs = np.ones(len(source_xs)) * cortical_surface_height #- 141
#source_amps = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * amp

total_n_runs = 20
cells = []
glb_vmem = []
glb_vext = []
spread = np.linspace(-200, -1000, total_n_runs)
num=0


cell = LFPy.Cell(**cell_parameters)
#zs = [int(.95*np.min(cell.zmid)), int(.5*np.min(cell.zmid)), int(.1934*np.min(cell.zmid)), 0, int(.95*np.max(cell.zmid))]
if 'my[0]' in cell.allsecnames:
    zs = [cell.get_idx('apic[0]')[0], cell.get_idx('soma')[0], cell.get_idx('axon[0]')[0], cell.get_idx('node[0]')[0],cell.get_idx('my[0]')[int(len(cell.get_idx('my[0]'))/2)], cell.get_idx('node[1]')[len(cell.get_idx('axon[1]'))-1]    ]
else:
    zs = [cell.get_idx('apic[0]')[0], cell.get_idx('soma')[0], cell.get_idx('axon[0]')[0], cell.get_idx('axon[1]')[0], cell.get_idx('axon[1]')[len(cell.get_idx('axon[1]'))-1]    ]

#zs = [-750, -500, -100, 0, 100]
#v_idxs = [cell.get_closest_idx(0, 0, z) for z in zs]
v_idxs = zs
#v_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(zs)) / (np.max(zs) - np.min(zs)))
v_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.zend)) / (np.max(np.abs(cell.zmid) - np.min(np.abs(cell.zmid)))))


for s in spread:
    amp = s
    source_amps = np.array([0, 0, 1, 1, 1, 1, 0, 0]) * amp
    ExtPot = surface_electrodes.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
    
    # Find external potential field at all cell positions as a function of time
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
    
    
    v_field_ext_stick = np.zeros((len(zs), n_tsteps))
    #v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
    v_field_ext_stick = v_cell_ext[v_idxs]
    
    # Insert external potential at cell
    cell.insert_v_ext(v_cell_ext, t)
    glb_vext.append(v_cell_ext)
    
    
    # Run simulation, electrode object argument in cell.simulate
    print "running simulation {0} of {1}".format(num, total_n_runs)
    cell.simulate(rec_imem=True, rec_vmem=True)
    spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
    if spike_time_loc[0]!=None:
        print("spike! at time {0} and position {1} segment {2}".format(spike_time_loc[0], cell.get_idx_name(spike_time_loc[1])[1], cell.get_idx_name(spike_time_loc[1])[0]   ))
    glb_vmem.append(cell.vmem)
    #cells.append(cell)
    cell = LFPy.Cell(**cell_parameters)
    num += 1

# plot 3d view of evolution of vmem as a function of stimulation amplitude (legend)


widx = 5 #to select in len(v_idxs)

fig = plt.figure()
ax = fig.gca(projection='3d', title="Vmem evolution at "+ cell.get_idx_name(v_idxs[widx])[1], xlabel="t [ms]", ylabel="Vext [mV]", zlabel="Vmem [mV]")

color=iter(plt.cm.rainbow(np.linspace(0,1,total_n_runs)))
yinfo = np.zeros(total_n_runs)

for i in range(num):
    #ax.plot_wireframe(t, glb_vext[i][v_idxs[0]], glb_vmem[i][v_idxs[0]], cmap=plt.cm.bwr )
    ax.plot_wireframe(t, np.sign(np.min(glb_vext[i][v_idxs[widx]]))*round( np.max(np.abs(glb_vext[i][v_idxs[widx]])), 2) , glb_vmem[i][v_idxs[widx]],  color=next(color) )



#ax.axes.set_yticks(yinfo)
#ax.axes.set_yticklabels(yinfo)
plt.show()
#ax2 = plt.subplot(111, title="Cell model", aspect=1, projection='3d',  xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]")













