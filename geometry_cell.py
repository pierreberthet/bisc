#!/usr/bin/env python
'''
Simulation of electrical stimulations on neurons.
Determine the threshold of current delivery needed to elicitate an AP on a neuron/axon at various depths.
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# from GeneticOptimization import MEAutility as MEA
from os.path import join
import utils
import neuron


plt.close('all')

###############################################################################
# Main script, set parameters and create cell, synapse and electrode objects
###############################################################################

# folder = "morphologies/cell_hallermann_myelin"
# folder = "morphologies/cell_hallermann_unmyelin"
# folder = "morphologies/simple_axon_hallermann"
folder = "morphologies/HallermannEtAl2012"
# folder = "morphologies/almog"
# folder = "morphologies/hay_model"
neuron.load_mechanisms(join(folder))
# morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
morph = join(folder, '28_04_10_num19.hoc')  # HallermannEtAl2012
# morph = join(folder, 'A140612.hoc')  # Almog model
# morph = join(folder, 'cell1.hoc')  # Hay model
# morph = join('morphologies', 'axon.hoc')  # Mainen&Sejnowski, 1996
# morph = join(folder, 'cell_simple_long.hoc')  # stick model based on Hallermann's
# custom_code = [join(folder, 'Cell parameters.hoc'),
custom_code = [join(folder, 'Cell parameters.hoc'),
               join(folder, 'charge.hoc')]
               # join(folder, 'pruning.hoc')]
               # join(folder, 'custom_code.hoc')]
               # join(folder, 'initialize_mechs.hoc')]


# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology': morph,  # simplified neuron model from HallermannEtAl2012
    # rm': 30000.,      # membrane resistance
    'cm': 1.0,         # membrane capacitance
    'Ra': 150,        # axial resistance
    # 'passive_parameters':dict(g_pas=1/30., e_pas=-65),
    'v_init': -85.,    # initial crossmembrane potential
    # 'e_pas': -65.,     # reversal potential passive mechs
    'passive': False,   # switch on passive mechs
    'nsegs_method': 'lambda_f',
    'lambda_f': 1000.,
    'dt': 2.**-6,   # [ms] dt's should be in powers of 2 for both,
    'tstart': -50.,    # start time of simulation, recorders start at t=0
    'tstop': 50.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments in cell.simulation()
    "extracellular": True,
    "pt3d": True,
    'custom_code': custom_code}


names = ["axon myelin", "soma axon myelin", "dendrite soma axon myelin", "axon nonmyelin"]

form_name = folder[folder.find('/')+1:]

cell = LFPy.Cell(**cell_parameters)
# Assign cell positions
# TEST with different distance between cells
x_cell_pos = [-10, 0, 0, 10]
y_cell_pos = [-1500, -10, 10, 0]
# z_cell_pos = np.zeros(len(x_cell_pos))
z_cell_pos = [-30., -1000 - (np.sum(cell.length)), 0.]

# utils.reposition_stick_horiz(cell)
# utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])

# xrot = [0., 2., 1.]
# cell.set_rotation(y=xrot[cell_id] * np.pi / 2)
cell.set_rotation(x=np.pi / 2)
cell.set_pos(x=x_cell_pos[1], y=y_cell_pos[0], z=z_cell_pos[0])


fig = plt.figure(figsize=[10, 8])
fig.subplots_adjust(wspace=0.1)

ax1 = plt.subplot(111, projection="3d",
                  title="t = " + str(spike_time_loc[0] * cell.dt) + "ms", aspect=1, xlabel="x [$\mu$m]",
                  ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[-600, 600], ylim=[-600, 600], zlim=[-1800, 200])
# [ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
#          c='k', clip_on=False) for idx in range(cell.totnsegs)]
# [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=-100, vmax=50)
col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
[ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
          # '-', c='k', clip_on=False) for idx in range(cell.totnsegs)]
          # '-', c=plt.cm.viridis((cell.vmem[idx][spike_time_loc[0]] - np.min(cell.vmem.T[spike_time_loc[0]])) / np.linalg.norm(cell.vmem.T[spike_time_loc[0]])),
          '-', c=cmap(col[idx]), clip_on=False) for idx in range(cell.totnsegs)]
# [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
ax1.scatter(cell.xmid[initial], cell.ymid[initial], cell.zmid[initial], '*', c='r')
# for idx in range(cell.totnsegs):
#     ax1.text(cell.xmid[idx], cell.ymid[idx], cell.zmid[idx], "{0}.".format(cell.get_idx_name(idx)[1]))

elev = 15     # Default 30
azim = 0    # Default 0
ax1.view_init(elev, azim)


# ax.axes.set_yticks(yinfo)
# ax.axes.set_yticklabels(yinfo)
plt.savefig("geo_morph_" + form_name + ".png", dpi=300)
plt.show()
