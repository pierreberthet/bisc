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
import plotting_convention

plt.close('all')

###############################################################################
# Main script, set parameters and create cell, synapse and electrode objects
###############################################################################
lambdaf = 200.


# folder = "morphologies/cell_hallermann_myelin"
folder = "morphologies/EyalEtAl2016"
# folder = "morphologies/L23_PC_cADpyr229_5"
# get the template name
f = file(join(folder, "ActiveModels/model_0603_cell08_cm045.hoc"), 'r')
templatename = utils.get_templatename(f)
f.close()
neuron.load_mechanisms(join(folder, "mechanisms"))

add_synapses = False
cell = LFPy.TemplateCell(
                         morphology=join(folder, "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"),
# cell = LFPy.TemplateCell(morphology=join(folder, "morphology/dend-C260897C-P3_axon-C220797A-P3_-_Clone_0.asc"),
                         templatefile=join(folder, 'ActiveModels/model_0603_cell08_cm045.hoc'),
                         templatename=templatename,
                         templateargs=1 if add_synapses else 0,
                         tstop=50.,
                         dt=2. ** -4,
                         extracellular=True,
                         tstart=-50,
                         lambda_f=lambdaf,
                         nsegs_method='lambda_f',)

custom_code = []
custom_code = [join(folder, 'morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC'),
               join(folder, 'biophysics.hoc'),
               join(folder, 'synapses/synapses.hoc')]

# Define cell parameters
cell_parameters = {          # various cell parameters,
    # 'morphology': morph,  # simplified neuron model from HallermannEtAl2012
    # rm': 30000.,      # membrane resistance
    'cm': 1.0,         # membrane capacitance
    'Ra': 150,        # axial resistance
    # 'passive_parameters':dict(g_pas=1/30., e_pas=-65),
    'v_init': -85.,    # initial crossmembrane potential
    # 'e_pas': -65.,     # reversal potential passive mechs
    'passive': False,   # switch on passive mechs
    'nsegs_method': 'lambda_f',
    'lambda_f': lambdaf,
    'dt': 2. ** -4,   # [ms] dt's should be in powers of 2 for both,
    'tstart': -50.,    # start time of simulation, recorders start at t=0
    'tstop': 50.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments in cell.simulation()
    "extracellular": True,
    "pt3d": False,
    'custom_code': custom_code}

# names = ["axon myelin", "soma axon myelin", "dendrite soma axon myelin", "axon nonmyelin"]

form_name = folder[folder.find('/') + 1:]
cell = LFPy.TemplateCell(morphology=join(folder, "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"),
                         templatefile=join(folder, 'ActiveModels/model_0603_cell08_cm045.hoc'),
                         templatename=templatename,
                         templateargs=1 if add_synapses else 0,
                         tstop=50.,
                         dt=2. ** -4,
                         extracellular=True,
                         tstart=-50,
                         lambda_f=lambdaf,
                         nsegs_method='lambda_f')

cell.set_rotation(x=-np.pi / 2.)
cell.set_rotation(y=np.pi / 8.)
# Assign cell positions
# TEST with different distance between cells
x_cell_pos = [0, 0, 0, 10]
y_cell_pos = [0, -10, 10, 0]
# z_cell_pos = np.zeros(len(x_cell_pos))
z_cell_pos = [0., -1000 - (np.sum(cell.length)), 0.]

# utils.reposition_stick_horiz(cell)
# utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])

# xrot = [0., 2., 1.]
# # cell.set_rotation(y=xrot[cell_id] * np.pi / 2)
# cell.set_rotation(x=np.pi / 2)
# cell.set_pos(x=x_cell_pos[1], y=y_cell_pos[0], z=-np.max(cell.zend))

n_sec, names = utils.get_sections_number(cell)


fig = plt.figure(figsize=[10, 8])
fig.subplots_adjust(wspace=0.1)

ax1 = plt.subplot(111, projection="3d",
                  title="", aspect=1, xlabel="x [$\mu$m]",
                  ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[-600, 600], ylim=[-600, 600], zlim=[-1800, 200])
#          c='k', clip_on=False) for idx in range(cell.totnsegs)]
# [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=-100, vmax=50)
# col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
# col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
colr = plt.cm.Set2(np.arange(n_sec))
for i, sec in enumerate(names):
    [ax1.plot([cell.xstart[idx], cell.xend[idx]],
              [cell.ystart[idx], cell.yend[idx]],
              [cell.zstart[idx], cell.zend[idx]],
              '-', c=colr[i], clip_on=False) for idx in cell.get_idx(sec)]
    if sec != 'soma':
        ax1.plot([cell.xstart[cell.get_idx(sec)[0]], cell.xend[cell.get_idx(sec)[0]]],
                 [cell.ystart[cell.get_idx(sec)[0]], cell.yend[cell.get_idx(sec)[0]]],
                 [cell.zstart[cell.get_idx(sec)[0]], cell.zend[cell.get_idx(sec)[0]]],
                 '-', c=colr[i], clip_on=False, label=sec)
ax1.scatter(cell.xmid[cell.get_idx('soma')[0]], cell.ymid[cell.get_idx('soma')[0]],
            cell.zmid[cell.get_idx('soma')[0]], s=33, marker='o', c='k', alpha=.7, label='soma')

plt.legend()
# [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
# ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
# ax1.scatter(cell.xmid[initial], cell.ymid[initial], cell.zmid[initial], '*', c='r')
# for idx in range(cell.totnsegs):
#     ax1.text(cell.xmid[idx], cell.ymid[idx], cell.zmid[idx], "{0}.".format(cell.get_idx_name(idx)[1]))

elev = 15     # Default 30
azim = 45    # Default 0
ax1.view_init(elev, azim)


# ax.axes.set_yticks(yinfo)
# ax.axes.set_yticklabels(yinfo)
plt.savefig("geo_morph_" + form_name + ".png", dpi=200)
plt.show()
