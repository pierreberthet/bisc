#!/usr/bin/env python
'''
Simulation of electrical stimulations on neurons.
Test the activation function from Rattay 1989
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
import plotting_convention as pconv

plt.close('all')

###############################################################################
# Main script, set parameters and create cell, synapse and electrode objects
###############################################################################

folder = "morphologies/cell_hallermann_myelin"
# folder = "morphologies/cell_hallermann_unmyelin"
# folder = "morphologies/simple_axon_hallermann"
# folder = "morphologies/HallermannEtAl2012"
# folder = "morphologies/almog"
# folder = "morphologies/hay_model"
neuron.load_mechanisms(join(folder))
# morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
# morph = join(folder, '28_04_10_num19.hoc')  # HallermannEtAl2012
# morph = join(folder, 'A140612.hoc')  # Almog model
# morph = join(folder, 'cell1.hoc')  # Hay model
# morph = join('morphologies', 'axon.hoc')  # Mainen&Sejnowski, 1996
morph = join(folder, 'L_cell_simple.hoc')  # stick model based on Hallermann's
# custom_code = [join(folder, 'Cell parameters.hoc'),
custom_code = [join(folder, 'Cell parameters.hoc'),
               join(folder, 'charge.hoc')]
               # join(folder, 'pruning_full.hoc')]
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
	'tstop': 30.,   # stop simulation at tstop ms. These can be overridden
	# by setting these arguments in cell.simulation()
	"extracellular": True,
	"pt3d": False,
    'custom_code': custom_code}

names = ["axon myelin"]

cell = LFPy.Cell(**cell_parameters)
# Assign cell positions
# TEST with different distance between cells
x_cell_pos = [0]
y_cell_pos = [0]
z_cell_pos = [-50]

# utils.reposition_stick_horiz(cell)
# utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])

# cell.set_rotation(x=np.pi)
cell.set_pos(x=x_cell_pos[0], y=y_cell_pos[0], z=z_cell_pos[0])
# utils.reposition_stick_flip(cell)



n_tsteps = int(cell.tstop / cell.dt + 1)
t = np.arange(n_tsteps) * cell.dt

pulse_start = 240
pulse_duration = 160
print("pulse duration: {0} ms".format(pulse_duration * cell.dt))

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 50


# Parameters for the external field
sigma = 0.3

amp = 50  # * 10**3  # nA

polarity, n_elec, positions = utils.create_array_shape('line', 15)
source_xs = positions[0]
source_ys = positions[1]
source_zs = positions[2]

source_zs = np.ones(len(source_xs)) * cortical_surface_height

# utils.clamp_ends(cell, pulse_start, pulse_start + pulse_duration, voltage=-76., axis='z')
# utils.clamp_ends(cell, 0, pulse_start + pulse_duration, voltage=-76., axis='z')
source_amps = np.multiply(polarity, amp)

v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))

v_cell_ext[:, :] = utils.linear_field(cell, pulse_start, pulse_start + pulse_duration, n_tsteps, axis='z') * amp

cell.insert_v_ext(v_cell_ext, t)

# Run simulation, electrode object argument in cell.simulate
# print("running cell {2} distance from electrode: {0} current intensity: {1}").format(depth, amp, cell_id)
cell.simulate(rec_imem=True, rec_vmem=True)
spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
if spike_time_loc[0] is not None:
	print("!!!spike at segment {0} @ t = {1} ms").format(spike_time_loc[1], spike_time_loc[0] * cell.dt)
	c_vext = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]

# FIGURES
fig = plt.figure(figsize=[12, 8])
initial = spike_time_loc[1]

# ax1 = plt.subplot(111, projection="3d",
#                   title="t = " + str(spike_time_loc[0] * cell.dt) + "ms", aspect=1, xlabel="x [$\mu$m]",
#                   ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[-600, 600], ylim=[-600, 600], zlim=[-1800, 200])
# # [ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
# #          c='k', clip_on=False) for idx in range(cell.totnsegs)]
# # [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
# cmap = plt.cm.viridis
# norm = mpl.colors.Normalize(vmin=-100, vmax=50)
# col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
# [ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
#           # '-', c='k', clip_on=False) for idx in range(cell.totnsegs)]
#           # '-', c=plt.cm.viridis((cell.vmem[idx][spike_time_loc[0]] - np.min(cell.vmem.T[spike_time_loc[0]])) / np.linalg.norm(cell.vmem.T[spike_time_loc[0]])),
#           '-', c=cmap(col[idx]), clip_on=False) for idx in range(cell.totnsegs)]
# # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
# # ax1.colorbar()

# lin_field = utils.test_linear(axis='xz', dim=[-100, 100, 50, -2000]) 
# span = len(lin_field)
# # ax1.plot_surface(np.arange(-100, 100, len(lin_field)), np.zeros((len(lin_field[0]), len(lin_field))), lin_field, alpha=.5)
# ax1.plot_surface(np.arange(span), np.zeros(span), np.arange(span), color='k', cmap=plt.cm.inferno, alpha=.5)
# # ax1.plot_surface(np.mgrid[0:span, 0:span], np.zeros((span, span)), np.mgrid[0:span, 0:span], rstride=1, cstride=1, color='k', cmap=plt.cm.inferno, alpha=.5)
# ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
# ax1.scatter(cell.xmid[initial], cell.ymid[initial], cell.zmid[initial], '*', c='r')
# # for idx in range(cell.totnsegs):
# #     ax1.text(cell.xmid[idx], cell.ymid[idx], cell.zmid[idx], "{0}.".format(cell.get_idx_name(idx)[1]))


# Derivatives
space_resolution = 500
x_extent = 200
z_extent = 3500
z_top = 0
# v_field_ext = np.zeros((space_resolution, space_resolution))
v_field_ext = utils.test_linear(axis='xz', dim=[-x_extent, x_extent, cortical_surface_height, -z_extent]) * amp

# xf = np.linspace(-x_extent, x_extent)
zf = np.arange(-z_extent, z_top)
d_v_field_ext = np.zeros((v_field_ext.shape[0], v_field_ext.shape[1] - 1))
dz = zf[1] - zf[0]

for zidx in zf[::-1]:
    d_v_field_ext[:, zidx] = (v_field_ext[:, zidx + 1] - v_field_ext[:, zidx]) / dz

# double derivative of V in z-direction
dd_v_field_ext = np.zeros((v_field_ext.shape[0], d_v_field_ext.shape[1] - 1))

for zidx in zf[::-2]:
    dd_v_field_ext[:, zidx] = (d_v_field_ext[:, zidx + 1] - d_v_field_ext[:, zidx]) / dz


if spike_time_loc[0] is None:
	spike_time_loc = [pulse_start + 50]



ax1 = plt.subplot(231, title="t = " + str(spike_time_loc[0] * cell.dt) + "ms", aspect=3, xlabel="x [$\mu$m]",
                  ylabel="y [$\mu$m]", xlim=[-x_extent, x_extent], ylim=[-z_extent, cortical_surface_height])
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=-100, vmax=50)
col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
[ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
          # '-', c='k', clip_on=False) for idx in range(cell.totnsegs)]
          # '-', c=plt.cm.viridis((cell.vmem[idx][spike_time_loc[0]] - np.min(cell.vmem.T[spike_time_loc[0]])) / np.linalg.norm(cell.vmem.T[spike_time_loc[0]])),
          '-', c=cmap(col[idx]), clip_on=False) for idx in range(cell.totnsegs)]
im1 = plt.imshow(v_field_ext.T, extent=[-x_extent, x_extent, -z_extent, cortical_surface_height], cmap=plt.cm.bone, aspect='auto')

ax1.scatter(source_xs, source_zs, c=source_amps)
if initial is not None:
	ax1.scatter(cell.xmid[initial], cell.zmid[initial], marker='*', c='r')
fig.colorbar(im1, label='External Potential [mV]')

if initial is None:
	ax2 = plt.subplot(232, title="Vm", xlabel='ms', ylabel='mV')
else: 
	ax2 = plt.subplot(232, title="Vm of segment {0} ({1})".format(initial, cell.get_idx_name(initial)[1]), xlabel='ms', ylabel='mV')
	ax2.plot(cell.vmem[initial], label=cell.get_idx_name(initial)[1])
top = np.argmax(cell.zend)
bottom = np.argmin(cell.zend)
ax2.plot(cell.vmem[top], label="top " + cell.get_idx_name(top)[1])
ax2.plot(cell.vmem[bottom], label="bottom " + cell.get_idx_name(bottom)[1])

prev_labels = [item.get_text() for item in ax2.get_xticklabels()]
# labels = np.linspace(0., np.max(t), len(prev_labels))

# ax2.set_xticklabels(labels)
ax2.legend()

# ax3 = plt.figure(3)
ax3 = plt.subplot(233, projection='3d', title='V_m along the cell', xlabel='z [$\mu$m]', ylabel='t [ms]', zlabel='mV', aspect='auto')
# 1 before the pulse, 1 or 2 during (one at spike time if any), 1 after
zidx = 1

for i in range(200, pulse_start+pulse_duration + 50, 10):
	ax3.plot(np.arange(cell.totnsegs), np.ones(cell.totnsegs) * i, cell.vmem.T[i][cell.zmid.argsort()][::-1])

ax4 = plt.subplot(234, title="derivative of external potential", aspect=3, xlabel="x [$\mu$m]",
                  ylabel="y [$\mu$m]", xlim=[-x_extent, x_extent], ylim=[-z_extent, cortical_surface_height])
cmap = plt.cm.viridis
# norm = mpl.colors.Normalize(vmin=-100, vmax=50)
col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
[ax4.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
          # '-', c='k', clip_on=False) for idx in range(cell.totnsegs)]
          # '-', c=plt.cm.viridis((cell.vmem[idx][spike_time_loc[0]] - np.min(cell.vmem.T[spike_time_loc[0]])) / np.linalg.norm(cell.vmem.T[spike_time_loc[0]])),
          '-', c=cmap(col[idx]), clip_on=False) for idx in range(cell.totnsegs)]
im4 = plt.imshow(d_v_field_ext.T, extent=[-x_extent, x_extent, -z_extent, cortical_surface_height], cmap=plt.cm.bone, aspect='auto')

ax4.scatter(source_xs, source_zs, c=source_amps)
if initial is not None:
	ax4.scatter(cell.xmid[initial], cell.zmid[initial], marker='*', c='r')
fig.colorbar(im4, label='[mV / $\mu$m]')

ax5 = plt.subplot(235, title="2nd derivative", aspect=3, xlabel="x [$\mu$m]",
                  ylabel="y [$\mu$m]", xlim=[-x_extent, x_extent], ylim=[-z_extent, cortical_surface_height])
cmap = plt.cm.viridis
# norm = mpl.colors.Normalize(vmin=-100, vmax=50)
col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
[ax5.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
          # '-', c='k', clip_on=False) for idx in range(cell.totnsegs)]
          # '-', c=plt.cm.viridis((cell.vmem[idx][spike_time_loc[0]] - np.min(cell.vmem.T[spike_time_loc[0]])) / np.linalg.norm(cell.vmem.T[spike_time_loc[0]])),
          '-', c=cmap(col[idx]), clip_on=False) for idx in range(cell.totnsegs)]
im5 = plt.imshow(dd_v_field_ext.T, extent=[-x_extent, x_extent, -z_extent, cortical_surface_height], cmap=plt.cm.bone, aspect='auto')

ax5.scatter(source_xs, source_zs, c=source_amps)
if initial is not None:
	ax5.scatter(cell.xmid[initial], cell.zmid[initial], marker='*', c='r')
fig.colorbar(im5, label='[mV / $\mu m^2$]')

ax6 = plt.subplot(236, title='V_ext along the cell', xlabel='z [$\mu$]m', ylabel='mV')
for i in range(200, pulse_start+pulse_duration + 50, 10):
	ax6.plot(np.arange(cell.totnsegs), np.asarray(cell.v_ext).T[i][cell.zmid.argsort()][::-1])

plt.tight_layout()
pconv.mark_subplots(ax1, 'A', xpos=-.25 , ypos=.99)
pconv.mark_subplots(ax2, 'B', xpos=-.25 , ypos=.99)
pconv.mark_subplots(ax4, 'D', xpos=-.25 , ypos=.99)
pconv.mark_subplots(ax5, 'E', xpos=-.25 , ypos=.99)
# pconv.mark_subplots(ax3, 'C', xpos=-.25 , ypos=.99)
# pconv.mark_subplots(ax3, 'C')
pconv.mark_subplots(ax6, 'F', xpos=-.25 , ypos=.99)
fig.subplots_adjust(left=.08, right=.96, bottom=.07, top=.96, wspace=None, hspace=None)


plt.show()
