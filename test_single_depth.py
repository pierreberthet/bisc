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
# morph = join(folder, 'cell_simple.hoc')  # stick model based on Hallermann's
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
    'lambda_f': 500.,
    'dt': 2.**-4,   # [ms] dt's should be in powers of 2 for both,
    'tstart': -50.,    # start time of simulation, recorders start at t=0
    'tstop': 50.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments in cell.simulation()
    "extracellular": True,
    "pt3d": True,
    'custom_code': custom_code}

names = ["axon myelin", "soma axon myelin", "dendrite soma axon myelin", "axon nonmyelin"]

cell = LFPy.Cell(**cell_parameters)
# Assign cell positions
# TEST with different distance between cells
x_cell_pos = [-10, 0, 0, 10]
y_cell_pos = [0, -10, 10, 0]
# z_cell_pos = np.zeros(len(x_cell_pos))
z_cell_pos = [-1300., -1000 - (np.sum(cell.length)), 0.]

# utils.reposition_stick_horiz(cell)
# utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])

# xrot = [0., 2., 1.]
# cell.set_rotation(y=xrot[cell_id] * np.pi / 2)
cell.set_rotation(x=np.pi / 2)
cell.set_pos(y=y_cell_pos[0], z=z_cell_pos[0])
# if RANK == 0:
#    cell.set_rotation(x=np.pi/2)
# cell.set_rotation(y=cell_id*np.pi/2)
# cell.set_rotation(z=np.pi/2)
# cell.set_rotation(x=0, y=0, z=z_rotation[cell_id])
# cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id])
# cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])

# cell.set_rotation(x=np.pi/2)
# cell.set_rotation(y=np.pi/2)
# cell.set_rotation(z=np.pi/2)

n_tsteps = int(cell.tstop / cell.dt + 1)

# print("number of segments: ", cell.totnsegs)

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
source_xs = np.array([-50, -50, -10, -10, 10, 10, 50, 50])
source_ys = np.array([-50, 50, -10, 10, 10, -10, -50, 50])

# source_xs = np.array([-50, 0, 50, 0, 0])
# source_ys = np.array([0, 50, 0, -50, 0])

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
stim_amp = 1.
n_stim_amp = -stim_amp / 4
# source_geometry = np.array([0, 0, 0, 0, stim_amp])
# source_geometry = np.array([-stim_amp / 4, -stim_amp / 4, -stim_amp / 4, -stim_amp / 4, stim_amp])
# source_geometry = np.array([stim_amp, stim_amp, stim_amp, stim_amp, -stim_amp])

source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])

# while loop? For loop?
spatial_resolution = 10
max_distance = 1000
distance = np.linspace(30, max_distance, spatial_resolution)
current = np.zeros(spatial_resolution)
c_vext = np.zeros(spatial_resolution)

source_zs = np.ones(len(source_xs)) * distance[0]

# Stimulation Parameters:
max_current = 10000000.   # nA
current_resolution = 2
# amp_range = np.exp(np.linspace(1, np.log(max_current), current_resolution))
amp_range = np.linspace(500 * (10 ** 3), max_current, current_resolution)
amp = amp_range[0]
num = 0

# WHILE/FOR
click = 0
# is_spike = np.zeros(n_cells)
is_spike = False

depth = distance[0]

while amp < max_current and not is_spike:

    amp = amp_range[click]
    source_zs = np.ones(len(source_xs)) * depth
    source_amps = source_geometry * amp
    ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)

    # Find external potential field at all cell positions as a function of time
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))

    # v_field_ext_stick = np.zeros((len(zs), n_tsteps))
    # v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
    # v_field_ext_stick = v_cell_ext[v_idxs]

    # Insert external potential at cell
    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=np.pi / 2)
    cell.set_pos(y=y_cell_pos[0], z=z_cell_pos[0])

    # utils.reposition_stick_horiz(cell)
    # utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])
    v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
                                        cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

    cell.insert_v_ext(v_cell_ext, t)

    # Run simulation, electrode object argument in cell.simulate
    # print("running cell {2} distance from electrode: {0} current intensity: {1}").format(depth, amp, cell_id)
    cell.simulate(rec_imem=True, rec_vmem=True)
    spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
    # COMM.Barrier()
    if spike_time_loc[0] is not None:
        is_spike = True
        current[np.where(distance == depth)[0][0]] = amp
        c_vext[np.where(distance == depth)] = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
    # glb_vmem.append(cell.vmem)
    # cells.append(cell)

    else:
        click += 1

initial = spike_time_loc[1]

print("Source current = {0} uA").format(amp / 1000.)
print("AP in {0} at t= {1}").format(cell.get_idx_name(spike_time_loc[1])[1], spike_time_loc[0])
print("v_ext = {0} mV").format(c_vext[np.where(distance == depth)][0])


fig = plt.figure(figsize=[10, 8])
fig.subplots_adjust(wspace=0.1)

ax1 = plt.subplot(111, projection="3d", title="t = " + str(spike_time_loc[0]), aspect=1, xlabel="x [$\mu$m]",
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
plt.show()
# plt.close()

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=-110, vmax=55)

window = 150
azim = 0
for t in range(spike_time_loc[0] - 10, spike_time_loc[0] + window):
    col = (cell.vmem.T[t] + 100) / 150.

    fig = plt.figure(figsize=[6, 8])
    fig.subplots_adjust(wspace=0.6)

    ax1 = plt.subplot(111, projection="3d", title="t = " + str(t), aspect=1, xlabel="x [$\mu$m]",
                      ylabel="y [$\mu$m]", zlabel="z [$\mu$m]",
                      xlim=[-600, 600], ylim=[-600, 600], zlim=[-1800, 200])

    [ax1.plot([cell.xstart[idx], cell.xend[idx]],
              [cell.ystart[idx], cell.yend[idx]],
              [cell.zstart[idx], cell.zend[idx]],
              '-', c=cmap(col[idx]), clip_on=False) for idx in range(cell.totnsegs)]
    # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
    ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
    ax1.scatter(cell.xmid[initial], cell.ymid[initial], cell.zmid[initial], '*', c='r')
    ax1.view_init(elev, azim)
    plt.savefig("outputs/activ_gif_" + str(t) + ".png", dpi=100)
    plt.close()
    print("fig {} out of {}").format(t - spike_time_loc[0], window)
    azim += 1








# plot 3d view of evolution of vmem as a function of stimulation amplitude (legend)


# fig = plt.figure(figsize=[18, 7])
# fig.subplots_adjust(wspace=0.6)
# # fig = plt.figure()
# ax = plt.subplot(122, title="Minimum current required to elicit a spike
# as a function of cell distance from the current source", xlabel="depth [um]", ylabel="current [mA]")

# color = iter(plt.cm.rainbow(np.linspace(0, 1, spatial_resolution)))

# # for i in range(num):
#     # ax.plot_wireframe(t, glb_vext[i][v_idxs[0]], glb_vmem[i][v_idxs[0]], cmap=plt.cm.bwr )
#     # ax.plot_wireframe(t, np.sign(np.min(glb_vext[i][v_idxs[widx]])) *
#                          round(np.max(np.abs(glb_vext[i][v_idxs[widx]])), 2),
#                                glb_vmem[i][v_idxs[widx]],  color=next(color) )

# ax.plot(distance, current[cell_id])

# ax1 = plt.subplot(121, title="3D view", aspect=1, projection='3d', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]",
#                   zlabel="z [$\mu$m]", xlim=[-1000, 1000], ylim=[-1000, 1000], zlim=[-1800, 200])
# [ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]],
#           '-', c='k', clip_on=False) for idx in range(cell.totnsegs)]

# ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)

# color = iter(plt.cm.rainbow(np.linspace(0, 1)))



#     fig = plt.figure(figsize=[18, 7])
#     fig.subplots_adjust(wspace=.6)
#     ax = plt.subplot(133, title="Stim threshold")
#     # axd = ax.twinx()
#     ax.set_xlabel("depth [$\mu$m]")
#     ax.set_ylabel("stimulation current [$\mu$A]")
#     # axd.set_ylabel("V_Ext [mV]")
#     for i in range(n_cells):
#         ax.plot(distance[:len(gather_current[i]['current'].nonzero()[0])],
#                 gather_current[i]['current'][gather_current[i]['current'].nonzero()[0]] / 1000.,
#                 color=next(color), label=names[i])
#         # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
#     # plt.xticks(np.linspace(0, max_distance, 10))
#     # plt.locator_params(tight=True)
#     plt.legend(loc="upper left")

#     ax2 = plt.subplot(131, title="V_ext", xlabel="x [$\mu$m]", ylabel='z [$\mu$m]')
#     source_amps = source_geometry * max_current
#     ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
#     plot_field_length = 1000
#     space_resolution = 200
#     v_field_ext_xz = np.zeros((space_resolution, space_resolution))
#     xf = np.linspace(-plot_field_length, plot_field_length, space_resolution)
#     zf = np.linspace(-plot_field_length, cortical_surface_height, space_resolution)
#     for xidx, x in enumerate(xf):
#         for zidx, z in enumerate(zf):
#             v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(x, 0, z)
#     v_field_ext_xy = np.zeros((space_resolution, space_resolution))
#     xf2 = np.linspace(-plot_field_length, plot_field_length, space_resolution)
#     yf2 = np.linspace(-plot_field_length, plot_field_length, space_resolution)
#     for xidx, x in enumerate(xf2):
#         for yidx, y in enumerate(yf2):
#             v_field_ext_xy[xidx, yidx] = ExtPot.ext_field(x, y, cortical_surface_height)

#     # vmin = -100
#     # vmax = 100
#     vmin = np.min([np.min(v_field_ext_xz), np.min(v_field_ext_xy)])
#     vmax = np.max([np.max(v_field_ext_xz), np.max(v_field_ext_xy)])
#     logthresh = 0
#     # maxlog = int(np.ceil(np.log10(vmax)))
#     # minlog = int(np.ceil(np.log10(-vmin)))
#     # tick_locations = ([-(10 ** x) for x in xrange(minlog, -logthresh - 1, -1)] +
#     #                   [0.0] + [(10**x) for x in xrange(-logthresh, maxlog + 1)])
#     imshow_dict = dict(origin='lower', interpolation='nearest',
#                        cmap=plt.cm.inferno, vmin=vmin, vmax=vmax,
#                        norm=matplotlib.colors.SymLogNorm(10**-logthresh))

#     img1 = ax2.imshow(v_field_ext_xz.T,
#                       extent=[-plot_field_length, plot_field_length,
#                               -plot_field_length, cortical_surface_height],
#                       **imshow_dict)
#     # cax = plt.axes([0.4, 0.1, 0.01, 0.33])
#     # cb = plt.colorbar(img1)
#     # cb.set_ticks(tick_locations)
#     # cb.set_label('mV', labelpad=-10)

#     ax2.scatter(source_xs, np.ones(len(source_xs)) * cortical_surface_height, c=source_amps, s=100, vmin=-1.4,
#                 vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

#     [ax2.scatter(source_xs[i], np.ones(len(source_xs))[i] * cortical_surface_height,
#                  marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)]
#     [ax2.scatter(source_xs[i], np.ones(len(source_xs))[i] * cortical_surface_height,
#                  marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)]

#     ax3 = plt.subplot(132, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
#     ax3.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
#                 edgecolor='k', lw=2, cmap=plt.cm.bwr)
#     [ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)]
#     [ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)]



#     img2 = ax3.imshow(v_field_ext_xy.T,
#                       extent=[-plot_field_length, plot_field_length,
#                               -plot_field_length, plot_field_length],
#                       **imshow_dict)
#     plt.colorbar(img2, label="mV")
#     # plt.colorbar(img1, ax=ax2, shrink=0.7, label="mV")
#     # plt.colorbar(img2, ax=ax3, shrink=0.7, label="mV")

#     # cax = plt.axes([0.335, 0.26, 0.01, 0.45])
#     # cb = plt.colorbar(img2, cax=cax)
#     # cb.set_ticks(tick_locations)
#     # cb.set_label('mV', labelpad=-10)
#     plt.savefig("sweep_test.png")

#     # fig2 = plt.figure(2)
#     # axu = fig2.gca(title="Stim threshold")
#     # # axd = ax.twinx()
#     # axu.set_xlabel("depth [$\mu$m]")
#     # axu.set_ylabel("stimulation current [mA]")
#     # # axd.set_ylabel("V_Ext [mV]")
#     # for i in range(n_cells):
#     #     axu.plot(gather_current[i]['current'] / 1000., color=next(color), label=names[i])
#     #     # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
#     # plt.xticks(np.arange(spatial_resolution), [format(depth, ".0f") for depth in distance])
#     # plt.legend(loc="upper left")

#     plt.show()
# # ax2 = plt.subplot(111, title="Cell model", aspect=1, projection='3d',
# #                   xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]")
