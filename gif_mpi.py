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
from mpi4py import MPI
import plotting_convention

# initialize the MPI interface
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# number of units
n_cells = SIZE
cell_id = RANK

print("cell {0} of {1}").format(cell_id, n_cells)
# plt.interactive(1)
plt.close('all')
COMM.Barrier()

###############################################################################
# Main script, set parameters and create cell, synapse and electrode objects
###############################################################################
if RANK == 0:
    # folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    morph = join(folder, '28_04_10_num19.hoc') # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc') # Mainen&Sejnowski, 1996
    # morph = join(folder, 'cell_simple.hoc')
    # morph = join(folder, 'cell_simple_long.hoc')
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc')]
                   # join(folder, 'pruning_full.hoc')]

if RANK == 1:
    # folder = "morphologies/cell_hallermann_myelin"
    # # folder = "morphologies/cell_hallermann_unmyelin"
    # # folder = "morphologies/simple_axon_hallermann"
    # # folder = "morphologies/HallermannEtAl2012"
    # neuron.load_mechanisms(join(folder))
    # # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # # morph = join(folder, '28_04_10_num19.hoc') # HallermannEtAl2012
    # # morph = join('morphologies', 'axon.hoc') # Mainen&Sejnowski, 1996
    # morph = join(folder, 'cell_simple_long.hoc')
    # custom_code = [join(folder, 'Cell parameters.hoc'),
    #                join(folder, 'charge.hoc'),
    #                join(folder, 'pruning_full.hoc')]

    folder = "morphologies/EyalEtAl2016"
    # folder = "morphologies/L23_PC_cADpyr229_5"
    # get the template name
    f = file(join(folder, "ActiveModels/model_0603_cell08_cm045.hoc"), 'r')
    templatename = utils.get_templatename(f)
    f.close()
    neuron.load_mechanisms(join(folder, "mechanisms"))

    add_synapses = False
    cell = LFPy.TemplateCell(morphology=join(folder, "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"),
    # cell = LFPy.TemplateCell(morphology=join(folder, "morphology/dend-C260897C-P3_axon-C220797A-P3_-_Clone_0.asc"),
                             templatefile=join(folder, 'ActiveModels/model_0603_cell08_cm045.hoc'),
                             templatename=templatename,
                             templateargs=join(folder, "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"),
                             tstop=50.,
                             dt=2. ** -4,
                             extracellular=True,
                             tstart=-50,
                             lambda_f=200.,
                             nsegs_method='lambda_f',)




if RANK == 2:
    folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/hay_model"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # neuron.load_mechanisms(join(folder+"/mod")) # Hay model
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder, 'cell_simple_long.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc')]
    # ,join(folder, 'pruning.hoc')]

if RANK == 3:
    # folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/hay_model"
    folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # neuron.load_mechanisms(join(folder+"/mod")) # Hay model
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder, 'cell_simple_long.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning_full.hoc')]

if RANK == 4:
    # folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/hay_model"
    folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # neuron.load_mechanisms(join(folder+"/mod")) # Hay model
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder, 'cell_simple_long.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning.hoc')]


# NAMES
# names = ["axon myelin", "neuron myelin", "neuron nonmyelin", "axon nonmyelin"]
# names = ["long axon myelin clamped", "long axon myelin", "neuron myelin", "long axon unmyelin"]
names = ['Layer 5', 'Layer 2/3']

if names[cell_id] != 'Layer 2/3':
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
        'lambda_f': 200.,
        'dt': 2.**-4,   # [ms] dt's should be in powers of 2 for both,
        'tstart': -50.,    # start time of simulation, recorders start at t=0
        'tstop': 50.,   # stop simulation at 200 ms. These can be overridden
                            # by setting these arguments in cell.simulation()
        "extracellular": True,
        "pt3d": True,
        'custom_code': custom_code}
    cell = LFPy.Cell(**cell_parameters)


COMM.Barrier()



plt.close('all')

# Assign cell positions
# TEST with different distance between cells
# x_cell_pos = [-20, 0, 20, 10]

# x_cell_pos = [-1550, -1530, 0, 0]
x_cell_pos = np.zeros(n_cells)
y_cell_pos = np.linspace(-25, 25, n_cells)
# z_cell_pos = np.zeros(len(x_cell_pos))
z_cell_pos = [0, 0, 0, 0]

if names[cell_id] == 'Layer 5':
    cell = LFPy.Cell(**cell_parameters)
# Assign cell positions
# TEST with different distance between cells
# x_cell_pos = [-10, 0, 0, 10]
# y_cell_pos = [0, -10, 10, 0]
x_cell_pos = [0, 0]
y_cell_pos = [0, 0]
# z_cell_pos = np.zeros(len(x_cell_pos))
z_ratio = np.ones(n_cells) * -1.
if names[cell_id] == 'Layer 5':
    cell.set_rotation(x=np.pi / 2)
elif names[cell_id] == 'Layer 2/3':
    cell.set_rotation(x=-np.pi / 2.)
    cell.set_rotation(y=np.pi / 8.)

z_cell_pos_init = np.ones(n_cells)
COMM.Barrier()
z_cell_pos_init[cell_id] = -np.max(cell.zend)
z_cell_pos = z_cell_pos_init
cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])


# utils.reposition_stick_horiz(cell, x_cell_pos[cell_id], y_cell_pos[cell_id], z_cell_pos[cell_id])
# utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])

# xrot = [0., 2., 1.]
# cell.set_rotation(y=xrot[cell_id] * np.pi / 2)
# cell.set_rotation(x=np.pi / 2)
# cell.set_pos(x=x_cell_pos[1], y=y_cell_pos[0], z=z_cell_pos[0])
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

clamp = False
# CLAMPING
if clamp:
    if cell_id == 1:
        utils.clamp_ends(cell, 0, pulse_start + pulse_duration, -76.)

# Parameters for the external field
sigma = 0.3
# source_xs = np.array([-50, -50, -10, -10, 10, 10, 50, 50])
# source_ys = np.array([-50, 50, -10, 10, 10, -10, -50, 50])

# source_xs = np.array([-50, 0, 50, 0, 0])
# source_ys = np.array([0, 50, 0, -50, 0])

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
# stim_amp = 1.
# n_stim_amp = -stim_amp / 4
# source_geometry = np.array([0, 0, 0, 0, stim_amp])
# source_geometry = np.array([-stim_amp / 4, -stim_amp / 4, -stim_amp / 4, -stim_amp / 4, stim_amp])
# source_geometry = np.array([stim_amp, stim_amp, stim_amp, stim_amp, -stim_amp])

# source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])

distance = 50
c_vext = 0.

polarity, n_elec, positions = utils.create_array_shape('multipole3', 25)
source_xs = positions[0]
source_ys = positions[1]
source_zs = positions[2]

# Stimulation Parameters:

amp = 200 * (10**3)  # uA
num = 0

source_zs = np.ones(len(source_xs)) * distance
source_amps = np.multiply(polarity, amp)
ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)

# Find external potential field at all cell positions as a function of time
v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))

v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
                                    cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

# v_field_ext_stick = np.zeros((len(zs), n_tsteps))
# v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
# v_field_ext_stick = v_cell_ext[v_idxs]

# Insert external potential at cell
# cell.set_rotation(x=np.pi / 2)
# cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_], z=z_cell_pos[0])

# utils.reposition_stick_horiz(cell)
# utils.reposition_stick_flip(cell, x_cell_pos[0], y_cell_pos[0], z_cell_pos[0])

# v_cell_ext[:, :] = utils.linear_field(cell, pulse_start, pulse_start + pulse_duration, n_tsteps, 'z') * amp


cell.insert_v_ext(v_cell_ext, t)

# Run simulation, electrode object argument in cell.simulate
# print("running cell {2} distance from electrode: {0} current intensity: {1}").format(depth, amp, cell_id)
cell.simulate(rec_imem=True, rec_vmem=True)
spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
# COMM.Barrier()
if spike_time_loc[0] is not None:
    print("!!!spike  @  cell {0}").format(cell_id)
    c_vext = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
spike_time_loc = utils.spike_soma(cell)
if spike_time_loc[0] is not None:
    print("!!!SOMA spike  @  cell {0}").format(cell_id)
    c_vext = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
# print("cell {0} vmem {1}").format(cell_id, cell.vmem.T[spike_time_loc[0]])
vxmin, vxmax = utils.sanity_vext(cell.v_ext, t)

# glb_vmem.append(cell.vmem)
# cells.append(cell)

COMM.Barrier()
if cell_id == 0:
    tmin = vxmin
    tmax = vxmax
    for i in range(1, n_cells):
        temp = COMM.recv(source=i)
        if temp[0] < tmin:
            tmin = temp[0]
        if temp[1] > tmax:
            tmax = temp[1]
else:
    COMM.send([vxmin, vxmax], dest=0)

COMM.Barrier()

if cell_id == 0:
    cells = []
    cells.append(utils.built_for_mpi_space(cell, cell_id, spike_time_loc, cell.allsecnames))
    for i_proc in range(1, SIZE):
        cells.append(COMM.recv(source=i_proc))
else:
    COMM.send(utils.built_for_mpi_space(cell, cell_id, spike_time_loc, cell.allsecnames), dest=0)
COMM.Barrier()

if spike_time_loc[0] is not None:
    print("AP in {0} at t= {1}").format(cell.get_idx_name(spike_time_loc[1])[1], spike_time_loc[0])

if cell_id == 0:
    # print("Source current = {0} uA").format(amp / 1000.)
    # print("v_ext = {0} mV").format(c_vext)

    fig = plt.figure(figsize=[10, 8])
    fig.subplots_adjust(wspace=0.1)


    xlim_min = -750
    xlim_max = 750
    ylim_min = -200
    ylim_max = 200
    zlim_min = -2000
    zlim_max = 50

    ax1 = plt.subplot(111, projection="3d", title="t = " + str(spike_time_loc[0]), aspect='auto', xlabel="x [$\mu$m]",
                      ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])

    cmap = plt.cm.viridis
    cmap = [plt.cm.autumn, plt.cm.spring]
    norm = mpl.colors.Normalize(vmin=-110, vmax=55)
    for i in range(n_cells):
        # initial = cells[i]['extra'][1]
        initial = pulse_start
        # n_sec, names = utils.get_sections_number(cells[i])


        col = (cells[i]['vmem'].T[initial] + 100) / 150.
        [ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]], [cells[i]['ystart'][idx], cells[i]['yend'][idx]],
                  [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                  '-', c=cmap[i](col[idx]), clip_on=False) for idx in range(cells[i]['totnsegs'])]
        ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
        ap_i = cells[i]['extra1'][1]
        if ap_i is not None:
            ax1.scatter(cells[i]['xmid'][ap_i], cells[i]['ymid'][ap_i], cells[i]['zmid'][ap_i], '*', c='r')
        ax1.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0] - 20 * i, names[cells[i]['rank']])

        # for idx in range(cell.totnsegs):
        #     ax1.text(cell.xmid[idx], cell.ymid[idx], cell.zmid[idx], "{0}.".format(cell.get_idx_name(idx)[1]))

    elev = 10     # Default 30
    azim = -90    # Default 0
    ax1.view_init(elev, azim)

    # ax.axes.set_yticks(yinfo)
    # ax.axes.set_yticklabels(yinfo)
    plt.show()
    # plt.close()

    cmap = plt.cm.viridis
    cmap = [plt.cm.autumn, plt.cm.spring]

    norm = mpl.colors.Normalize(vmin=-110, vmax=55)

    ts_max = 0
    ts_min = len(cell.vmem[0])
    for i in range(n_cells):
        if np.max(cells[i]['extra1'][0]) > ts_max:
            ts_max = np.max(cells[i]['extra1'][0])
        if np.min(cells[i]['extra1'][0]) < ts_min:
            ts_min = np.min(cells[i]['extra1'][0])


    # window = 80
    initial = pulse_start
    window = pulse_duration + 250
    pre_spike = 10
    azim = 0
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=tmin, vmax=tmax))
    # # fake up the array of the scalar mappable. Urgh...
    # sm._A = []

    for t in range(initial - pre_spike, initial + window):
        fig = plt.figure(figsize=[6, 6])
        fig.subplots_adjust(wspace=0.6)
        fig.suptitle("Electric field")
        ax1 = plt.subplot(111, projection="3d", title="t = " + ("%.3f" % (t * cell.dt)) + " ms",
                          aspect='auto', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]",
                          # xlim=[-600, 600], ylim=[-600, 600], zlim=[-400, 200])
                          xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])

        for i in range(n_cells):
            col = []
            sm = plt.cm.ScalarMappable(cmap=cmap[i], norm=plt.Normalize(vmin=tmin, vmax=tmax))
            # fake up the array of the scalar mappable. Urgh...
            sm._A = []
            for j in range(cells[i]['totnsegs']):
                col.append((cells[i]['vext'][j][t] + abs(tmin)) / (tmax + abs(tmin)))
            ap_i = cells[i]['extra1'][1]

            [ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]],
                      [cells[i]['ystart'][idx], cells[i]['yend'][idx]],
                      [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                      '-', c=cmap[i](col[idx]), clip_on=False) for idx in range(cells[i]['totnsegs'])]
            # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
            ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
            if ap_i is not None:
                ax1.scatter(cells[i]['xmid'][ap_i], cells[i]['ymid'][ap_i], cells[i]['zmid'][ap_i], '*', c='r')
            ax1.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0] - 20 * i, names[cells[i]['rank']])

            ax1.view_init(elev, azim)
            # plt.colorbar(ax1, min=tmin, max=tmax, label="mV")
            # plt.colorbar(ax1, label="mV")
        plt.colorbar(sm, label="mV", shrink=0.4)

        plt.savefig("outputs/temp/mpi_gif_vext" + str(t) + ".png", dpi=75)
        plt.close()
        print("fig {} out of {}").format(t - initial, window)
        azim += 1

    azim = 0

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-110, vmax=55))
    # # fake up the array of the scalar mappable. Urgh...
    # sm._A = []
    for t in range(initial - pre_spike, initial + window):
        fig = plt.figure(figsize=[6, 6])
        fig.subplots_adjust(wspace=0.6)
        fig.suptitle("Membrane potential")
        ax1 = plt.subplot(111, projection="3d", title="t = " + ("%.3f" % (t * cell.dt)) + " ms",
                          aspect='auto', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]",
                          # xlim=[-600, 600], ylim=[-600, 600], zlim=[-400, 200])
                          xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])
        for i in range(n_cells):

            col = (cells[i]['vmem'].T[t] + 100) / 150.
            ap_i = cells[i]['extra1'][1]
            sm = plt.cm.ScalarMappable(cmap=cmap[i], norm=plt.Normalize(vmin=-100., vmax=50.))
            # fake up the array of the scalar mappable. Urgh...
            sm._A = []

            [ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]],
                      [cells[i]['ystart'][idx], cells[i]['yend'][idx]],
                      [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                      '-', c=cmap[i](col[idx]), clip_on=False) for idx in range(cells[i]['totnsegs'])]
            # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
            ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
            if ap_i is not None:
                ax1.scatter(cells[i]['xmid'][ap_i], cells[i]['ymid'][ap_i], cells[i]['zmid'][ap_i], '*', c='r')
            ax1.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0] - 20 * i, names[cells[i]['rank']])

            ax1.view_init(elev, azim)
        plt.colorbar(sm, label="mV", shrink=.4)

        plt.savefig("outputs/temp/mpi_gif_vmem" + str(t) + ".png", dpi=75)
        plt.close()
        print("fig {} out of {}").format(t - initial, window)
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
