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
from mpi4py import MPI

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
    folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc') # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc') # Mainen&Sejnowski, 1996
    # morph = join(folder, 'cell_simple.hoc')
    morph = join(folder, 'cell_simple.hoc')
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning_full.hoc')]

if RANK == 1:
    folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc') # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc') # Mainen&Sejnowski, 1996
    morph = join(folder, 'cell_simple.hoc')
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc')]
                   # join(folder, 'pruning_full.hoc')]
if RANK == 2:
    folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/hay_model"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # neuron.load_mechanisms(join(folder+"/mod")) # Hay model
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder, 'cell_simple.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc'),
    ,join(folder, 'pruning_somaball.hoc')]

if RANK == 3:
    folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/hay_model"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # neuron.load_mechanisms(join(folder+"/mod")) # Hay model
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder, 'cell_simple.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning_somaball.hoc')]

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
    morph = join(folder, 'cell_simple.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc')]
    #             , join(folder, 'pruning.hoc')]


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

COMM.Barrier()

# names = ["axon myelin", "neuron myelin", "neuron nonmyelin", "axon nonmyelin"]
names = ["H axon myelin", "V axon myelin", "H soma dend", "V soma dend"]
# names = ["Layer I parallel myelin", "neuron myelin", "neuron unmyelin", "axon unmyelin"]
clcurrent = True

cell = LFPy.Cell(**cell_parameters)
# Assign cell positions
# TEST with different distance between cells
# x_cell_pos = [-10, 0, 0, 10]
# y_cell_pos = [0, -10, 10, 0]
x_cell_pos = [0, 0, 0, 0]
y_cell_pos = [15, 5, -5, -15]
# z_cell_pos = np.zeros(len(x_cell_pos))
z_ratio = np.array([0, 1, 1, 1]) * -1.
z_cell_pos_init = np.multiply(np.ones(len(x_cell_pos)), z_ratio)
z_cell_pos = z_cell_pos_init
# z_cell_pos = [-1000., -1000 - (np.sum(cell.length)), 0.]

# xrot = [0., 2., 1.]
# cell.set_rotation(y=xrot[cell_id] * np.pi / 2)
# cell.set_rotation(y=np.pi/2)
# cell.set_pos(y=y_cell_pos[cell_id])
# if RANK == 0:
#    cell.set_rotation(x=np.pi/2)
# cell.set_rotation(y=cell_id*np.pi/2)
# cell.set_rotation(z=np.pi/2)
# cell.set_rotation(x=0, y=0, z=z_rotation[cell_id])
# cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id])
if cell_id == 0 or cell_id == 2:
    utils.reposition_stick_horiz(cell, x_cell_pos[cell_id], y_cell_pos[cell_id], z_cell_pos[cell_id])
else:
    cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])
    # utils.reposition_stick_flip(cell, x_cell_pos[cell_id], y_cell_pos[cell_id], z_cell_pos[cell_id])
# cell.set_rotation(x=np.pi/2)
# cell.set_rotation(y=np.pi/2)
# cell.set_rotation(z=np.pi/2)

n_tsteps = int(cell.tstop / cell.dt + 1)

# print("number of segments: ", cell.totnsegs)

t = np.arange(n_tsteps) * cell.dt

pulse_start = 240
pulse_duration = 160

# CLcurrentING 1/2
if clcurrent:
    if cell_id == 0 or cell_id == 1:
        utils.clcurrent_ends(cell, pulse_start, pulse_start + pulse_duration)


if RANK == 0:
    print("pulse duration: {0} ms".format(pulse_duration * cell.dt))

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 50


# Parameters for the external field
sigma = 0.3

polarity, n_elec, positions = utils.create_array_shape('line', 15)
source_xs = positions[0]
source_ys = positions[1]
source_zs = positions[2]

dura_height = 50

# source_xs = np.array([-50, -50, -15, -15, 15, 15, 50, 50])
# source_ys = np.array([-50, 50, -15, 15, 15, -15, -50, 50])
# source_xs = np.array([-50, 0, 50, 0, 0])
# source_ys = np.array([0, 50, 0, -50, 0])
# source_zs = np.ones(len(source_xs))

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
# stim_current = 1.
# n_stim_current = -stim_current / 4
# source_geometry = np.array([0, 0, 0, 0, stim_current])  # monopole
# source_geometry = np.array([-stim_current, 0, stim_current, 0, 0])  # dipole
# source_geometry = np.array([-stim_current / 4, -stim_current / 4, -stim_current / 4, -stim_current / 4, stim_current])
# source_geometry = np.array([stim_current, stim_current, stim_current, stim_current, -stim_current])

# source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])

# while loop? For loop?
spatial_resolution = 10
max_distance = 200
distance = np.linspace(0, max_distance, spatial_resolution)
current = np.zeros((n_cells, spatial_resolution))
c_vext = np.zeros((n_cells, spatial_resolution))
ap_loc = np.zeros((n_cells, spatial_resolution), dtype=np.int)

# source_zs = np.ones(len(source_xs)) * dura_height

# Stimulation Parameters:
max_voltage = 2.5 ** 10**3   # mV
current_resolution = 20
# current_range = np.exp(np.linspace(1, np.log(max_voltage), current_resolution))
current_range = np.linspace(10, max_voltage, current_resolution)
current = current_range[0]
if cell_id == 0:
    cells = []
glb_vmem = []
glb_vext = []
num = 0

# current = (5. * 10**3) / n_elec

# TO IMPROVE
# zs = [int(.95*np.min(cell.zmid)), int(.5*np.min(cell.zmid)),
#       int(.1934*np.min(cell.zmid)), 0, int(.95*np.max(cell.zmid))]

# zs = [-750, -500, -100, 0, 100]
# v_idxs = [cell.get_closest_idx(0, 0, z) for z in zs]
# v_idxs = zs
# v_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(zs)) / (np.max(zs) - np.min(zs)))
# v_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.zend)) / (np.max(np.abs(cell.zmid) - np.min(np.abs(cell.zmid)))))


# WHILE/FOR
click = 0
# is_spike = np.zeros(n_cells)
is_spike = False

for depth in distance:

    while current < max_voltage and not is_spike:

        current = current_range[click]
        source_voltages = np.multiply(polarity, current)
        ExtPot = utils.ImposedPotentialField(source_voltages, positions[0], positions[1],
                                             positions[2] + dura_height, sigma)

        # source_voltages = source_geometry * current
        # ExtPot = utils.ImposedPotentialField(source_voltages, source_xs, source_ys, source_zs, sigma)

        # Find external potential field at all cell positions as a function of time

        # v_field_ext_stick = np.zeros((len(zs), n_tsteps))
        # v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
        # v_field_ext_stick = v_cell_ext[v_idxs]

        # Insert external potential at cell
        cell = LFPy.Cell(**cell_parameters)
        v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))

        # cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])
        z_cell_pos = z_cell_pos_init * depth

        if cell_id == 0:
            utils.reposition_stick_horiz(cell, x_cell_pos[cell_id], y_cell_pos[cell_id], z_cell_pos[cell_id])
        else:
            cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])
            # utils.reposition_stick_flip(cell, x_cell_pos[cell_id], y_cell_pos[cell_id], z_cell_pos[cell_id])
        # CLcurrentING 2/2
        if clcurrent:
            if cell_id == 0 or cell_id == 1:
                utils.clamp_ends(cell, pulse_start, pulse_start + pulse_duration)

        v_cell_ext[:, :] = ExtPot.ext_field_v(cell.xmid, cell.ymid,
                                              cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

        cell.insert_v_ext(v_cell_ext, t)

        # Run simulation, electrode object argument in cell.simulate
        print("running cell {2} distance from electrode: {0} current intensity: {1}").format(abs(z_cell_pos[2]),
                                                                                             current, cell_id)
        cell.simulate(rec_imem=True, rec_vmem=True)
        spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
        # COMM.Barrier()
        if spike_time_loc[0] is not None:
            is_spike = True
            current[cell_id][np.where(distance == depth)[0][0]] = current
            print("spike! at time {0} and position {1}, cell {2} segment {3}".format(spike_time_loc[0],
                  cell.get_idx_name(spike_time_loc[1])[1], cell_id, cell.get_idx_name(spike_time_loc[1])[0]))
            c_vext[cell_id][np.where(distance == depth)] = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
            ap_loc[cell_id][np.where(distance == depth)] = spike_time_loc[1]
        # glb_vmem.append(cell.vmem)
        # cells.append(cell)

        else:
            click += 1
    COMM.Barrier()
    is_spike = False
COMM.Barrier()

if RANK == 0:
    cells.append(utils.built_for_mpi_space(cell, cell_id))
    for i_proc in range(1, SIZE):
        cells.append(COMM.recv(source=i_proc))
else:
    COMM.send(utils.built_for_mpi_space(cell, cell_id), dest=0)

COMM.Barrier()

if cell_id == 0:
    gather_current = []
    # plot_current = np.zeros((n_cells, spatial_resolution))
    # print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
    # single_cells = [utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK)]
    gather_current.append({"current": current[cell_id], "v_ext_at_pulse": c_vext[cell_id],
                          "ap_loc": ap_loc[cell_id], "rank": cell_id})
    for i_proc in range(1, n_cells):
        # single_cells = np.r_[single_cells, COMM.recv(source=i_proc)]
        gather_current.append(COMM.recv(source=i_proc))
else:
    # print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
    COMM.send({"current": current[cell_id], "v_ext_at_pulse": c_vext[cell_id],
              "ap_loc": ap_loc[cell_id], "rank": cell_id}, dest=0)

COMM.Barrier()

if cell_id == 0:
    print gather_current[3]['ap_loc']

# plot 3d view of evolution of vmem as a function of stimulation currentlitude (legend)


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

# ax1.scatter(source_xs, source_ys, source_zs, c=source_voltages)

color = iter(plt.cm.rainbow(np.linspace(0, 1, n_cells)))

if cell_id == 0:
    figview = plt.figure(1)
    axview = plt.subplot(111, title="3D view", aspect=1, projection='3d', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]",
                         zlabel="z [$\mu$m]", xlim=[-1000, 1000], ylim=[-250, 250], zlim=[-500, 500])
    for nc in range(0, n_cells):
        [axview.plot([cells[nc]['xstart'][idx], cells[nc]['xend'][idx]], [cells[nc]['ystart'][idx],
                     cells[nc]['yend'][idx]], [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
                     c='k', clip_on=False) for idx in range(cells[nc]['totnsegs'])]
        [axview.scatter(cells[nc]['xmid'][ap], cells[nc]['ymid'][ap], cells[nc]['zmid'][ap],
                        '*', c='r') for ap in gather_current[nc]['ap_loc']]
        axview.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0], names[cells[nc]['rank']])

        # [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]], [cells[nc]['zmid'][idx]], 'D',
        #              c= c_idxs(cells[nc]['v_idxs'].index(idx))) for idx in cells[nc]['v_idxs']]
        # [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]],
        #              [cells[nc]['zmid'][idx]], 'D', c= 'k') for idx in cells[nc]['v_idxs']]
        # ax1.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0],
        #          "cell {0}".format(cells[nc]['rank']))
        # axview.text(cells[nc]['xmid'][v_idxs[widx]], cells[nc]['ymid'][v_idxs[widx]], cells[nc]['zmid'][v_idxs[widx]],
        #             "cell {0}.".format(cells[nc]['rank']) + cells[nc]['name'])

    axview.scatter(source_xs, source_ys, source_zs, c=source_voltages)

    fig = plt.figure(figsize=[18, 7])
    if max_voltage < 0:
        fig.suptitle("Stimulation threshold as a function of distance and orientation, negative current")
    else:
        fig.suptitle("Stimulation threshold as a function of distance and orientation, positive current")
    fig.subplots_adjust(wspace=.6)
    ax = plt.subplot(133, title="Stim threshold")
    # axd = ax.twinx()
    ax.set_xlabel("depth [$\mu$m]")
    ax.set_ylabel("stimulation current [$\mu$A]")
    # axd.set_ylabel("V_Ext [mV]")
    for i in range(n_cells):
        ax.plot(distance[:len(gather_current[i]['current'].nonzero()[0])],
                gather_current[i]['current'][gather_current[i]['current'].nonzero()[0]] / 1000.,
                color=next(color), label=names[i])
        # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
    # plt.xticks(np.linspace(0, max_distance, 10))
    # plt.locator_params(tight=True)
    if max_voltage < 0:
        plt.gca().invert_yaxis()
    plt.legend(loc="upper left")

    ax2 = plt.subplot(131, title="V_ext", xlabel="x [$\mu$m]", ylabel='z [$\mu$m]')
    # source_voltages = source_geometry * max_voltage
    source_voltages = np.multiply(polarity, current)
    ExtPot = utils.ImposedPotentialField(source_voltages, positions[0], positions[1], positions[2], sigma)
    plot_field_length_v = 500
    plot_field_length_h = 200
    space_resolution = 200
    v_field_ext_xz = np.zeros((space_resolution, space_resolution))
    xf = np.linspace(-plot_field_length_v, plot_field_length_v, space_resolution)
    zf = np.linspace(-plot_field_length_v, 0, space_resolution)
    for xidx, x in enumerate(xf):
        for zidx, z in enumerate(zf):
            v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(x, 0, z)
    v_field_ext_xy = np.zeros((space_resolution, space_resolution))
    xf2 = np.linspace(-plot_field_length_h, plot_field_length_h, space_resolution)
    yf2 = np.linspace(-plot_field_length_h, plot_field_length_h, space_resolution)
    for xidx, x in enumerate(xf2):
        for yidx, y in enumerate(yf2):
            v_field_ext_xy[xidx, yidx] = ExtPot.ext_field(x, y, 0)

    vmin = -100
    vmax = 100
    # vmin = np.min([np.min(v_field_ext_xz), np.min(v_field_ext_xy)])
    # vmax = np.max([np.max(v_field_ext_xz), np.max(v_field_ext_xy)])
    logthresh = 0
    # maxlog = int(np.ceil(np.log10(vmax)))
    # minlog = int(np.ceil(np.log10(-vmin)))
    # tick_locations = ([-(10 ** x) for x in xrange(minlog, -logthresh - 1, -1)] +
    #                   [0.0] + [(10**x) for x in xrange(-logthresh, maxlog + 1)])
    imshow_dict = dict(origin='lower', interpolation='nearest',
                       cmap=plt.cm.inferno, vmin=vmin, vmax=vmax,
                       norm=matplotlib.colors.SymLogNorm(10**-logthresh))

    img1 = ax2.imshow(v_field_ext_xz.T,
                      extent=[-plot_field_length_v, plot_field_length_v,
                              -plot_field_length_v, 0],
                      **imshow_dict)
    # cax = plt.axes([0.4, 0.1, 0.01, 0.33])
    # cb = plt.colorbar(img1)
    # cb.set_ticks(tick_locations)
    # cb.set_label('mV', labelpad=-10)

    ax2.scatter(source_xs, np.zeros(len(source_xs)), c=source_voltages, s=100, vmin=-1.4,
                vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

    [ax2.scatter(source_xs[i], np.zeros(len(source_xs))[i],
                 marker='+', s=50, lw=2, c='k') for i in np.where(source_voltages < 0)]
    [ax2.scatter(source_xs[i], np.zeros(len(source_xs))[i],
                 marker='_', s=50, lw=2, c='k') for i in np.where(source_voltages > 0)]

    ax3 = plt.subplot(132, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
    ax3.scatter(source_xs, source_ys, c=source_voltages, s=100, vmin=-1.4, vmax=1.4,
                edgecolor='k', lw=2, cmap=plt.cm.bwr)
    [ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_voltages > 0)]
    [ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_voltages < 0)]

    img2 = ax3.imshow(v_field_ext_xy.T,
                      extent=[-plot_field_length_h, plot_field_length_h,
                              -plot_field_length_h, plot_field_length_h],
                      **imshow_dict)
    plt.colorbar(img2, label="mV")
    # plt.colorbar(img1, ax=ax2, shrink=0.7, label="mV")
    # plt.colorbar(img2, ax=ax3, shrink=0.7, label="mV")

    # cax = plt.axes([0.335, 0.26, 0.01, 0.45])
    # cb = plt.colorbar(img2, cax=cax)
    # cb.set_ticks(tick_locations)
    # cb.set_label('mV', labelpad=-10)
    plt.savefig("sweep_test_v.png")

    # fig2 = plt.figure(2)
    # axu = fig2.gca(title="Stim threshold")
    # # axd = ax.twinx()
    # axu.set_xlabel("depth [$\mu$m]")
    # axu.set_ylabel("stimulation current [mA]")
    # # axd.set_ylabel("V_Ext [mV]")
    # for i in range(n_cells):
    #     axu.plot(gather_current[i]['current'] / 1000., color=next(color), label=names[i])
    #     # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
    # plt.xticks(np.arange(spatial_resolution), [format(depth, ".0f") for depth in distance])
    # plt.legend(loc="upper left")

    plt.show()
# ax2 = plt.subplot(111, title="Cell model", aspect=1, projection='3d',
#                   xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]")
