#!/usr/bin/env python
'''
Simulation of electrical stimulations on neurons.
Determine the threshold of current delivery needed to elicitate an AP on a neuron/axon at various depths.
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
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
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning.hoc')]
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
    morph = join(folder, 'cell_simple.hoc')
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
    morph = join(folder, 'cell_simple.hoc')
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
    'custom_code': custom_code}

COMM.Barrier()

names = ["axon myelin", "soma axon myelin", "dendrite soma axon myelin", "axon nonmyelin"]

cell = LFPy.Cell(**cell_parameters)
# Assign cell positions
# TEST with different distance between cells
x_cell_pos = [-10, 0, 0, 10]
y_cell_pos = [0, -10, 10, 0]
z_cell_pos = np.zeros(len(x_cell_pos))
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
cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])

# cell.set_rotation(x=np.pi/2)
# cell.set_rotation(y=np.pi/2)
# cell.set_rotation(z=np.pi/2)

n_tsteps = int(cell.tstop / cell.dt + 1)

# print("number of segments: ", cell.totnsegs)

t = np.arange(n_tsteps) * cell.dt

pulse_start = 240
pulse_duration = 160
if RANK == 0:
    print("pulse duration: {0} ms".format(pulse_duration * cell.dt))

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 50


# Parameters for the external field
sigma = 0.3
source_xs = np.array([-50, -50, -10, -10, 10, 10, 50, 50])
source_ys = np.array([-50, 50, -10, 10, 10, -10, -50, 50])

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])

# source_amps = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * amp

# while loop? For loop?
spatial_resolution = 20
max_distance = 1000
distance = np.linspace(10, max_distance, spatial_resolution)
current = np.zeros((n_cells, spatial_resolution))
c_vext = np.zeros((n_cells, spatial_resolution))

source_zs = np.ones(len(source_xs)) * distance[0]

# Stimulation Parameters:
max_current = 100000.   # mA
current_resolution = 10
# amp_range = np.exp(np.linspace(1, np.log(max_current), current_resolution))
amp_range = np.linspace(1, max_current, current_resolution)
amp = amp_range[0]
cells = []
glb_vmem = []
glb_vext = []
num = 0

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

    while amp < max_current and not is_spike:

        amp = amp_range[click]
        source_zs = np.ones(len(source_xs)) * depth
        source_amps = source_geometry * amp
        ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)

        # Find external potential field at all cell positions as a function of time
        v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
        v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
                                            cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

        # v_field_ext_stick = np.zeros((len(zs), n_tsteps))
        # v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
        # v_field_ext_stick = v_cell_ext[v_idxs]

        # Insert external potential at cell
        cell = LFPy.Cell(**cell_parameters)
        cell.insert_v_ext(v_cell_ext, t)

        # Run simulation, electrode object argument in cell.simulate
        print("running cell {2} distance from electrode: {0} current intensity: {1}").format(depth, amp, cell_id)
        cell.simulate(rec_imem=True, rec_vmem=True)
        spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
        # COMM.Barrier()
        if spike_time_loc[0] is not None:
            is_spike = True
            current[cell_id][np.where(distance == depth)[0][0]] = amp
            print("spike! at time {0} and position {1}, cell {2} segment {3}".format(spike_time_loc[0],
                  cell.get_idx_name(spike_time_loc[1])[1], cell_id, cell.get_idx_name(spike_time_loc[1])[0]))
            c_vext[cell_id][np.where(distance == depth)] = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
        # glb_vmem.append(cell.vmem)
        # cells.append(cell)

        else:
            click += 1
    COMM.Barrier()
    is_spike = False
COMM.Barrier()

if cell_id == 0:
    gather_current = []
    # plot_current = np.zeros((n_cells, spatial_resolution))
    # print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
    # single_cells = [utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK)]
    gather_current.append({"current": current[cell_id], "v_ext_at_pulse": c_vext[cell_id], "rank": cell_id})
    for i_proc in range(1, n_cells):
        # single_cells = np.r_[single_cells, COMM.recv(source=i_proc)]
        gather_current.append(COMM.recv(source=i_proc))
else:
    # print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
    COMM.send({"current": current[cell_id], "v_ext_at_pulse": c_vext[cell_id], "rank": cell_id}, dest=0)

COMM.Barrier()


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

color = iter(plt.cm.rainbow(np.linspace(0, 1, n_cells)))

if cell_id == 0:
    fig = plt.figure(figsize=[18, 7])
    fig.subplots_adjust(wspace=.6)
    ax = plt.subplot(133, title="Stim threshold")
    # axd = ax.twinx()
    ax.set_xlabel("depth [um]")
    ax.set_ylabel("stimulation current [mA]")
    # axd.set_ylabel("V_Ext [mV]")
    for i in range(n_cells):
        ax.plot(gather_current[i]['current'] / 1000., color=next(color), label=names[i])
        # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
    plt.xticks(np.arange(spatial_resolution), [format(depth, ".0f") for depth in distance])
    plt.legend(loc="upper left")

    ax2 = plt.subplot(131, title="V_ext", xlabel="x [$\mu$um]", ylabel='z [$\mu$m]')
    source_amps = source_geometry * max_current
    ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
    plot_field_length = 400
    v_field_ext_xz = np.zeros((100, 100))
    xf = np.linspace(-plot_field_length, plot_field_length, 100)
    zf = np.linspace(-plot_field_length, cortical_surface_height, 100)
    for xidx, x in enumerate(xf):
        for zidx, z in enumerate(zf):
            v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(x, 0, z)

    vmin = -1000
    vmax = 1000
    logthresh = 0
    imshow_dict = dict(origin='lower', interpolation='nearest',
                       cmap=plt.cm.bwr, vmin=vmin, vmax=vmax,
                       norm=matplotlib.colors.SymLogNorm(10**-logthresh))

    img1 = ax2.imshow(v_field_ext_xz.T,
                      extent=[-plot_field_length, plot_field_length,
                              -plot_field_length, cortical_surface_height],
                      **imshow_dict)
    cax = plt.axes([0.4, 0.1, 0.01, 0.33])
    cb = plt.colorbar(img1, cax=cax, )
    # cb.set_ticks(tick_locations)
    cb.set_label('mV', labelpad=-10)

    ax2.scatter(source_xs, np.ones(len(source_xs)) * cortical_surface_height, c=source_amps, s=100, vmin=-1.4,
                vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

    [ax2.scatter(source_xs[i], np.ones(len(source_xs))[i] * cortical_surface_height,
                 marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)]
    [ax2.scatter(source_xs[i], np.ones(len(source_xs))[i] * cortical_surface_height,
                 marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)]

    ax3 = plt.subplot(132, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
    ax3.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
                edgecolor='k', lw=2, cmap=plt.cm.bwr)
    [ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)]
    [ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)]

    v_field_ext_xy = np.zeros((100, 100))
    xf = np.linspace(-plot_field_length, plot_field_length, 100)
    yf = np.linspace(-plot_field_length, plot_field_length, 100)
    for xidx, x in enumerate(xf):
        for yidx, y in enumerate(yf):
            v_field_ext_xy[xidx, yidx] = ExtPot.ext_field(x, y, cortical_surface_height)

    img2 = ax3.imshow(v_field_ext_xy.T,
                      extent=[-plot_field_length, plot_field_length,
                              -plot_field_length, plot_field_length],
                      **imshow_dict)
    cax = plt.axes([0.4, 0.1, 0.01, 0.33])
    cb = plt.colorbar(img2)
    # cb.set_ticks(tick_locations)
    cb.set_label('mV', labelpad=-10)
    plt.savefig("sweep_test.png")
    plt.show()
# ax2 = plt.subplot(111, title="Cell model", aspect=1, projection='3d',
#                   xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]")
