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
import plotting_convention

# initialize the MPI interface
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# number of units
n_cells = SIZE
cell_id = RANK

print("cell {0} of {1}").format(cell_id + 1, n_cells)
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
    # folder = "morphologies/cell_hallermann_myelin"
    folder = "morphologies/cell_hallermann_unmyelin"
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
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning.hoc')]

if RANK == 3:
    # folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/L23_PC_cADpyr229_1"
    # folder = "morphologies/dendritic_complexity"
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
    # morph = join(folder, 'morphology.hoc')
    # morph = join(folder, 'altered_complexity_model.hoc')
    # morph = join(folder,'cell1.hoc') # Hay model
    custom_code = [join(folder, 'Cell parameters.hoc'),
                   join(folder, 'charge.hoc'),
                   join(folder, 'pruning.hoc')]

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
    'lambda_f': 300.,
    'dt': 2.**-4,   # [ms] dt's should be in powers of 2 for both,
    'tstart': -50.,    # start time of simulation, recorders start at t=0
    'tstop': 50.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments in cell.simulation()
    "extracellular": True,
    "pt3d": True,
    'custom_code': custom_code}

COMM.Barrier()

# names = ["axon myelin", "neuron myelin", "neuron nonmyelin", "axon nonmyelin"]
names = ["axon myelinated", "axon unmyelinated",
         # "soma+axon myelinated", "soma+axon unmyelinated"]
         "long axon myelinated", "long axon unmyelinated"]


# x_cell_pos = np.zeros(n_cells)
# y_cell_pos = np.linspace(-50, 50, n_cells)
# z_ratio = np.ones(n_cells) * -1.
# z_cell_pos_init = np.multiply(np.ones(len(x_cell_pos)), z_ratio)
# z_cell_pos = z_cell_pos_init

x_cell_pos = [-20, -20, 20, 20]
y_cell_pos = [-20, 20, -20, 20]

n_tsteps = int(cell_parameters['tstop'] / cell_parameters['dt'] + 1)
t = np.arange(n_tsteps) * cell_parameters['dt']
pulse_start = 240
pulse_duration = 160

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 20


# Parameters for the external field
sigma = 0.3

polarity, n_elec, positions = utils.create_array_shape('monopole', 25)
source_xs = positions[0]
source_ys = positions[1]
source_zs = positions[2]

dura_height = 0

# Stimulation Parameters:
# max_current = -5000.   # mA
max_current = 25000.   # mA
current_resolution = 100
# amp_range = np.exp(np.linspace(1, np.log(max_current), current_resolution))
amp_range = np.linspace(10, max_current, current_resolution)
amp = amp_range[0]
if cell_id == 0:
    cells = []
glb_vmem = []
glb_vext = []
num = 0

current = np.zeros(n_cells)
c_vext = np.zeros(n_cells)
ap_loc = np.zeros(n_cells, dtype=np.int)

# WHILE/FOR
click = 0
# is_spike = np.zeros(n_cells)
is_spike = False

while amp < max_current and not is_spike:

    amp = amp_range[click]
    source_amps = np.multiply(polarity, amp)
    ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1], positions[2] + dura_height, sigma)

    # Insert external potential at cell
    cell = LFPy.Cell(**cell_parameters)
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    depth = int(np.abs(np.min(cell.zmid) / 2))
    z_cell_pos = np.ones(n_cells) * depth

    # utils.reposition_stick_horiz(cell, x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])
    cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])

    v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
                                        cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

    cell.insert_v_ext(v_cell_ext, t)

    # Run simulation, electrode object argument in cell.simulate
    print("running cell {2} distance from electrode: {0} current intensity: {1}").format(abs(z_cell_pos[2]),
                                                                                         amp, cell_id)
    cell.simulate(rec_imem=True, rec_vmem=True)

    spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
    # COMM.Barrier()
    if spike_time_loc[0] is not None:
        is_spike = True
        current[cell_id] = amp
        print("spike! at time {0} and position {1}, cell {2} segment {3}".format(spike_time_loc[0],
              cell.get_idx_name(spike_time_loc[1])[1], cell_id, cell.get_idx_name(spike_time_loc[1])[0]))
        c_vext[cell_id] = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
        ap_loc[cell_id] = spike_time_loc[1]
    # glb_vmem.append(cell.vmem)
    # cells.append(cell)

    else:
        click += 1
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
    print("=======================")
    for i in range(n_cells):
        print("min current for AP generation in cell {}  {}: {}, Vext: {}").format(i, names[i],
                                                                                   gather_current[i]['current'],
                                                                                   gather_current[i]['v_ext_at_pulse'])

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
    figview = plt.figure(figsize=[10, 6])
    axview = plt.subplot(111, title="3D view", aspect='auto', projection='3d', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]",
                         zlabel="z [$\mu$m]", xlim=[-200, 200], ylim=[-200, 200], zlim=[-1000, 100])
    for nc in range(0, n_cells):
        [axview.plot([cells[nc]['xstart'][idx], cells[nc]['xend'][idx]], [cells[nc]['ystart'][idx],
                     cells[nc]['yend'][idx]], [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
                     c='k', clip_on=False) for idx in range(cells[nc]['totnsegs'])]
        axview.scatter(cells[nc]['xmid'][gather_current[nc]['ap_loc']], cells[nc]['ymid'][gather_current[nc]['ap_loc']],
                       cells[nc]['zmid'][gather_current[nc]['ap_loc']],
                       '*', c='r')
        axview.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0], names[cells[nc]['rank']])

        # [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]], [cells[nc]['zmid'][idx]], 'D',
        #              c= c_idxs(cells[nc]['v_idxs'].index(idx))) for idx in cells[nc]['v_idxs']]
        # [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]],
        #              [cells[nc]['zmid'][idx]], 'D', c= 'k') for idx in cells[nc]['v_idxs']]
        # ax1.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0],
        #          "cell {0}".format(cells[nc]['rank']))
        # axview.text(cells[nc]['xmid'][v_idxs[widx]], cells[nc]['ymid'][v_idxs[widx]], cells[nc]['zmid'][v_idxs[widx]],
        #             "cell {0}.".format(cells[nc]['rank']) + cells[nc]['name'])

    axview.scatter(source_xs, source_ys, source_zs, c=source_amps)
    plt.tight_layout()
    plt.show()
