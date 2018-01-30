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
import imageio
import glob

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
    # folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(join(folder))
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    morph = join(folder, '28_04_10_num19.hoc')  # HallermannEtAl2012
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
                             templatefile=join(folder, 'ActiveModels/model_0603_cell08_cm045.hoc'),
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             tstop=50.,
                             dt=2. ** -4,
                             extracellular=True,
                             tstart=-50,
                             lambda_f=1000.,
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
        'lambda_f': 1000.,
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
x_cell_pos = [-50, 0]
y_cell_pos = [-50, 0]
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

polarity, n_elec, positions = utils.create_array_shape('monopole', 25)
source_xs = positions[0]
source_ys = positions[1]
source_zs = positions[2]

# Stimulation Parameters:

amp = 100 * (10**3)
num = 0

# source_zs = np.ones(len(source_xs)) * distance
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
# spike_time_loc = utils.return_first_spike_time_and_idx(cell.vmem)
spike_time_loc = utils.spike_compartments(cell)
# COMM.Barrier()

# if spike_time_loc[0] is not None:
#     print("!!!spike  @  cell {0}").format(cell_id)
#     c_vext = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
# spike_time_loc = utils.spike_soma(cell)
# if spike_time_loc[0] is not None:
#     print("!!!SOMA spike  @  cell {0}").format(cell_id)
#     c_vext = v_cell_ext[spike_time_loc[1]][spike_time_loc[0]]
# print("cell {0} vmem {1}").format(cell_id, cell.vmem.T[spike_time_loc[0]])
vxmin, vxmax = utils.sanity_vext(cell.v_ext, t)

# glb_vmem.append(cell.vmem)
# cells.append(cell)



if not np.isnan(spike_time_loc[0]):
    print("spike in soma, cell {}").format(cell_id + 1)
else:
    print("No somatic spike, cell {}").format(cell_id + 1)
if not np.all(np.isnan(spike_time_loc)):
    print("AP in cell {0}, loc {1} {2} at t= {3}").format(cell_id + 1,
                                                          cell.get_idx_name(np.nanargmin(spike_time_loc))[0],
                                                          cell.get_idx_name(np.nanargmin(spike_time_loc))[1],
                                                          np.nanmin(spike_time_loc))
else:
    print("No spike in cell {}").format(cell_id + 1)

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

xlim_min = -750
xlim_max = 750
ylim_min = -200
ylim_max = 200
zlim_min = -2000
zlim_max = 50

if cell_id == 0:

    # for i in range(n_cells):
    #     if not np.isnan(cells[i]['extra1'][0]):
    #         print("spike in soma, cell {}").format(i + 1)
    #     else:
    #         print("No somatic spike, cell {}").format(i + 1)
    #     if not np.all(np.isnan(cells[i]['extra1'])):
    #         print("AP in cell {0}, loc {1} at t= {2}").format(i, np.argmin(cells[i]['extra1']), np.nanmin(cells[i]['extra1']))
    #     else:
    #         print("No spike in cell {}").format(i + 1)


    # print("Source current = {0} uA").format(amp / 1000.)
    # print("v_ext = {0} mV").format(c_vext)

    fig = plt.figure(figsize=[10, 8])
    fig.subplots_adjust(wspace=0.1)

    time_min = cell.tstop / cell.dt
    time_max = 0
    for i in range(n_cells):
        temp_max = np.nanmax(cells[i]['extra1'])
        temp_min = np.nanmin(cells[i]['extra1'])
        if temp_max > time_max:
            time_max = temp_max
        if temp_min < time_min:
            time_min = temp_min
    print('tmin {}, tmax {}').format(time_min, time_max)
    norm = mpl.colors.Normalize(vmin=time_min, vmax=time_max)

    ax1 = plt.subplot(131, projection="3d", title="Vmem > -20", aspect='auto', xlabel="x [$\mu$m]",
                      ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[xlim_min, xlim_max],
                      ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])

    cmap = plt.cm.viridis
    cmap = [plt.cm.summer_r, plt.cm.winter_r,
            plt.cm.Greens_r, plt.cm.Blues_r, plt.cm.bone, plt.cm.pink, plt.cm.autumn, plt.cm.spring]

    sm1 = plt.cm.ScalarMappable(cmap=cmap[i], norm=plt.Normalize(vmin=time_min * cell.dt, vmax=time_max * cell.dt))
    # fake up the array of the scalar mappable. Urgh...
    sm1._A = []
    for i in range(n_cells):
        # initial = cells[i]['extra'][1]
        # n_sec, names = utils.get_sections_number(cells[i])

        # Find range for activation plots
        # time_max = np.max(spike_time_loc)
        # time_min = np.min(spike_time_loc)

        # for i in spike_time_loc.keys():
        #     if time_max > spike_time_loc[i][0]:
        #         time_max = spike_time_loc[i][0]
        #     if time_min < spike_time_loc[i][0]:
        #         time_min = spike_time_loc[i][0]
        col = np.zeros(cells[i]['totnsegs'])
        for idx in range(cells[i]['totnsegs']):
            # col.append(1. - (spike_time_loc[idx] - time_min) / (time_max - time_min))
            col[idx] = (spike_time_loc[idx] - time_min) / (time_max - time_min)

        # col = (cells[i]['vmem'].T[initial] + 100) / 150.
        # for idx in range(cells[i]['totnsegs']):

            if not np.isnan(cells[i]['extra1'][idx]):
                ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]], [cells[i]['ystart'][idx],
                         cells[i]['yend'][idx]], [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                         # '-', c=cmap[0](cells[i]['extra1'][idx]), clip_on=False)
                         '-', c=cmap[0](col[idx]), clip_on=False)
                         # '-', c='k', clip_on=False)
            else:
                ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]], [cells[i]['ystart'][idx],
                         cells[i]['yend'][idx]], [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                         '-', c='k', clip_on=False)
        if not np.all(np.isnan(cells[i]['extra1'])):
            first = np.nanargmin(spike_time_loc)
            ax1.scatter(cells[i]['xmid'][first], cells[i]['ymid'][first], cells[i]['zmid'][first], c='r', marker='*')
    # if 'axon[8]' in cell.allsecnames:
    #     for node in cell.get_idx('axon[8]')[::5]:
    #         ax1.text(cell.xmid[node], cell.ymid[node], cell.zmid[node], cell.get_idx_name(node))
    plt.colorbar(sm1, label="t [ms]", shrink=0.4)

    ax1.scatter(source_xs, source_ys, source_zs, c=source_amps, cmap=plt.cm.bwr)

    # sm2 = plt.cm.ScalarMappable(cmap=cmap[i], norm=plt.Normalize(vmin=np.min(cells[i]['vmem']), vmax=np.max(cells[i]['vmem'])))
    sm2 = plt.cm.ScalarMappable(cmap=cmap[i], norm=plt.Normalize(vmin=np.min(np.amax(cells[i]['vmem'], 1)), vmax=np.max(cells[i]['vmem'])))
    # fake up the array of the scalar mappable. Urgh...
    sm2._A = []

    ax2 = plt.subplot(132, projection="3d", title="max Vmem",
                      aspect='auto', xlabel="x [$\mu$m]",
                      ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[xlim_min, xlim_max],
                      ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])

    for i in range(n_cells):
        col = []
        for idx in range(cells[i]['totnsegs']):
            col.append((np.max(cells[i]['vmem'][idx]) - np.min(np.amax(cells[i]['vmem'], 1))) / (np.max(cells[i]['vmem']) - np.min(np.amax(cells[i]['vmem'], 1))))

        # col = (cells[i]['vmem'].T[initial] + 100) / 150.
        for idx in range(cells[i]['totnsegs']):
            ax2.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]], [cells[i]['ystart'][idx],
                     cells[i]['yend'][idx]], [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                     # '-', c=cmap[0](cells[i]['extra1'][idx]), clip_on=False)
                     '-', c=cmap[0](col[idx]), clip_on=False)
                     # '-', c='k', clip_on=False)
        if not np.all(np.isnan(cells[i]['extra1'])):
            first = np.nanargmin(spike_time_loc)
            ax2.scatter(cells[i]['xmid'][first], cells[i]['ymid'][first], cells[i]['zmid'][first], c='r', marker='*')
    # fig.colorbar(ax1, label="t [ms]")
    plt.colorbar(sm2, label="[mV]", shrink=0.4)
    ax2.scatter(source_xs, source_ys, source_zs, c=source_amps, cmap=plt.cm.bwr)


    elev = 10     # Default 30
    azim = -90    # Default 0
    ax1.view_init(elev, azim)
    ax2.view_init(elev, azim)

    # ax.axes.set_yticks(yinfo)
    # ax.axes.set_yticklabels(yinfo)
    plt.savefig('order_spike.png', dpi=200)

    plt.show()

COMM.Barrier()
initial = pulse_start
window = pulse_duration + 100
pre_spike = 10

elev = 10     # Default 30
azim = 0

if cell_id == 0:
    vmem_max = 0
    vmem_min = 0
    for i in range(n_cells):
        temp_max = np.nanmax(cells[i]['vmem'])
        temp_min = np.nanmin(cells[i]['vmem'])
        if temp_max > vmem_max:
            vmem_max = temp_max
        if temp_min < vmem_min:
            vmem_min = temp_min
    print('MPI tmin {}, tmax {}').format(time_min, time_max)
    temp = [vmem_min, vmem_max]
else:
    temp = []
COMM.Barrier()

temp = COMM.bcast(temp, root=0)
COMM.Barrier()

if cell_id != 0:
    cells = []
COMM.Barrier()

cells = COMM.bcast(cells, root=0)
COMM.Barrier()

vmin = temp[0]
vmax = temp[1]
diff = vmax - vmin
cmap = [plt.cm.gist_heat]
for t in range(initial - pre_spike, initial + window):
    if t % n_cells == cell_id:
        azim = 1 * (t + pre_spike - initial)
        fig = plt.figure(figsize=[12, 12])
        fig.subplots_adjust(wspace=0.6)
        fig.suptitle("Membrane potential")
        ax1 = plt.subplot(111, projection="3d", title="t = " + ("%.3f" % (t * cell.dt)) + " ms",
                          aspect='auto', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]",
                          # xlim=[-600, 600], ylim=[-600, 600], zlim=[-400, 200])
                          xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])
        for i in range(n_cells):

            col = (cells[i]['vmem'].T[t] + np.abs(vmin)) / diff
            sm = plt.cm.ScalarMappable(cmap=cmap[0], norm=plt.Normalize(vmin=vmin, vmax=vmax))
            # fake up the array of the scalar mappable. Urgh...
            sm._A = []

            [ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]],
                      [cells[i]['ystart'][idx], cells[i]['yend'][idx]],
                      [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
                      '-', c=cmap[0](col[idx]), clip_on=False) for idx in range(cells[i]['totnsegs'])]
            # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
            ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
            # if ap_i is not None:
            #     ax1.scatter(cells[i]['xmid'][ap_i], cells[i]['ymid'][ap_i], cells[i]['zmid'][ap_i], '*', c='r')
            # ax1.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0] - 20 * i, names[cells[i]['rank']])

            ax1.view_init(elev, azim)
        plt.colorbar(sm, label="mV", shrink=.4)

        plt.savefig("outputs/temp/gif_vmem" + str(azim) + ".png", dpi=100)
        plt.close()
        print("fig {} out of {}").format(t - initial, window)

COMM.Barrier()
if cell_id == 0:
    with imageio.get_writer('outputs/gif_vmem.gif', duration=.12, mode='I') as writer:
        for filename in np.sort(glob.glob('outputs/temp/gif_vmem*.png')):
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()




    # for t in range(initial - pre_spike, initial + window):
    #     fig = plt.figure(figsize=[12, 12])
    #     fig.subplots_adjust(wspace=0.6)
    #     fig.suptitle("Membrane potential")
    #     ax1 = plt.subplot(111, projection="3d", title="t = " + ("%.3f" % (t * cell.dt)) + " ms",
    #                       aspect='auto', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]",
    #                       # xlim=[-600, 600], ylim=[-600, 600], zlim=[-400, 200])
    #                       xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max], zlim=[zlim_min, zlim_max])
    #     for i in range(n_cells):

    #         col = (cells[i]['vmem'].T[t] + np.abs(vmin)) / diff
    #         sm = plt.cm.ScalarMappable(cmap=cmap[i], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #         # fake up the array of the scalar mappable. Urgh...
    #         sm._A = []

    #         [ax1.plot([cells[i]['xstart'][idx], cells[i]['xend'][idx]],
    #                   [cells[i]['ystart'][idx], cells[i]['yend'][idx]],
    #                   [cells[i]['zstart'][idx], cells[i]['zend'][idx]],
    #                   '-', c=cmap[i](col[idx]), clip_on=False) for idx in range(cells[i]['totnsegs'])]
    #         # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
    #         ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
    #         # if ap_i is not None:
    #         #     ax1.scatter(cells[i]['xmid'][ap_i], cells[i]['ymid'][ap_i], cells[i]['zmid'][ap_i], '*', c='r')
    #         # ax1.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0] - 20 * i, names[cells[i]['rank']])

    #         ax1.view_init(elev, azim)
    #     plt.colorbar(sm, label="mV", shrink=.4)

    #     plt.savefig("outputs/temp/gif_vmem" + str(t) + ".png", dpi=100)
    #     plt.close()
    #     print("fig {} out of {}").format(t - initial, window)
    #     azim += 1


    # plt.close()

    # cmap = plt.cm.viridis
    # cmap = [plt.cm.autumn, plt.cm.spring]

    # norm = mpl.colors.Normalize(vmin=-110, vmax=55)
