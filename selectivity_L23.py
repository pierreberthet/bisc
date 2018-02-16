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

lambdaf = 200.

print("cell {0} of {1}").format(cell_id + 1, n_cells)
# plt.interactive(1)
plt.close('all')
COMM.Barrier()

###############################################################################
# Main script, set parameters and create cell, synapse and electrode objects
###############################################################################
if RANK == 0:
    # folder = "morphologies/cell_hallermann_myelin"
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
                             lambda_f=lambdaf,
                             nsegs_method='lambda_f',)

    custom_code = []
    custom_code = [join(folder, 'morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC'),
                   join(folder, 'biophysics.hoc'),
                   join(folder, 'synapses/synapses.hoc')]

if RANK == 1:
    # folder = "morphologies/cell_hallermann_myelin"
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

COMM.Barrier()

# names = ["axon myelin", "neuron myelin", "neuron nonmyelin", "axon nonmyelin"]
names = ["L2/3 ref", "L2/3 m"]
# names = ["Layer I parallel myelin", "neuron myelin", "neuron unmyelin", "axon unmyelin"]

clamp = False

# cell = LFPy.Cell(**cell_parameters)

COMM.Barrier()

n_tsteps = int(cell_parameters['tstop'] / cell_parameters['dt'] + 1)

t = np.arange(n_tsteps) * cell_parameters['dt']

pulse_start = 240
pulse_duration = 160


if RANK == 0:
    print("pulse duration: {0} ms".format(pulse_duration * cell_parameters['dt']))

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 50


# Parameters for the external field
sigma = 0.3
name_shape_ecog = 'multipole3'
polarity, n_elec, positions = utils.create_array_shape(name_shape_ecog, 25)
# polarity, n_elec, positions = utils.create_array_shape('line', 25)  # low to high current
# polarity, n_elec, positions = utils.create_array_shape('stick', 25)  # very selective
# polarity, n_elec, positions = utils.create_array_shape('twosquare', 25) # low current
# polarity, n_elec, positions = utils.create_array_shape('bcross', 25) # ?


displacement_source = 50
dura_height = 50

source_xs = positions[0]
source_ys = positions[1] + displacement_source
source_zs = positions[2] + dura_height


# source_xs = np.array([-50, -50, -15, -15, 15, 15, 50, 50])
# source_ys = np.array([-50, 50, -15, 15, 15, -15, -50, 50])
# source_xs = np.array([-50, 0, 50, 0, 0])
# source_ys = np.array([0, 50, 0, -50, 0])
# source_zs = np.ones(len(source_xs))

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
# stim_amp = 1.
# n_stim_amp = -stim_amp / 4
# source_geometry = np.array([0, 0, 0, 0, stim_amp])  # monopole
# source_geometry = np.array([-stim_amp, 0, stim_amp, 0, 0])  # dipole
# source_geometry = np.array([-stim_amp / 4, -stim_amp / 4, -stim_amp / 4, -stim_amp / 4, stim_amp])
# source_geometry = np.array([stim_amp, stim_amp, stim_amp, stim_amp, -stim_amp])

# source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])


# source_zs = np.ones(len(source_xs)) * dura_height

# Stimulation Parameters:
max_current = 250. * 10**3   # uA
min_current = 50. * 10**3    # uA
current_resolution = 50
amp_range = np.linspace(min_current, max_current, current_resolution)
amp = amp_range[0]
if cell_id == 0:
    cells = []
glb_vmem = []
glb_vext = []
num = 0


# WHILE/FOR
click = 0
# is_spike = np.zeros(n_cells)
is_spike = False

# Assign cell positions
min_distance = 0
max_distance = 100
num_steps = 50
distance = np.linspace(min_distance, max_distance, num_steps)
print("distances = {}").format(distance)

# while loop? For loop?
current = np.zeros((n_cells, num_steps))
c_vext = np.zeros((n_cells, num_steps))
ap_loc = np.zeros((n_cells, num_steps), dtype=np.int)


x_cell_pos = np.zeros(n_cells)
y_cell_pos = np.zeros(n_cells)
z_cell_pos = np.zeros(n_cells)
# x_cell_pos[1] = distance[0]

for d_idx, depth in enumerate(distance):

    while amp < max_current and not is_spike:

        amp = amp_range[click]
        source_amps = np.multiply(polarity, amp)
        ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1] + displacement_source,
                                             positions[2] + dura_height, sigma)

        # Insert external potential at cell
        # if cell_id == 1:
        cell = LFPy.TemplateCell(morphology=join(folder, "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"),
                                 templatefile=join(folder, 'ActiveModels/model_0603_cell08_cm045.hoc'),
                                 templatename=templatename,
                                 templateargs=join(folder, "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"),
                                 tstop=50.,
                                 dt=2. ** -4,
                                 extracellular=True,
                                 tstart=-50,
                                 lambda_f=lambdaf,
                                 nsegs_method='lambda_f')

        cell.set_rotation(x=-np.pi / 2.)
        cell.set_rotation(y=np.pi / 8.)

        z_cell_pos[cell_id] = -np.max(cell.zend)

        # print("z_cell_pos {}, cell {}").format(z_cell_pos, cell_id)
        if cell_id == 0:
            cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id], z=z_cell_pos[cell_id])
            # utils.reposition_cell_flip(cell)
        else:
            cell.set_pos(x=x_cell_pos[cell_id], y=distance[d_idx], z=z_cell_pos[cell_id])

            print("Y-axis displacement {0}").format(distance[d_idx])

        v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))

        v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
                                            cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

        cell.insert_v_ext(v_cell_ext, t)

        # Run simulation, electrode object argument in cell.simulate
        print("running cell {2} soma distance from electrode: {0} current intensity: {1}").format(z_cell_pos[cell_id],
                                                                                                  amp, names[cell_id])
        cell.simulate(rec_imem=True, rec_vmem=True)
        spike_time_loc = utils.spike_soma(cell)

        # COMM.Barrier()
        if spike_time_loc[0] is not None:
            is_spike = True
            current[cell_id][np.where(distance == depth)[0][0]] = amp
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
    print(gather_current[1]['ap_loc'])

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
    col = ['b', 'r']
    figview = plt.figure(1)
    axview = plt.subplot(111, title="3D view", aspect='auto', projection='3d', xlabel="x [$\mu$m]", ylabel="y [$\mu$m]",
                         zlabel="z [$\mu$m]", xlim=[-750, 750], ylim=[-200, 200], zlim=[-2000, 100])
    for nc in range(0, n_cells):
        [axview.plot([cells[nc]['xstart'][idx], cells[nc]['xend'][idx]], [cells[nc]['ystart'][idx],
                     cells[nc]['yend'][idx]], [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
                     c=col[nc], clip_on=False) for idx in range(cells[nc]['totnsegs'])]
        [axview.scatter(cells[nc]['xmid'][ap], cells[nc]['ymid'][ap], cells[nc]['zmid'][ap],
                        '*', c='k') for ap in gather_current[nc]['ap_loc']]
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

    source_ys -= displacement_source  # to avoid tedious operations (source position / computed field (centered on 0))

    fig = plt.figure(figsize=[18, 7])
    if max_current < 0:
        fig.suptitle("Stimulation threshold as a function of distance and orientation, negative current")
    else:
        fig.suptitle("Stimulation threshold as a function of distance and orientation, positive current")
    fig.subplots_adjust(wspace=.6)
    ax = plt.subplot(133, title="Stim threshold")
    # axd = ax.twinx()
    ax.set_xlabel("distance [$\mu$m]")
    ax.set_ylabel("stimulation current [$\mu$A]")
    # axd.set_ylabel("V_Ext [mV]")
    for i in range(n_cells):
        ax.plot(distance[:len(gather_current[i]['current'].nonzero()[0])],
                gather_current[i]['current'][gather_current[i]['current'].nonzero()[0]] / 1000.,
                color=col[i], label=names[i])
        # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
    # plt.xticks(np.linspace(0, max_distance, 10))
    # plt.locator_params(tight=True)
    if max_current < 0:
        plt.gca().invert_yaxis()
    plt.legend(loc="upper left")

    ax2 = plt.subplot(131, title="V_ext", xlabel="x [$\mu$m]", ylabel='z [$\mu$m]')
    # source_amps = source_geometry * max_current
    source_amps = np.multiply(polarity, amp)
    ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1], positions[2], sigma)
    plot_field_length_v = 1000
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
                       cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax,
                       norm=matplotlib.colors.SymLogNorm(10**-logthresh))

    img1 = ax2.imshow(v_field_ext_xz.T,
                      extent=[-plot_field_length_v, plot_field_length_v,
                              -plot_field_length_v, 0],
                      **imshow_dict)
    # cax = plt.axes([0.4, 0.1, 0.01, 0.33])
    # cb = plt.colorbar(img1)
    # cb.set_ticks(tick_locations)
    # cb.set_label('mV', labelpad=-10)

    ax2.scatter(source_xs, np.zeros(len(source_xs)), c=source_amps, s=100, vmin=-1.4,
                vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

    [ax2.scatter(source_xs[i], np.zeros(len(source_xs))[i],
                 marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]
    [ax2.scatter(source_xs[i], np.zeros(len(source_xs))[i],
                 marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]

    ax3 = plt.subplot(132, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
    ax3.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
                edgecolor='k', lw=2, cmap=plt.cm.bwr)
    [ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]
    [ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]

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
    if max_current > 0:
        plt.savefig("geometry_selectivity_L23_" + name_shape_ecog + "_positive.png", dpi=300)
    else:
        plt.savefig("geometry_selectivity_L23_" + name_shape_ecog + "_negative.png", dpi=300)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_cells)))

    fig = plt.figure(figsize=[10, 7])

    fig.subplots_adjust(wspace=.6)
    ax = plt.subplot(111, title="Stimulation threshold")
    # axd = ax.twinx()
    ax.set_xlabel("distance [$\mu$m]")
    ax.set_ylabel("stimulation current [$\mu$A]")
    # axd.set_ylabel("V_Ext [mV]")
    for i in range(n_cells):
        ax.plot(distance[:len(gather_current[i]['current'].nonzero()[0])],
                gather_current[i]['current'][gather_current[i]['current'].nonzero()[0]] / 1000.,
                color=col[i], label=names[i])
        # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
    # plt.xticks(np.linspace(0, max_distance, 10))
    # plt.locator_params(tight=True)
    if max_current < 0:
        plt.gca().invert_yaxis()
    plt.legend(loc="upper left")
    if max_current > 0:
        plt.savefig("selectivity_L23_" + name_shape_ecog + "_positive.png", dpi=300)
    else:
        plt.savefig("selectivity_L23_" + name_shape_ecog + "_negative.png", dpi=300)
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
