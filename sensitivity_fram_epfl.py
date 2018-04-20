import matplotlib
matplotlib.use('Agg')
import os
import posixpath
# import sys
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.collections import PolyCollection, LineCollection
# from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import numpy as np
# from warnings import warn
# import scipy.signal as ss
import neuron
import LFPy
from mpi4py import MPI
import utils
import plotting_convention
import json
import global_parameters as glob_params


# plt.rcParams.update({'axes.labelsize': 8,
#                      'axes.titlesize': 8,
#                      # 'figure.titlesize' : 8,
#                      'font.size': 8,
#                      'ytick.labelsize': 8,
#                      'xtick.labelsize': 8,
#                      })

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


params = glob_params.parameter()

print("Size {}, Rank {}".format(SIZE, RANK))


def posixpth(pth):
    """
    Replace Windows path separators with posix style separators
    """
    return pth.replace(os.sep, posixpath.sep)


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within
    Arguments
    ---------
    f : file, mode 'r'
    Returns
    -------
    templatename : str
    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print('template {} found!'.format(templatename))
            continue
    return templatename


# working dir
CWD = os.getcwd()
compilation_folder = params.filename['compilation_folder']

# load some required neuron-interface files
neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

# get names of neuron models, layers options are 'L1', 'L23', 'L4', 'L5' and 'L6'
layer_name = params.sim['layer']
neuron_type = params.sim['neuron_type']
# neurons = utils.init_neurons_epfl(layer_name, SIZE)
neurons = utils.init_neurons_epfl(layer_name, SIZE, neuron_type)
print("loaded models: {}".format(utils.get_epfl_model_name(neurons, short=True)))

print("REACH")

# flag for cell template file to switch on (inactive) synapses
add_synapses = False

COMM.Barrier()
neuron.load_mechanisms(compilation_folder)

os.chdir(CWD)

FIGS = 'outputs/epfl_column'
if not os.path.isdir(FIGS):
    os.mkdir(FIGS)


# load the LFPy SinSyn mechanism for stimulus
# neuron.load_mechanisms(os.path.join(LFPy.__path__[0], "test"))


# PARAMETERS
# sim duration
tstop = params.sim['t_stop']
dt = params.sim['dt']

# output folder
output_f = params.filename['output_folder']

'''
SIMULATION SETUP
pulse duration  should be set to .2 ms, 200 us (typical of empirical in vivo microstimulation experiments)

'''

# PointProcParams = {'idx': 0,
#                    'pptype': 'SinSyn',
#                    'delay': 200.,
#                    'dur': tstop - 30.,
#                    'pkamp': 0.5,
#                    'freq': 0.,
#                    'phase': np.pi / 2,
#                    'bias': 0.,
#                    'record_current': False
#                    }


# spike sampling
threshold = params.sim['spike_threshold']
# samplelength = int(2. / dt)

n_tsteps = int(tstop / dt + 1)

t = np.arange(n_tsteps) * dt

pulse_start = params.sim['pulse_start']
pulse_duration = params.sim['pulse_duration']
amp = params.sim['ampere']
min_current = params.sim['min_stim_current']
max_current = params.sim['max_stim_current']
n_intervals = params.sim['n_intervals']
amp_spread = np.linspace(min_current, max_current, n_intervals)
# amp_spread = np.geomspace(min_current, max_current, n_intervals)
max_distance = params.sim['max_distance']
distance = np.linspace(0, max_distance, n_intervals)

if RANK == 0:
    print("pulse duration: {0} ms ; pulse amplitude: {1} - {2} uA".format(pulse_duration * dt,
                                                                          np.min(amp_spread), np.max(amp_spread)))

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 50


# Parameters for the external field
sigma = 0.3
name_shape_ecog = params.sim['ecog_type']
polarity, n_elec, positions = utils.create_array_shape(name_shape_ecog, 25)
dura_height = 50
displacement_source = 0

current = np.zeros((SIZE, n_intervals))
c_vext = np.zeros((SIZE, n_intervals))
ap_loc = np.zeros((SIZE, n_intervals), dtype=np.int)
max_vmem = np.zeros((SIZE, n_intervals))
t_max_vmem = np.zeros((SIZE, n_intervals))


source_xs = positions[0]
source_ys = positions[1] + displacement_source
# source_ys = positions[1]
source_zs = positions[2] + dura_height

# communication buffer where all simulation output will be gathered on RANK 0
COMM_DICT = {}

COUNTER = 0
for i, NRN in enumerate(neurons):
    if RANK == i:
        print("DEBUG that is me !!! rank {}".format(RANK))
        # os.chdir(CWD)
        os.chdir(NRN)
        print("Now in {} rank {}".format(NRN, RANK))

        # get the template name
        f = open("template.hoc", 'r')
        templatename = get_templatename(f)
        f.close()

        # get biophys template name
        f = open("biophysics.hoc", 'r')
        biophysics = get_templatename(f)
        f.close()

        # get morphology template name
        f = open("morphology.hoc", 'r')
        morphology = get_templatename(f)
        f.close()

        # get synapses template name
        f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
        synapses = get_templatename(f)
        f.close()

        print('Loading constants')
        neuron.h.load_file('constants.hoc')

        if not hasattr(neuron.h, morphology):
            """Create the cell model"""
            # Load morphology
            neuron.h.load_file(1, "morphology.hoc")
        if not hasattr(neuron.h, biophysics):
            # Load biophysics
            neuron.h.load_file(1, "biophysics.hoc")
        if not hasattr(neuron.h, synapses):
            # load synapses
            neuron.h.load_file(1, posixpth(os.path.join('synapses', 'synapses.hoc')))
        if not hasattr(neuron.h, templatename):
            # Load main cell template
            neuron.h.load_file(1, "template.hoc")
        print("DEBUG_ morphology file {}".format(glob(os.path.join('morphology', '*'))))
        for idx, morphologyfile in enumerate(glob(os.path.join('morphology', '*'))):
            # Instantiate the cell(s) using LFPy
            print("debug idx {} rank {} morph {}".format(idx, RANK, morphologyfile))
            # cell = LFPy.TemplateCell(morphology=morphologyfile,
            #                          templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
            #                          templatename=templatename,
            #                          templateargs=1 if add_synapses else 0,
            #                          tstop=tstop,
            #                          dt=dt,
            #                          extracellular=True,
            #                          nsegs_method=None)

            # # set view as in most other examples
            # cell.set_rotation(x=np.pi / 2)
            # cell.set_pos(z=utils.set_z_layer(layer_name, cell))
            # # cell.set_pos(z=-np.max(cell.zend))
            z_init = utils.set_z_layer(layer_name)
            # spiked = True  # artificially set to True, to engage the loop, but anyway tested for distance = 0
            for i_amp, amp in enumerate(amp_spread):
                dis = distance[0]
                loop = 0
                spiked = True  # artificially set to True, to engage the loop, but anyway tested for distance = 0
                print("debug01 loop {} rank {} amp {} distance {}".format(loop, RANK, amp, dis))

                while spiked and loop <= len(distance):
                    # displacement, amp, loop, PROBLEM
                    dis = distance[loop]
                    print("debug001 loop {} rank {} amp {} distance {}".format(loop, RANK, amp, dis))

                    source_amps = np.multiply(polarity, amp)
                    ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1] + displacement_source,
                                                         positions[2] + dura_height, sigma)
                    LFPy.cell.neuron.init()
                    cell = LFPy.TemplateCell(morphology=morphologyfile,
                                             templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                                             templatename=templatename,
                                             templateargs=1 if add_synapses else 0,
                                             tstop=tstop,
                                             dt=dt,
                                             extracellular=True,
                                             nsegs_method=None)

                    # set view as in most other examples
                    cell.set_rotation(x=np.pi / 2)
                    if np.max((cell.zend) + z_init) > 0:
                        cell.set_pos(z=-np.max(cell.zend) - (dis + params.sim['safety_distance_surface_neuron']))
                    else:
                        cell.set_pos(z=z_init - dis)

                    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
                    v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid
                                                        ).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

                    cell.insert_v_ext(v_cell_ext, t)
                    # print("DEBUG i5 rank {}".format(RANK))

                    # pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

                    # electrode = LFPy.RecExtElectrode(x = np.array([-40, 40., 0, 0]),
                    #                                  y=np.array([0, 0, -40, 40]),
                    #                                  z=np.zeros(4),
                    #                                  sigma=0.3, r=5, n=50,
                    #                                  N=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                    #                                  method='soma_as_point')

                    # run simulation
                    # cell.simulate(electrode=electrode)
                    # print("DEBUG about to simulate rank {}".format(RANK))
                    cell.simulate(rec_vmem=True, rec_imem=True)
                    print("simulation running ... loop {} ; amp {}nA ; distance {}um ; rank {}".format(loop, amp, dis, RANK))
                    # DEBUG
                    if np.isnan(cell.vmem).any():
                        print("NaN for cell {}".format(RANK))

                    utils.dendritic_spike(cell)
                    spike_time_loc = utils.spike_soma(cell)

                    if spike_time_loc[0] is not None:
                        spiked = True

                        detail_spike = utils.return_first_spike_time_and_idx(cell.vmem)
                        print("!spike! at time {0} and position {1}, rank {2}".format(
                            detail_spike[0], cell.get_idx_name(detail_spike[1])[1], RANK))
                        # print("DEBUG 0 rank {}".format(RANK))
                        # current[RANK][np.where(amp_spread == amp)[0][0]] = dis
                        current[RANK][i_amp] = dis
                        # print("DEBUG 1 rank {}".format(RANK))

                        # c_vext[RANK][np.where(amp_spread == amp)[0][0]] = v_cell_ext[detail_spike[1]][detail_spike[0]]
                        c_vext[RANK][i_amp] = v_cell_ext[detail_spike[1]][detail_spike[0]]
                        # print("DEBUG 2 rank {}".format(RANK))
                        # ap_loc[RANK][np.where(amp_spread == amp)[0][0]] = detail_spike[1]
                        ap_loc[RANK][i_amp] = detail_spike[1]
                        # print("DEBUG 3 rank {}".format(RANK))
                        max_vmem[RANK][i_amp] = np.max(cell.vmem[0])
                        t_max_vmem[RANK][i_amp] = np.argmax(cell.vmem[0])
                        print("Max vmem {}, at t {} loop {} rank {}".format(
                              max_vmem[RANK][i_amp], t_max_vmem[RANK][i_amp], loop, RANK))

                    else:
                        spiked = False
                        if loop == 0:
                            max_vmem[RANK][i_amp] = np.max(cell.vmem[0])
                            t_max_vmem[RANK][i_amp] = np.argmax(cell.vmem[0])
                            print("Max vmem {}, at t {}, loop {} rank {}".format(
                                  max_vmem[RANK][i_amp], t_max_vmem[RANK][i_amp], loop, RANK))

                    loop += 1
                    # print('loop {}'.format(loop))
            print("loop: {}, dis: {}, spike: {}, amp: {}, rank: {}".format(loop, dis, spiked, amp, RANK))
#             #electrode.calc_lfp()
#             LFP = electrode.LFP
#             if apply_filter:
#                 LFP = ss.filtfilt(b, a, LFP, axis=-1)

#             #detect action potentials from intracellular trace
#             AP_train = np.zeros(cell.somav.size, dtype=int)
#             crossings = (cell.somav[:-1] < threshold) & (cell.somav[1:] >= threshold)
#             spike_inds = np.where(crossings)[0]
#             #sampled spike waveforms for each event
#             spw = np.zeros((crossings.sum()*LFP.shape[0], 2*samplelength))
#             tspw = np.arange(-samplelength, samplelength)*dt
#             #set spike time where voltage gradient is largest
#             n = 0 #counter
#             for j, i in enumerate(spike_inds):
#                 inds = np.arange(i - samplelength, i + samplelength)
#                 w = cell.somav[inds]
#                 k = inds[:-1][np.diff(w) == np.diff(w).max()][0]
#                 AP_train[k] = 1
#                 #sample spike waveform
#                 for l in LFP:
#                     spw[n, ] = l[np.arange(k - samplelength, k + samplelength)]
#                     n += 1

#             #fill in sampled spike waveforms and times of spikes in comm_dict
#             COMM_DICT.update({
#                 os.path.split(NRN)[-1] + '_' + os.path.split(morphologyfile)[-1].strip('.asc') : dict(
#                     spw = spw,
#                 )
#             })

#             #plot
#             gs = GridSpec(2, 3)
#             fig = plt.figure(figsize=(10, 8))
#             fig.suptitle(NRN + '\n' + os.path.split(morphologyfile)[-1].strip('.asc'))

#             #morphology
#             zips = []
#             for x, z in cell.get_idx_polygons(projection=('x', 'z')):
#                 zips.append(list(zip(x, z)))
#             polycol = PolyCollection(zips,
#                                      edgecolors='none',
#                                      facecolors='k',
#                                      rasterized=True)
#             ax = fig.add_subplot(gs[:, 0])
#             ax.add_collection(polycol)
#             plt.plot(electrode.x, electrode.z, 'ro')
#             ax.axis(ax.axis('equal'))
#             ax.set_title('morphology')
#             ax.set_xlabel('(um)', labelpad=0)
#             ax.set_ylabel('(um)', labelpad=0)

#             #soma potential and spikes
#             ax = fig.add_subplot(gs[0, 1])
#             ax.plot(cell.tvec, cell.somav, rasterized=True)
#             ax.plot(cell.tvec, AP_train*20 + 50)
#             ax.axis(ax.axis('tight'))
#             ax.set_title('soma voltage, spikes')
#             ax.set_ylabel('(mV)', labelpad=0)

#             #extracellular potential
#             ax = fig.add_subplot(gs[1, 1])
#             for l in LFP:
#                 ax.plot(cell.tvec, l, rasterized=True)
#             ax.axis(ax.axis('tight'))
#             ax.set_title('extracellular potential')
#             ax.set_xlabel('(ms)', labelpad=0)
#             ax.set_ylabel('(mV)', labelpad=0)

#             #spike waveform
#             ax = fig.add_subplot(gs[0, 2])
#             n = electrode.x.size
#             for j in range(n):
#                 zips = []
#                 for x in spw[j::n,]:
#                     zips.append(list(zip(tspw, x)))
#                 linecol = LineCollection(zips,
#                                          linewidths=0.5,
#                                          colors=plt.cm.Spectral(int(255.*j/n)),
#                                          rasterized=True)
#                 ax.add_collection(linecol)
#                 #ax.plot(tspw, x, rasterized=True)
#             ax.axis(ax.axis('tight'))
#             ax.set_title('spike waveforms')
#             ax.set_ylabel('(mV)', labelpad=0)

#             #spike width vs. p2p amplitude
#             ax = fig.add_subplot(gs[1, 2])
#             w = []
#             p2p = []
#             for x in spw:
#                 j = x == x.min()
#                 i = x == x[np.where(j)[0][0]:].max()
#                 w += [(tspw[i] - tspw[j])[0]]
#                 p2p += [(x[i] - x[j])[0]]
#             ax.plot(w, p2p, 'o', lw=0.1, markersize=5, mec='none')
#             ax.set_title('spike peak-2-peak time and amplitude')
#             ax.set_xlabel('(ms)', labelpad=0)
#             ax.set_ylabel('(mV)', labelpad=0)

#             fig.savefig(os.path.join(CWD, FIGS, os.path.split(NRN)[-1] + '_' +
#             os.path.split(morphologyfile)[-1].replace('.asc', '.pdf')), dpi=200)
#             plt.close(fig)

#         COUNTER += 1
#         os.chdir(CWD)

# COMM.Barrier()

# #gather sim output
# if SIZE > 1:
#     if RANK == 0:
#         for i in range(1, SIZE):
#             COMM_DICT.update(COMM.recv(source=i, tag=123))
#             print('received from RANK {} on RANK {}'.format(i, RANK))
#     else:
#         print('sent from RANK {}'.format(RANK))
#         COMM.send(COMM_DICT, dest=0, tag=123)
# else:
#     pass
print("i got out! rank {}".format(RANK))
COMM.Barrier()


# DATA DUMP ##########################################################################3


if RANK == 0:
    output_f = os.path.join(output_f, "D_sensitivity_" + layer_name + '_' + name_shape_ecog +
                            "_" + str(int(min(amp_spread))) + "." + str(int(max(amp_spread))))
    try:
        os.mkdir(output_f)
    except OSError:
        duplicate = str(np.random.random_integers(0, 666))
        output_f = output_f + '_' + duplicate
        os.mkdir(output_f)

    for i_proc in range(1, SIZE):
        current[i_proc] = COMM.recv(source=i_proc)
else:
    COMM.send(current[RANK], dest=0)

COMM.Barrier()

output_f = COMM.bcast(output_f, root=0)

COMM.Barrier()

if RANK == 0:
    f_tempdump = params.filename['current_dump']
    # print("DUMPING data JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(current.tolist(), f_dump)
    # print("DEBUG 0 dumping completed")

current = COMM.bcast(current, root=0)

COMM.Barrier()

if RANK == 0:
    for i_proc in range(1, SIZE):
        c_vext[i_proc] = COMM.recv(source=i_proc)
else:
    COMM.send(c_vext[RANK], dest=0)

c_vext = COMM.bcast(c_vext, root=0)

COMM.Barrier()

if RANK == 0:
    f_tempdump = params.filename['c_vext_dump']
    # print("DUMPING c_vext JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(c_vext.tolist(), f_dump)
    # print("DEBUG 1 dumping completed")

COMM.Barrier()

if RANK == 0:
    for i_proc in range(1, SIZE):
        max_vmem[i_proc] = COMM.recv(source=i_proc)
else:
    COMM.send(max_vmem[RANK], dest=0)

max_vmem = COMM.bcast(max_vmem, root=0)

COMM.Barrier()

if RANK == 0:
    f_tempdump = params.filename['max_vmem_dump']
    # print("DUMPING c_vext JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(max_vmem.tolist(), f_dump)
    # print("DEBUG 1 dumping completed")

COMM.Barrier()

if RANK == 0:
    for i_proc in range(1, SIZE):
        t_max_vmem[i_proc] = COMM.recv(source=i_proc)
else:
    COMM.send(t_max_vmem[RANK], dest=0)

t_max_vmem = COMM.bcast(t_max_vmem, root=0)

COMM.Barrier()

if RANK == 0:
    f_tempdump = params.filename['t_max_vmem_dump']
    # print("DUMPING c_vext JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(t_max_vmem.tolist(), f_dump)
    # print("DEBUG 1 dumping completed")

COMM.Barrier()



if RANK == 0:
    for i_proc in range(1, SIZE):
        ap_loc[i_proc] = COMM.recv(source=i_proc)
else:
    COMM.send(ap_loc[RANK], dest=0)

COMM.Barrier()

ap_loc = COMM.bcast(ap_loc, root=0)

names = utils.get_epfl_model_name(neurons, short=True)

if RANK == 0:
    f_tempdump = params.filename['ap_loc_dump']
    # print("DUMPING ap_loc JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(ap_loc.tolist(), f_dump)
    # print("DEBUG 2 dumping completed")

    f_tempdump = params.filename['model_names']
    # print("DUMPING names JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(names, f_dump)
    # print("DEBUG names dumping completed")

    f_tempdump = params.filename['simulation_filenames_dump']
    # print("DUMPING names JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(params.filename, f_dump)

    f_tempdump = params.filename['simulation_parameters_dump']
    # print("DUMPING names JSON to {}".format(f_tempdump))
    with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
        json.dump(params.sim, f_dump)


COMM.Barrier()







# if RANK == 0:

#     gather_current = []
#     # plot_current = np.zeros((SIZE, spatial_resolution))
#     # print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
#     # single_cells = [utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK)]
#     gather_current.append({"current": current[RANK], "v_ext_at_pulse": c_vext[RANK],
#                           "ap_loc": ap_loc[RANK], "rank": RANK})
#     for i_proc in range(1, SIZE):
#         # single_cells = np.r_[single_cells, COMM.recv(source=i_proc)]
#         gather_current.append(COMM.recv(source=i_proc))
# else:
#     # print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
#     COMM.send({"current": current[RANK], "v_ext_at_pulse": c_vext[RANK],
#               "ap_loc": ap_loc[RANK], "rank": RANK}, dest=0)

# COMM.Barrier()


# if RANK == 0 and SIZE > 1:
#     output_f = "D_sensitivity_" + layer_name + '_' + name_shape_ecog +\
#         "_" + str(int(min(amp_spread))) + "." + str(int(max(amp_spread)))
#     print(gather_current[1]['ap_loc'])
#     data_name = ['current', 'v_ext_at_pulse', 'ap_loc', 'names']

#     print("DUMPING data JSON to {}".format(dump_data_filename))
#     with open(dump_data_filename, 'w') as f_dump:
#         json.dump(gather_current.tolist(), f_dump)
#     print("DEBUG dumping completed")


###############################################################

if RANK == 0:
    print("simulation done")
    cells = []
    cells.append(utils.built_for_mpi_space_light(cell, RANK))
    for i_proc in range(1, SIZE):
        cells.append(COMM.recv(source=i_proc))
else:
    COMM.send(utils.built_for_mpi_space_light(cell, RANK), dest=0)

# Traceback (most recent call last):
#   File "sensitivity_fram_epfl.py", line 471, in <module>
#     cells.append(COMM.recv(source=i_proc))
#   File "MPI/Comm.pyx", line 1192, in mpi4py.MPI.Comm.recv (src/mpi4py.MPI.c:106889)
#   File "MPI/msgpickle.pxi", line 292, in mpi4py.MPI.PyMPI_recv (src/mpi4py.MPI.c:43053)
#   File "MPI/msgpickle.pxi", line 143, in mpi4py.MPI.Pickle.load (src/mpi4py.MPI.c:41248)
# _pickle.UnpicklingError: invalid load key, '\x00'.
# SOLVED by removing size of the contained data (stripped vmem and v_ext)


COMM.Barrier()
# print("DEBUG 0 rank {}".format(RANK))

# BROADCAST Data to parallelize figure processing
if RANK != 0:
    cells = []
cells = COMM.bcast(cells, root=0)

print("BROADCAST Succesful")


COMM.Barrier()


if RANK == 0:
    # dump_geo_filename = "D_geo_" + layer_name + '_' + name_shape_ecog +\
    #     "_" + str(int(min(amp_spread))) + "." + str(int(max(amp_spread))) + ".json"
    # print("DUMPING geo JSON to {}".format(dump_geo_filename))
    # with open(dump_geo_filename, 'w') as f_dump:
    #     json.dump(cells.tolist(), f_dump)
    # print("DEBUG dumping completed")
    utils.mpi_dump_geo(cells, SIZE, output_f)

    # f_tempdump = 'geometry.json'
    # print("DUMPING ap_loc JSON to {}".format(f_tempdump))
    # with open(os.path.join(output_f, f_tempdump), 'w') as f_dump:
    #     json.dump(cells.tolist(), f_dump)
    # print("DEBUG 3 dumping completed")






# FIGURES ###############################################################################

#     font_text = {'family': 'serif',
#                  'color': 'black',
#                  'weight': 'normal',
#                  'size': 13,
#                  }
#     hbetween = params.fig['space_between_neurons']
#     spread = np.linspace(-hbetween * (SIZE - 1), hbetween * (SIZE - 1), SIZE)

#     color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))
#     # col = ['b', 'r']
#     # col = iter(plt.cm.tab10(np.linspace(0, 1, SIZE)))

#     figview = plt.figure()
#     axview = plt.subplot(111, title="2D view XZ", aspect='auto', xlabel="x [$\mu$m]", ylabel="z [$\mu$m]")
#     for nc in range(0, SIZE):
#         # spread cells along x-axis for a better overview in the 2D view
#         cells[nc]['xstart'] += spread[nc]
#         cells[nc]['xmid'] += spread[nc]
#         cells[nc]['xend'] += spread[nc]
#         current_color = next(color)
#         [axview.plot([cells[nc]['xstart'][idx], cells[nc]['xend'][idx]],
#                      [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
#                      c=current_color, clip_on=False) for idx in range(cells[nc]['totnsegs'])]
#         axview.scatter(cells[nc]['xmid'][0], cells[nc]['zmid'][0],
#                        c=current_color, label=names[nc])
#     art = []
#     lgd = axview.legend(loc=9, prop={'size': 6}, bbox_to_anchor=(0.5, -0.1), ncol=6)
#     art.append(lgd)
#     plt.savefig(os.path.join(output_f, "2d_view_XZ.png"), additional_artists=art, bbox_inches="tight", dpi=200)
#     plt.close()
#     # print("DEBUG 1 rank {}".format(RANK))

# if RANK == 1:
#     figview = plt.figure()
#     axview = plt.subplot(111, title="2D view YZ", aspect='auto', xlabel="y [$\mu$m]", ylabel="z [$\mu$m]")

#     color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))
#     for nc in range(0, SIZE):
#         # spread cells along x-axis for a better overview in the 2D view
#         # current_color = color.next()
#         current_color = next(color)
#         [axview.plot([cells[nc]['ystart'][idx], cells[nc]['yend'][idx]],
#                      [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
#                      c=current_color, clip_on=False) for idx in range(cells[nc]['totnsegs'])]
#         axview.scatter(cells[nc]['ymid'][0], cells[nc]['zmid'][0],
#                        c=current_color, label=names[nc])
#         axview.legend()
#     art = []
#     lgd = axview.legend(loc=9, prop={'size': 6}, bbox_to_anchor=(0.5, -0.1), ncol=6)
#     art.append(lgd)
#     plt.savefig(os.path.join(output_f, "2d_view_YZ.png"), additional_artists=art, bbox_inches="tight", dpi=200)
#     plt.close()

if RANK == 2:

    color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

    fig = plt.figure(figsize=[10, 7])
    fig.subplots_adjust(wspace=.6)
    ax = plt.subplot(111, title="Stimulation threshold")
    # axd = ax.twinx()
    ax.set_xlabel("stimulation current [$\mu$A]")
    ax.set_ylabel("depth [$\mu$m]")
    # axd.set_ylabel("V_Ext [mV]")
    for i in range(SIZE):
        ax.plot(amp_spread, current[i],
                color=next(color), label=names[i])
        # ax.plot(distance[:len(gather_current[i]['current'].nonzero()[0])],
        #         gather_current[i]['current'][gather_current[i]['current'].nonzero()[0]] / 1000.,
        #         color=next(color), label=names[i])
        # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
    # plt.xticks(np.linspace(0, max_distance, 10))
    # plt.locator_params(tight=True)
    # if max_current < 0:
    #     plt.gca().invert_yaxis()
    # plt.legend(loc="upper left")
    art = []
    lgd = ax.legend(loc=9, prop={'size': 6}, bbox_to_anchor=(0.5, -0.1), ncol=6)
    art.append(lgd)
    # plt.savefig(os.path.join(output_f, "2d_view_YZ.png"),  dpi=200)

    # if max_current > 0:
    plt.savefig(os.path.join(output_f, "sensitivity_" + layer_name + '_' + name_shape_ecog +
                "_" + str(int(min(amp_spread))) + "." + str(int(max(amp_spread))) + ".png"),
                additional_artists=art, bbox_inches="tight", dpi=300)
    # else:
    #     plt.savefig("sensitivity_" + layer_name + '_' + name_shape_ecog +
    #                 "_negative_" + str(min_distance) + "." + str(max_distance) + ".png", dpi=300)

    # [axview.scatter(cells[nc]['xmid'][ap], cells[nc]['ymid'][ap], cells[nc]['zmid'][ap],
    #                 '*', c='k') for ap in gather_current[nc]['ap_loc']]
    # for i, nrn in enumerate(neurons):
    #     axview.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0], names[i], fontdict=font_text)

    # [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]], [cells[nc]['zmid'][idx]], 'D',
    #              c= c_idxs(cells[nc]['v_idxs'].index(idx))) for idx in cells[nc]['v_idxs']]
    # [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]],
    #              [cells[nc]['zmid'][idx]], 'D', c= 'k') for idx in cells[nc]['v_idxs']]
    # ax1.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0],
    #          "cell {0}".format(cells[nc]['rank']))
    # axview.text(cells[nc]['xmid'][v_idxs[widx]], cells[nc]['ymid'][v_idxs[widx]], cells[nc]['zmid'][v_idxs[widx]],
    #             "cell {0}.".format(cells[nc]['rank']) + cells[nc]['name'])

    # axview.scatter(source_xs, source_ys, source_zs, c=source_amps)
    plt.tight_layout()
    # plt.show()

# #project data
# if RANK == 0:
#     fig = plt.figure(figsize=(10, 8))
#     fig.suptitle('spike peak-2-peak time and amplitude')
#     n = electrode.x.size
#     for k in range(n):
#         ax = fig.add_subplot(n, 2, k*2+1)
#         for key, val in COMM_DICT.items():
#             spw = val['spw'][k::n, ]
#             w = []
#             p2p = []
#             for x in spw:
#                 j = x == x.min()
#                 i = x == x[np.where(j)[0][0]:].max()
#                 w += [(tspw[i] - tspw[j])[0]]
#                 p2p += [(x[i] - x[j])[0]]
#             if 'MC' in key:
#                 marker = 'x'
#             elif 'NBC' in key:
#                 marker = '+'
#             elif 'LBC' in key:
#                 marker = 'd'
#             elif 'TTPC' in key:
#                 marker = '^'
#             ax.plot(w, p2p, marker, lw=0.1, markersize=5, mec='none', label=key, alpha=0.25)
#         ax.set_xlabel('(ms)', labelpad=0)
#         ax.set_ylabel('(mV)', labelpad=0)
#         if k == 0:
#             ax.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False, fontsize=7)
#     fig.savefig(os.path.join(CWD, FIGS, 'P2P_time_amplitude.pdf'))
#     print("wrote {}".format(os.path.join(CWD, FIGS, 'P2P_time_amplitude.pdf')))
#     plt.close(fig)
# else:
#     pass
    plt.close()

print("END rank {}".format(RANK))

COMM.Barrier()
if RANK == 0:
    os.chdir(CWD)

COMM.Barrier()
