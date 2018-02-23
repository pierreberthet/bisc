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

print("Size {}, Rank {}").format(SIZE, RANK)


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
NMODL = 'morphologies/hoc_combos_syn.1_0_10.allmods'

# load some required neuron-interface files
neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

# get names of neuron models, layers options are 'L1', 'L23', 'L4', 'L5' and 'L6'
layer_name = 'L5'
neurons = utils.init_neurons_epfl(layer_name, SIZE)
print("loaded models: {}").format(utils.get_epfl_model_name(neurons, short=True))

# flag for cell template file to switch on (inactive) synapses
add_synapses = False

# attempt to set up a folder with all unique mechanism mod files, compile, and
# load them all
if RANK == 0:
    if not os.path.isdir(NMODL):
        os.mkdir(NMODL)
    for NRN in neurons:
        for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
            while not os.path.isfile(os.path.join(NMODL,
                                                  os.path.split(nmodl)[-1])):
                os.system('cp {} {}'.format(nmodl,
                                            os.path.join(NMODL, '.')))
    os.chdir(NMODL)
    os.system('nrnivmodl')
    os.chdir(CWD)
COMM.Barrier()
neuron.load_mechanisms(NMODL)

os.chdir(CWD)

FIGS = 'outputs/epfl_column'
if not os.path.isdir(FIGS):
    os.mkdir(FIGS)


# load the LFPy SinSyn mechanism for stimulus
neuron.load_mechanisms(os.path.join(LFPy.__path__[0], "test"))


# PARAMETERS
# sim duration
tstop = 200.
dt = 2**-6

# output folder
output_f = "/nird/home/bertehtp/outputs/"

'''
SIMULATION SETUP
pulse duration  should be set to .2 ms, 200 us (typical of empirical in vivo microstimulation experiments)

'''

PointProcParams = {'idx': 0,
                   'pptype': 'SinSyn',
                   'delay': 200.,
                   'dur': tstop - 30.,
                   'pkamp': 0.5,
                   'freq': 0.,
                   'phase': np.pi / 2,
                   'bias': 0.,
                   'record_current': False
                   }


# spike sampling
threshold = -20  # spike threshold (mV)
samplelength = int(2. / dt)

n_tsteps = int(tstop / dt + 1)

t = np.arange(n_tsteps) * dt

pulse_start = 80
pulse_duration = 50
amp = 200 * 10**3  # uA

if RANK == 0:
    print("pulse duration: {0} ms ; pulse amplitude: {1} uA".format(pulse_duration * dt, amp))

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.


# TO DETERMINE OR NOT, maybe just start from zmin = - max cortical thickness
cortical_surface_height = 50


# Parameters for the external field
sigma = 0.3
name_shape_ecog = 'circle'
polarity, n_elec, positions = utils.create_array_shape(name_shape_ecog, 25)
dura_height = 50
displacement_source = 50

source_xs = positions[0]
source_ys = positions[1] + displacement_source
# source_ys = positions[1]
source_zs = positions[2] + dura_height


source_amps = np.multiply(polarity, amp)
ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1] + displacement_source,
                                     positions[2] + dura_height, sigma)


# communication buffer where all simulation output will be gathered on RANK 0
COMM_DICT = {}

COUNTER = 0
for i, NRN in enumerate(neurons):
    if RANK == i:
        os.chdir(CWD)
        os.chdir(NRN)

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

        for idx, morphologyfile in enumerate(glob(os.path.join('morphology', '*'))):
            # Instantiate the cell(s) using LFPy
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
            cell.set_pos(z=utils.set_z_layer(layer_name))

            v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
            v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
                                                cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
            cell.insert_v_ext(v_cell_ext, t)

            # pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

            # electrode = LFPy.RecExtElectrode(x = np.array([-40, 40., 0, 0]),
            #                                  y=np.array([0, 0, -40, 40]),
            #                                  z=np.zeros(4),
            #                                  sigma=0.3, r=5, n=50,
            #                                  N=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
            #                                  method='soma_as_point')

            # run simulation
            # cell.simulate(electrode=electrode)
            cell.simulate(rec_vmem=True, rec_imem=True)
            print("simulation running ... cell {}").format(RANK)
            utils.dendritic_spike(cell)

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
COMM.Barrier()

if RANK == 0:
    print("simulation done")
    cells = []
    cells.append(utils.built_for_mpi_space(cell, RANK))
    for i_proc in range(1, SIZE):
        cells.append(COMM.recv(source=i_proc))
else:
    COMM.send(utils.built_for_mpi_space(cell, RANK), dest=0)

COMM.Barrier()

if RANK == 0:
    names = utils.get_epfl_model_name(neurons, short=True)
    font_text = {'family': 'serif',
                 'color': 'black',
                 'weight': 'normal',
                 'size': 13,
                 }
    hbetween = 50
    spread = np.linspace(-hbetween * (SIZE - 1), hbetween * (SIZE - 1), SIZE)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))
    # col = ['b', 'r']
    col = iter(plt.cm.tab10(np.linspace(0, 1, SIZE)))
    figview = plt.figure()
    axview = plt.subplot(111, title="2D view", aspect='auto', xlabel="x [$\mu$m]", ylabel="z [$\mu$m]",
                         xlim=[-750, 750], ylim=[-2500, 100])
    for nc in range(0, SIZE):
        # spread cells along x-axis for a better overview in the 2D view
        cells[nc]['xstart'] += spread[nc]
        cells[nc]['xmid'] += spread[nc]
        cells[nc]['xend'] += spread[nc]
        current_color = color.next()
        [axview.plot([cells[nc]['xstart'][idx], cells[nc]['xend'][idx]],
                     [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
                     c=current_color, clip_on=False) for idx in range(cells[nc]['totnsegs'])]
        axview.scatter(cells[nc]['xmid'][0], cells[nc]['zmid'][0],
                       c=current_color, label=names[nc])
        axview.legend()
    plt.savefig(os.path.join(output_f, "2d_view_XZ.png"), dpi=200)

    figview = plt.figure()
    axview = plt.subplot(111, title="2D view", aspect='auto', xlabel="y [$\mu$m]", ylabel="z [$\mu$m]",
                         xlim=[-750, 750], ylim=[-2500, 100])
    for nc in range(0, SIZE):
        # spread cells along x-axis for a better overview in the 2D view
        current_color = color.next()
        [axview.plot([cells[nc]['ystart'][idx], cells[nc]['yend'][idx]],
                     [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
                     c=current_color, clip_on=False) for idx in range(cells[nc]['totnsegs'])]
        axview.scatter(cells[nc]['ymid'][0], cells[nc]['zmid'][0],
                       c=current_color, label=names[nc])
        axview.legend()
    plt.savefig(os.path.join(output_f, "2d_view_YZ.png"), dpi=200)












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
    plt.show()

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
    os.chdir(CWD)
COMM.Barrier()
