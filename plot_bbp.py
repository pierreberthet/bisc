#!/usr/bin/env python
'''
Test loading and running cell models from the Blue Brain Project Database
'''
import matplotlib
matplotlib.use('Agg')
import os
import posixpath
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import numpy as np
from warnings import warn
import scipy.signal as ss
import neuron
import LFPy
from mpi4py import MPI
import utils
import plotting_convention

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

print("Size {}, Rank {}".format(SIZE, RANK))

base_dir = os.getcwd()

bbp_f = 'morphologies/hoc_combos_syn.1_0_10.allzips_full_axon/'  # set to where the models have been downloaded
NMODL = 'morphologies/hoc_combos_syn.1_0_10.allmods'

FIGS = 'outputs/epfl_column/morphologies_full'
if not os.path.isdir(FIGS):
    os.mkdir(FIGS)


neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

# get names of neuron models, layers options are 'L1', 'L23', 'L4', 'L5' and 'L6'
layer_name = 'L5'
# neuron_type = 'DBC'
# neurons = utils.init_neurons_epfl(layer_name, SIZE, neuron_type)
# neurons = utils.init_neurons_epfl(layer_name, SIZE)
neurons = glob(os.path.join(bbp_f, '*'))
neuron_names = utils.get_epfl_model_name(neurons, short=False)
print("loaded models: {}".format(neuron_names))

# os.chdir(CWD)

# flag for cell template file to switch on (inactive) synapses
add_synapses = False

# load the LFPy SinSyn mechanism for stimulus
# neuron.load_mechanisms(os.path.join(LFPy.__path__[0], "test"))


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


# PARAMETERS
# sim duration
tstop = 3.  # ms
dt = 2**-6

'''
SIMULATION SETUP
pulse duration  should be set to .2 ms, 200 us (typical of empirical in vivo microstimulation experiments)

'''



# spike sampling
threshold = -20  # spike threshold (mV)
samplelength = int(2. / dt)

# filter settings for extracellular traces
b, a = ss.butter(N=3, Wn=(300 * dt * 2 / 1000, 5000 * dt * 2 / 1000), btype='bandpass')
apply_filter = True


n_tsteps = int(tstop / dt + 1)

t = np.arange(n_tsteps) * dt


# Parameters for the external field
name_shape_ecog = 'multipole3'
polarity, n_elec, positions = utils.create_array_shape(name_shape_ecog, 25)
dura_height = 50
displacement_source = 50

source_xs = positions[0]
source_ys = positions[1] + displacement_source
# source_ys = positions[1]
source_zs = positions[2] + dura_height


# communication buffer where all simulation output will be gathered on RANK 0
COMM_DICT = {}
neuron.load_mechanisms(NMODL)

COUNTER = 0
for i, NRN in enumerate(neurons):
    if i % SIZE == RANK:
        # os.chdir(CWD)
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
            # cell.set_pos(z=-1200)

            # v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
            # v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid,
            #                                     cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
            # cell.insert_v_ext(v_cell_ext, t)

            # run simulation
            # cell.simulate(electrode=electrode)
            # cell.simulate(rec_vmem=True, rec_imem=True)
            # print("simulation running ... cell {}".format(RANK))
            # print(utils.spike_segments(cell))
            # plot
            gs = GridSpec(2, 3)
            fig = plt.figure(figsize=(10, 8))
            fig.suptitle(neuron_names[i])

            # morphology
            zips = []
            for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                zips.append(list(zip(x, z)))
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors='k',
                                     rasterized=True)
            ax = plt.subplot(111)
            ax.add_collection(polycol)
            ax.axis(ax.axis('equal'))
            ax.set_title('morphology')
            ax.set_xlabel('(um)', labelpad=0)
            ax.set_ylabel('(um)', labelpad=0)

            # fig.savefig(os.path.join(base_dir, FIGS, neuron_names[i]) + '.png', dpi=300)

            fig.savefig(os.path.join(base_dir, FIGS, os.path.split(NRN)[-1] + '_' +
                        os.path.split(morphologyfile)[-1].replace('.asc', '.png')), dpi=300)

            plt.close(fig)

        COUNTER += 1
        os.chdir(base_dir)

COMM.Barrier()
os.chdir(base_dir)
