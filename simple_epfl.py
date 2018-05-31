import os
import posixpath
import neuron
import LFPy
from glob import glob
from mpi4py import MPI
import utils
import plotting_convention
import json
import numpy as np
import global_parameters as glob_params
from matplotlib import pyplot as pl


neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

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


params = glob_params.parameter()


# tstop = 33.  # ms
# dt = 2**-6

# for NRN in neurons:
        

# NMODL = 'morphologies/hoc_combos_syn.1_0_10.allmods'
# compilation_folder = params.filename['compilation_folder']
compilation_folder = 'morphologies/hoc_combos_syn.1_0_10.allmods'

CWD = os.getcwd()

# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_SBC_dNAC222_3/"
# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips_full_axon/L23_SBC_dNAC222_3/"

# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips_full_axon/L23_NGC_cSTUT189_4"
# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_NGC_cSTUT189_4"

NRN = "morphologies/hoc_combos_syn.1_0_10.allzips_full_axon/L23_NBC_dNAC222_4"
# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_NBC_dNAC222_4"

# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips_full_axon/L23_NBC_cNAC187_3"
# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_NBC_cNAC187_3"

# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips_full_axon/L23_BP_cNAC187_2"
# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_BP_cNAC187_2"

# L23_NBC_dNAC222_4 L23_NBC_cNAC187_3 L23_BP_cNAC187_2

# NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_SBC_dNAC222_3/"
# for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
#     while not os.path.isfile(os.path.join(compilation_folder,
#                                           os.path.split(nmodl)[-1])):
#         os.system('cp {} {}'.format(nmodl,
#                                     os.path.join(compilation_folder, '.')))
#     os.chdir(compilation_folder)
#     # os.system('nrnivmodl')
#     os.chdir(CWD)
# neuron.load_mechanisms(NMODL)
neuron.load_mechanisms(compilation_folder)


os.chdir(CWD)

# neuron.load_mechanisms(NRN)

# os.chdir(CWD)
os.chdir(NRN)

# PARAMETERS
# sim duration
# tstop = params.sim['t_stop']
tstop = 200.
dt = params.sim['dt']

# output folder
output_f = params.filename['output_folder']

n_tsteps = int(tstop / dt + 1)
t = np.arange(n_tsteps) * dt
pulse = np.zeros(n_tsteps)




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
if not hasattr(neuron.h, templatename):
    # Load main cell template
    neuron.h.load_file(1, "template.hoc")

add_synapses = False

morphologyfile = glob(os.path.join('morphology', '*'))
# os.chdir(CWD)

# morphologyfile = "morphology/sm090918b1-3_idA_-_Scale_x1.000_y1.025_z1.000.asc"
# cell = LFPy.TemplateCell(morphology=morphologyfile,
#                          templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
#                          templatename=templatename,
#                          templateargs=1 if add_synapses else 0,
#                          tstop=tstop,
#                          dt=dt,
#                          extracellular=True,
#                          nsegs_method=None)
LFPy.cell.neuron.init()
print('simulating...')
cell = LFPy.TemplateCell(morphology=morphologyfile[0],
                         templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                         templatename=templatename,
                         templateargs=1 if add_synapses else 0,
                         tstop=tstop,
                         dt=dt,
                         extracellular=True,
                         nsegs_method=None)

# set view as in most other examples
# cell.set_rotation(x=np.pi / 2)
# if np.max((cell.zend) + z_init) > 0:
#     cell.set_pos(z=-np.max(cell.zend) - (dis + params.sim['safety_distance_surface_neuron']))
# else:
#     cell.set_pos(z=z_init - dis)

# source_amps = np.multiply(polarity, amp)
# ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1] + displacement_source,
#                                      positions[2] + dura_height, sigma)
# v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
# v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid
#                                     ).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

# cell.insert_v_ext(v_cell_ext, t)
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
# print("simulation running ... loop {} ; amp {}nA ; distance {}um ; rank {}".format(loop, amp, dis, RANK))
# DEBUG
os.chdir(CWD)

pl.figure()
pl.plot(cell.vmem[0])
pl.show()

neuron.h.delete_section(neuron.h.allsec())

