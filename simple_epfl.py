import os
import sys
import posixpath
import neuron
import LFPy
from glob import glob


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



tstop = 33.  # ms
dt = 2**-6

# for NRN in neurons:
        

NMODL = 'morphologies/hoc_combos_syn.1_0_10.allmods'
CWD = os.getcwd()

NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L1_DAC_bNAC219_1/"
for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
    while not os.path.isfile(os.path.join(NMODL,
                                          os.path.split(nmodl)[-1])):
        os.system('cp {} {}'.format(nmodl,
                                    os.path.join(NMODL, '.')))
    os.chdir(NMODL)
    # os.system('nrnivmodl')
    os.chdir(CWD)
neuron.load_mechanisms(NMODL)

os.chdir(CWD)

# neuron.load_mechanisms(NRN)

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
if not hasattr(neuron.h, templatename):
    # Load main cell template
    neuron.h.load_file(1, "template.hoc")

add_synapses = False

morphologyfile = "morphology/sm090918b1-3_idA_-_Scale_x1.000_y1.025_z1.000.asc"
cell = LFPy.TemplateCell(morphology=morphologyfile,
                         templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                         templatename=templatename,
                         templateargs=1 if add_synapses else 0,
                         tstop=tstop,
                         dt=dt,
                         extracellular=True,
                         nsegs_method=None)
