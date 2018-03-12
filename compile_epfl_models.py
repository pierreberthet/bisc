import os
from glob import glob

CWD = os.getcwd()

neurons = glob(os.path.join('morphologies/hoc_combos_syn.1_0_10.allzips', '*'))
mech = 'mechanisms'
for i, NRN in enumerate(neurons):
    os.chdir(CWD)
    os.chdir(NRN)
    os.chdir(mech)
    if os.path.isdir('x86_64'):
        os.system('rm -r x86_64')
    os.system('nrnivmodl')
    print("compiled {} out of {}".format(i + 1, len(neurons)))

os.chdir(CWD)
