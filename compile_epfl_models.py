import os
from glob import glob

CWD = os.getcwd()

NMODL = 'morphologies/hoc_combos_syn.1_0_10.allmods'

neurons = glob(os.path.join('morphologies/hoc_combos_syn.1_0_10.allzips', 'L23*'))
mech = 'mechanisms'

if not os.path.isdir(NMODL):
    os.mkdir(NMODL)
i = 0
for NRN in neurons:
    for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
        while not os.path.isfile(os.path.join(NMODL,
                                 os.path.split(nmodl)[-1])):
            os.system('cp {} {}'.format(nmodl,
                                        os.path.join(NMODL, '.')))
    print("copypasted {} out of {}".format(i + 1, len(neurons)))
    i += 1
os.chdir(NMODL)
os.system('nrnivmodl')
os.chdir(CWD)

# neuron.load_mechanisms(NMODL)



# for i, NRN in enumerate(neurons):
#     os.chdir(CWD)
#     os.chdir(NRN)
#     os.chdir(mech)
#     if os.path.isdir('x86_64'):
#         os.system('rm -r x86_64')
#     os.system('nrnivmodl')
#     print("compiled {} out of {}".format(i + 1, len(neurons)))

# os.chdir(CWD)
