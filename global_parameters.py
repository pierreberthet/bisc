'''
Parameters for simulation of neuronal models (bluebrain only?)
'''
# import numpy as np


# SIMULATION ####################################
sim = {}


sim['t_stop'] = 200.
sim['dt'] = 2**-6

sim['pulse_start'] = 80
sim['pulse_duration'] = 50
sim['ampere'] = 100 * 10**3  # uA

sim['spike_threshold'] = -20  # spike threshold (mV)

sim['min_stim_current'] = -300 * 10**3  # uA
sim['max_stim_current'] = 300 * 10**3  # uA
sim['n_intervals'] = 30

sim['max_distance'] = 300

sim['layer'] = 'L5'


# FILENAME ####################################
filename = {}

filename['current_dump'] = 'currents.json'
filename['ap_loc_dump'] = 'ap_loc.json'
filename['c_vext_dump'] = 'c_vext.json'

filename['model_names'] = 'names.json'

filename['xstart'] = 'xstart.json'
filename['xmid'] = 'xmid.json'
filename['xend'] = 'xend.json'
filename['ystart'] = 'ystart.json'
filename['ymid'] = 'ymid.json'
filename['yend'] = 'yend.json'
filename['zstart'] = 'zstart.json'
filename['zmid'] = 'zmid.json'
filename['zend'] = 'zend.json'

filename['compilation_folder'] = 'morphologies/hoc_combos_syn.1_0_10.allmods'  # relative path!
filename['output_folder'] = "/nird/home/berthetp/outputs/"
# filename[''] =
