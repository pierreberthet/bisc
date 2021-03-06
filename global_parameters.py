'''
Parameters for simulation of neuronal models (bluebrain only?)
'''
import json
import os
# import numpy as np


class parameter(object):
    """
    The parameter class storing the simulation parameters
    is derived from the ParameterContainer class.
    Parameters used (mostly) by different classes should be seperated into
    different functions.
    Common parameters are set in the set_default function
    """

    def __init__(self, args=None, params_fn=None):
        """
        Keyword arguments:
       _fn -- string, if None: set_filenames and set_default will be called
        """

        if params_fn is None:
            self.set_defaults()
            self.set_filenames()
            self.set_figures()

        else:
            self.sim = json.load(open(os.path.join(params_fn, 'simulation_parameters.json'), 'r'))
            self.filename = json.load(open(os.path.join(params_fn, 'simulation_filenames.json'), 'r'))
            self.set_figures()

    def set_defaults(self):

        # SIMULATION ####################################
        self.sim = {}

        self.sim['full_axon'] = True  # use the full axon description of the models, not the stub axon

        self.sim['t_stop'] = 500.  # ms
        self.sim['dt'] = 2**-6  # 1.5 us

        self.sim['pulse_start'] = int(250. / self.sim['dt'] + 1)  # 64 = 1 ms
        self.sim['pulse_duration'] = int(.5 / self.sim['dt'] + 1)   # 64 = 1 ms
        self.sim['buffer'] = int(.01 / self.sim['dt'] + 1)  # disregard what happened before pulse_start - buffer
        self.sim['phase_length'] = 3   # 64 = 1 ms
        self.sim['ampere'] = 100 * 10**3  # uA

        self.sim['total_tsteps'] = int(self.sim['t_stop'] / self.sim['dt'])

        self.sim['ecog_type'] = 'circle2'

        self.sim['spike_threshold'] = -20  # spike threshold (mV)

        self.sim['min_stim_current'] = -300 * 10**3  # uA
        self.sim['max_stim_current'] = 300 * 10**3  # uA
        self.sim['n_intervals'] = 20

        self.sim['max_distance'] = 300

        self.sim['safety_distance_surface_neuron'] = 10

        self.sim['layer'] = 'L5'
        # self.sim['neuron_type'] = 'LBC_cNAC187'
        self.sim['neuron_type'] = ''

    def set_figures(self):

        # FIGURES #####################################
        self.fig = {}
        self.fig['space_between_neurons'] = 1000  # 100

    def set_filenames(self):

        # FILENAME ####################################
        self.filename = {}

        self.filename['short_name'] = False

        self.filename['current_dump'] = 'currents.json'
        self.filename['ap_loc_dump'] = 'ap_loc.json'
        self.filename['c_vext_dump'] = 'c_vext.json'
        self.filename['max_vmem_dump'] = 'max_vmem.json'
        self.filename['t_max_vmem_dump'] = 't_max_vmem.json'
        self.filename['vext_soma_dump'] = 'vext_soma.json'
        self.filename['channels_dump'] = 'channels.json'
        self.filename['spikes_soma_dump'] = 'spikes.json'
        self.filename['vmem_soma_dump'] = 'vmem_soma.json'

        self.filename['model_names'] = 'names.json'

        self.filename['xstart'] = 'xstart.json'
        self.filename['xmid'] = 'xmid.json'
        self.filename['xend'] = 'xend.json'
        self.filename['ystart'] = 'ystart.json'
        self.filename['ymid'] = 'ymid.json'
        self.filename['yend'] = 'yend.json'
        self.filename['zstart'] = 'zstart.json'
        self.filename['zmid'] = 'zmid.json'
        self.filename['zend'] = 'zend.json'

        # self.filename['compilation_folder'] = 'morphologies/hoc_combos_syn.1_0_10.allmods'  # relative path!
        self.filename['compilation_folder'] = '/cluster/projects/nn9272k/pierre/darpa/models/hoc_combos_syn.1_0_10.allmods'  # relative path!

        self.filename['bbp_models_full_axon_folder'] = '/cluster/projects/nn9272k/pierre/darpa/models/hoc_combos_syn.1_0_10.allzips_full_axon'
        self.filename['bbp_models_stub_axon_folder'] = '/cluster/projects/nn9272k/pierre/darpa/models/hoc_combos_syn.1_0_10.allzips'


        # self.filename['bbp_models_full_axon_folder'] = '/media/terror/code/darpa/bisc/morphologies/hoc_combos_syn.1_0_10.allzips_full_axon'
        # self.filename['bbp_models_stub_axon_folder'] = '/media/terror/code/darpa/bisc/morphologies/hoc_combos_syn.1_0_10.allzips'

        # self.filename['output_folder'] = "/media/erebus/oslo/code/darpa/bisc/outputs/"   # local machine path
        self.filename['output_folder'] = "/nird/home/berthetp/outputs/"
        self.filename['simulation_parameters_dump'] = 'simulation_parameters.json'
        self.filename['simulation_filenames_dump'] = 'simulation_filenames.json'

        # filename[''] =