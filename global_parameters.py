'''
Parameters for simulation of neuronal models (bluebrain only?)
'''
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

    def set_defaults(self):

        # SIMULATION ####################################
        self.sim = {}

        self.sim['t_stop'] = 200.
        self.sim['dt'] = 2**-6

        self.sim['pulse_start'] = 80
        self.sim['pulse_duration'] = 50
        self.sim['ampere'] = 100 * 10**3  # uA

        self.sim['spike_threshold'] = -20  # spike threshold (mV)

        self.sim['min_stim_current'] = -300 * 10**3  # uA
        self.sim['max_stim_current'] = 300 * 10**3  # uA
        self.sim['n_intervals'] = 30

        self.sim['max_distance'] = 300

        self.sim['layer'] = 'L5'

    def set_filenames(self):

        # FILENAME ####################################
        self.filename = {}

        self.filename['current_dump'] = 'currents.json'
        self.filename['ap_loc_dump'] = 'ap_loc.json'
        self.filename['c_vext_dump'] = 'c_vext.json'

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

        self.filename['compilation_folder'] = 'morphologies/hoc_combos_syn.1_0_10.allmods'  # relative path!
        self.filename['output_folder'] = "/nird/home/berthetp/outputs/"
        self.filename['simulation_parameters_dump'] = 'simulation_parameters.json'
        self.filename['simulation_filenames_dump'] = 'simulation_filenames.json'

        # filename[''] =
