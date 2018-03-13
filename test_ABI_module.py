#!/usr/bin/env python
'''
Test script downloading and running cell models from the Allen Brain
Institute's Cell Type Database
'''
import os
import numpy as np
from glob import glob
import LFPy
import neuron
import matplotlib.pyplot as plt
import utils
import plotting_convention
try:
    from allensdk.model import biophysical as bp
    from allensdk.api.queries.biophysical_api import BiophysicalApi
    from allensdk.model.biophys_sim import neuron as abineuron
    from allensdk.morphology.validate_swc import validate_swc
except ImportError:
    print('install AllenSDK from http://alleninstitute.github.io/AllenSDK/')

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

base_dir = os.getcwd()
working_directory = 'morphologies/ABI/'  # set to where the models have been downloaded


# def get_info_model(spec_id):
# 	info = []
# 	url_spec = 'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::ApiCellTypesSpecimenDetail,\
# 	rma::criteria,[m__biophys_perisomatic$gt0],rma::criteria,[donor__species$il%27homo%20sapiens%27]'

# 	tree = ET.fromstring(urllib.urlopen(url_spec).read())
# 	neurons = []
# 	# here we go through all the results and ask for the neuronal-model
# 	for spec in tree.iter('specimen--id'):
# 	    url_model = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model::NeuronalModel,\
# 	    rma::critera,[specimen_id$eq" + spec.text + "],neuronal_model_template[name$il'*Bio*']"

# 	    neurons.append(ET.fromstring(urllib.urlopen(url_model).read()).iter('id').next().text)
# 	return info

neurons = glob(os.path.join('morphologies/ABI', '*'))
# assert len(neurons) > n_cells, "More threads than available neuron models"
# if len(neurons) < n_threads:
#     print("More threads than available neuron models")
# print("Found {} {} neuron models. Keeping {}.").format(len(neurons), layer, np.min([n_threads, len(neurons)]))
# neurons = neurons[:n_threads]

# neuron.load_mechanisms('.')
valide = []
for n_idx in range(len(neurons)):
	model = neurons[n_idx]
	# model = neurons[33]
	name = model[model.rfind('/') + 1:]
	os.chdir(model)
	neuron.load_mechanisms('.')
	# abineuron.load_mechanisms('.')

	# load model in neuron using the Allen SDK
	description = bp.runner.load_description('manifest.json')

	# configure NEURON
	# r_utils = bp.runner.create_utils(description)
	r_utils = bp.utils.create_utils(description)
	# h = r_utils.h

	# configure model
	manifest = description.manifest
	morphology_path = description.manifest.get_path('MORPHOLOGY')

	r_utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
	r_utils.load_cell_parameters()

	valide.append(validate_swc(morphology_path))
	print("cell {} of {}").format(n_idx + 1, len(neurons))
	# go back to root folder
	os.chdir(base_dir)
	morphology = morphology_path[morphology_path.find('/') + 1:]


# load cell into LFPy from NEURON namespace
# cell = LFPy.Cell(morphology=os.path.join(model, morphology), delete_sections=False,
#                  tstart=0, tstop=1000,
#                  passive=False, nsegs_method='lambda_f',
#                  lambda_f=1000.,
#                  extracellular=False, v_init=-100)

# cell.set_rotation(y=np.pi / 2)

# # perform a single sweep
# PointProcParams = {
#                    'idx': cell.get_idx('soma'),
#                    'record_current': True,
#                    'pptype': 'IClamp',
#                    'amp': 0.1,
#                    'dur': 2000,
#                    'delay': 500,
#                     }

# pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

# # run simulation, record input current and soma potential (default)
# cell.simulate(rec_vmem=True, rec_variables=[])

