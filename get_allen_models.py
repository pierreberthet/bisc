#!/usr/bin/env python
'''
Test script downloading and running cell models from the Allen Brain
Institute's Cell Type Database
'''
import os
import numpy as np
import urllib
import xml.etree.ElementTree as ET
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
try:
    # from allensdk.model.biophysical_perisomatic import runner
    from allensdk.api.queries.biophysical_api import BiophysicalApi
except ImportError as ie:
    raise ie, 'install AllenSDK from http://alleninstitute.github.io/AllenSDK/'

# fetch files from the Cell Type Database
# it seems it is not possible to get the neuronal-model data directly, so here are the two setps:
# this calls for all human cells with biophysical models
# For other queries, please visit the Allen brain atlas portal.
# (For some examples: http://help.brain-map.org/pages/viewpage.action?pageId=5308449)


url_spec = 'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::ApiCellTypesSpecimenDetail,\
rma::criteria,[m__biophys_perisomatic$gt0],rma::criteria,[donor__species$il%27homo%20sapiens%27]'

tree = ET.fromstring(urllib.urlopen(url_spec).read())
neurons = []
# here we go through all the results and ask for the neuronal-model
for spec in tree.iter('specimen--id'):
    url_model = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model::NeuronalModel,\
    rma::critera,[specimen_id$eq" + spec.text + "],neuronal_model_template[name$il'*Bio*']"

    neurons.append(ET.fromstring(urllib.urlopen(url_model).read()).iter('id').next().text)

print("Found {0} neurons").format(len(neurons))

base_dir = os.getcwd()
working_directory = 'morphologies/ABI/'  # where the model files will be downloaded
bp = BiophysicalApi('http://api.brain-map.org')
bp.cache_stimulus = False  # set to True to download the large stimulus NWB file

for i, neuron in enumerate(neurons):
    # os.mkdir(os.path.join(working_directory, neuron))
    bp.cache_data(int(neuron), working_directory=os.path.join(working_directory, neuron))

    # change directory
    os.chdir(os.path.join(working_directory, neuron))

    # compile and load NEURON NMODL files
    os.system('nrnivmodl modfiles')
    print("{} mechanism(s) compiled out of {}").format(i + 1, len(neurons))
    os.chdir(base_dir)

# neuronal_model_id = 472451419    # get this from the web site as above
# bp.cache_data(neuronal_model_id, working_directory=working_directory)
