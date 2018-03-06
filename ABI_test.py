#!/usr/bin/env python
'''
Test script downloading and running cell models from the Allen Brain
Institute's Cell Type Database
'''
import os
import numpy as np
from glob import glob
import LFPy
import matplotlib as mpl
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import neuron
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils
import plotting_convention
try:
    from allensdk.model.biophysical import runner
    from allensdk.api.queries.biophysical_api import BiophysicalApi
except ImportError:
    print('install AllenSDK from http://alleninstitute.github.io/AllenSDK/')


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
for n_idx in range(len(neurons)):
	model = neurons[n_idx]
	# model = neurons[33]
	name = model[model.rfind('/') + 1:]
	os.chdir(model)
	neuron.load_mechanisms('.')

	# load model in neuron using the Allen SDK
	description = runner.load_description('manifest.json')

	# configure NEURON
	r_utils = runner.create_utils(description)
	# h = r_utils.h

	# configure model
	manifest = description.manifest
	morphology_path = description.manifest.get_path('MORPHOLOGY')
	r_utils.load_cell_parameters()

	# r_utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))

	# go back to root folder
	os.chdir(base_dir)
	morphology = morphology_path[morphology_path.find('/') + 1:]
	# load cell into LFPy from NEURON namespace
	cell = LFPy.Cell(morphology=os.path.join(model, morphology), delete_sections=False,
	                 tstart=0, tstop=1000,
	                 passive=False, nsegs_method='lambda_f',
	                 lambda_f=1000.,
	                 extracellular=False, v_init=-100)

	cell.set_rotation(y=np.pi / 2)

	# perform a single sweep
	PointProcParams = {
	                   'idx': cell.get_idx('soma'),
	                   'record_current': True,
	                   'pptype': 'IClamp',
	                   'amp': 0.1,
	                   'dur': 2000,
	                   'delay': 500,
	                    }

	pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

	# run simulation, record input current and soma potential (default)
	cell.simulate(rec_vmem=True, rec_variables=[])

	# plot
	n_sec, names = utils.get_sections_number(cell)


	fig = plt.figure(figsize=[10, 8])
	fig.subplots_adjust(wspace=0.1)
	# 2D view

	ax1 = plt.subplot(131, title="XZ", aspect='auto', xlabel="x [$\mu$m]", ylabel="z [$\mu$m]")
	#          c='k', clip_on=False) for idx in range(cell.totnsegs)]
	# [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
	cmap = plt.cm.viridis
	norm = mpl.colors.Normalize(vmin=-100, vmax=50)
	# col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
	# col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
	colr = plt.cm.Set2(np.arange(n_sec))
	for i, sec in enumerate(names):
	    [ax1.plot([cell.xstart[idx], cell.xend[idx]],
	              [cell.zstart[idx], cell.zend[idx]],
	              '-', c=colr[i], clip_on=False) for idx in cell.get_idx(sec)]
	    if sec != 'soma':
	        ax1.plot([cell.xstart[cell.get_idx(sec)[0]], cell.xend[cell.get_idx(sec)[0]]],
	                 [cell.zstart[cell.get_idx(sec)[0]], cell.zend[cell.get_idx(sec)[0]]],
	                 '-', c=colr[i], clip_on=False, label=sec)
	ax1.scatter(cell.xmid[cell.get_idx('soma')[0]],
			    cell.zmid[cell.get_idx('soma')[0]], s=33, marker='o', c='k', alpha=.7, label='soma')

	ax2 = plt.subplot(132, title="YZ", aspect=1, xlabel="y [$\mu$m]", ylabel="z [$\mu$m]")
	#          c='k', clip_on=False) for idx in range(cell.totnsegs)]
	# [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
	cmap = plt.cm.viridis
	norm = mpl.colors.Normalize(vmin=-100, vmax=50)
	# col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
	# col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
	colr = plt.cm.Set2(np.arange(n_sec))
	for i, sec in enumerate(names):
	    [ax2.plot([cell.ystart[idx], cell.yend[idx]],
	              [cell.zstart[idx], cell.zend[idx]],
	              '-', c=colr[i], clip_on=False) for idx in cell.get_idx(sec)]
	    if sec != 'soma':
	        ax2.plot([cell.ystart[cell.get_idx(sec)[0]], cell.yend[cell.get_idx(sec)[0]]],
	                 [cell.zstart[cell.get_idx(sec)[0]], cell.zend[cell.get_idx(sec)[0]]],
	                 '-', c=colr[i], clip_on=False, label=sec)
	ax2.scatter(cell.ymid[cell.get_idx('soma')[0]],
			    cell.zmid[cell.get_idx('soma')[0]], s=33, marker='o', c='k', alpha=.7, label='soma')

	ax3 = plt.subplot(133, title="XY", aspect=1, xlabel="x [$\mu$m]", ylabel="y [$\mu$m]")
	#          c='k', clip_on=False) for idx in range(cell.totnsegs)]
	# [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
	cmap = plt.cm.viridis
	norm = mpl.colors.Normalize(vmin=-100, vmax=50)
	# col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
	# col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
	colr = plt.cm.Set2(np.arange(n_sec))
	for i, sec in enumerate(names):
	    [ax3.plot([cell.xstart[idx], cell.xend[idx]],
	              [cell.ystart[idx], cell.yend[idx]],
	              '-', c=colr[i], clip_on=False) for idx in cell.get_idx(sec)]
	    if sec != 'soma':
	        ax3.plot([cell.xstart[cell.get_idx(sec)[0]], cell.xend[cell.get_idx(sec)[0]]],
	                 [cell.ystart[cell.get_idx(sec)[0]], cell.yend[cell.get_idx(sec)[0]]],
	                 '-', c=colr[i], clip_on=False, label=sec)
	ax3.scatter(cell.xmid[cell.get_idx('soma')[0]],
			    cell.ymid[cell.get_idx('soma')[0]], s=33, marker='o', c='k', alpha=.7, label='soma')


	plt.legend()
	plt.tight_layout()

		# 3D view
		# ax1 = plt.subplot(111, projection="3d",
		#                   title="", aspect=1, xlabel="x [$\mu$m]",
		#                   ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[-600, 600], ylim=[-600, 600], zlim=[-1800, 200])
		# #          c='k', clip_on=False) for idx in range(cell.totnsegs)]
		# # [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
		# cmap = plt.cm.viridis
		# norm = mpl.colors.Normalize(vmin=-100, vmax=50)
		# # col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
		# # col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
		# colr = plt.cm.Set2(np.arange(n_sec))
		# for i, sec in enumerate(names):
		#     [ax1.plot([cell.xstart[idx], cell.xend[idx]],
		#               [cell.ystart[idx], cell.yend[idx]],
		#               [cell.zstart[idx], cell.zend[idx]],
		#               '-', c=colr[i], clip_on=False) for idx in cell.get_idx(sec)]
		#     if sec != 'soma':
		#         ax1.plot([cell.xstart[cell.get_idx(sec)[0]], cell.xend[cell.get_idx(sec)[0]]],
		#                  [cell.ystart[cell.get_idx(sec)[0]], cell.yend[cell.get_idx(sec)[0]]],
		#                  [cell.zstart[cell.get_idx(sec)[0]], cell.zend[cell.get_idx(sec)[0]]],
		#                  '-', c=colr[i], clip_on=False, label=sec)
		# ax1.scatter(cell.xmid[cell.get_idx('soma')[0]], cell.ymid[cell.get_idx('soma')[0]],
		#             cell.zmid[cell.get_idx('soma')[0]], s=33, marker='o', c='k', alpha=.7, label='soma')

		# plt.legend()
		# # [ax1.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
		# # ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
		# # ax1.scatter(cell.xmid[initial], cell.ymid[initial], cell.zmid[initial], '*', c='r')
		# # for idx in range(cell.totnsegs):
		# #     ax1.text(cell.xmid[idx], cell.ymid[idx], cell.zmid[idx], "{0}.".format(cell.get_idx_name(idx)[1]))

		# elev = 15     # Default 30
		# azim = 45    # Default 0
		# ax1.view_init(elev, azim)


	# ax.axes.set_yticks(yinfo)
	# ax.axes.set_yticklabels(yinfo)
	plt.savefig("outputs/ABI/geo_morph_2D_" + name + ".png", dpi=200)
	plt.close()
	print("neuron {} / {} done").format(n_idx + 1, len(neurons))

plt.show()









# plt.rcParams.update({'font.size': 9.0})
# fig = plt.figure(figsize=(12, 8))
# gs = GridSpec(3, 3)

# ax = fig.add_subplot(gs[:, 0])
# zips = []
# for x, z in cell.get_idx_polygons(projection=('x', 'y')):
#     zips.append(zip(x, z))

# polycol = PolyCollection(zips,
#                          edgecolors='none',
#                          facecolors='k')
# ax.add_collection(polycol)
# ax.axis(ax.axis('equal'))
# ax.set_title('morphology')
# ax.set_ylabel('y (mum)')
# ax.set_xlabel('x (mum)')

# # plot response
# ax = fig.add_subplot(gs[0:2, 1:])
# ax.plot(cell.tvec, cell.somav)
# ax.set_xticks([])
# ax.set_ylabel('V (mV)')
# ax.set_title('somatic response')

# # plot stimulus current
# ax = fig.add_subplot(gs[2, 1:])
# ax.plot(cell.tvec, pointProcess.i)
# ax.set_ylabel('I (nA)')
# ax.set_xlabel('t (ms)')
# ax.set_title('stimulus current')

# plt.show()