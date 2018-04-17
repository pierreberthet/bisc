import json
import numpy as np
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import global_parameters as g_param

source_folder = sys.argv[1]

# data_folder = os.path.join('outputs/fram/', source_folder
# print("source folder is {}".format(source_folder))
cwd = os.getcwd()

params = g_param.parameter(source_folder)

# params = {}
# params.filename = json.load(open(os.path.join(source_folder, 'simulation_filenames.json'), 'r'))
# param = json.load(open(os.path.join(source_folder, 'simulation_parameteres.json'), 'r'))


os.chdir(source_folder)
# JSON LOAD #############################################################################
currents = json.load(open(params.filename['current_dump'], 'r'))
ap_loc = json.load(open(params.filename['ap_loc_dump'], 'r'))
c_vext = json.load(open(params.filename['c_vext_dump'], 'r'))

neuron_names = json.load(open(params.filename['model_names'], 'r'))

xstart = json.load(open(params.filename['xstart'], 'r'))
xmid = json.load(open(params.filename['xmid'], 'r'))
xend = json.load(open(params.filename['xend'], 'r'))
ystart = json.load(open(params.filename['ystart'], 'r'))
ymid = json.load(open(params.filename['ymid'], 'r'))
yend = json.load(open(params.filename['yend'], 'r'))
zstart = json.load(open(params.filename['zstart'], 'r'))
zmid = json.load(open(params.filename['zmid'], 'r'))
zend = json.load(open(params.filename['zend'], 'r'))

print("DATA LOADED")
os.chdir(cwd)

# PROCESSING ############################################################################
SIZE = len(currents)

amp_spread = np.linspace(params.sim['min_stim_current'], params.sim['max_stim_current'], params.sim['n_intervals'])

# FIGURES ###############################################################################

font_text = {'family': 'serif',
             'color': 'black',
             'weight': 'normal',
             'size': 13,
             }
hbetween = 100

spread = np.linspace(-hbetween * (SIZE - 1), hbetween * (SIZE - 1), SIZE)

# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))
# # col = ['b', 'r']
# # col = iter(plt.cm.tab10(np.linspace(0, 1, SIZE)))

# figview = plt.figure(1)
# axview = plt.subplot(111, title="2D view XZ", aspect='auto', xlabel="x [$\mu$m]", ylabel="z [$\mu$m]")
# for nc in range(0, SIZE):
#     # spread cells along x-axis for a better overview in the 2D view
#     xstart += spread[nc]
#     xmid += spread[nc]
#     xend += spread[nc]
#     current_color = next(color)
#     [axview.plot([xstart[idx], xend[idx]],
#                  [zstart[idx], zend[idx]], '-',
#                  c=current_color, clip_on=False) for idx in range(len(xstart))]
#     axview.scatter(xmid[0], zmid[0],
#                    c=current_color, label=neuron_names[nc])
# art = []
# lgd = axview.legend(loc=9, prop={'size': 6}, bbox_to_anchor=(0.5, -0.1), ncol=6)
# art.append(lgd)
# # plt.savefig(os.path.join(output_f, "2d_view_XZ.png"), additional_artists=art, bbox_inches="tight", dpi=200)
# # plt.close()
# # print("DEBUG 1 rank {}".format(RANK))

# figview = plt.figure(2)
# axview = plt.subplot(111, title="2D view YZ", aspect='auto', xlabel="y [$\mu$m]", ylabel="z [$\mu$m]")

# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))
# for nc in range(0, SIZE):
#     # spread cells along x-axis for a better overview in the 2D view
#     # current_color = color.next()
#     current_color = next(color)

#     [axview.plot([ystart[idx], yend[idx]],
#                  [zstart[idx], zend[idx]], '-',
#                  c=current_color, clip_on=False) for idx in range(len(xstart))]
#     axview.scatter(ymid[0], zmid[0],
#                    c=current_color, label=neuron_names[nc])
#     axview.legend()
# art = []
# lgd = axview.legend(loc=9, prop={'size': 6}, bbox_to_anchor=(0.5, -0.1), ncol=6)
# art.append(lgd)
# # plt.savefig(os.path.join(output_f, "2d_view_YZ.png"), additional_artists=art, bbox_inches="tight", dpi=200)
# # plt.close()
# plt.show()

color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

fig = plt.figure(figsize=[10, 7])
fig.subplots_adjust(wspace=.6)
ax = plt.subplot(111, title="Stimulation threshold")
# axd = ax.twinx()
ax.set_xlabel("stimulation current [$\mu$A]")
ax.set_ylabel("depth [$\mu$m]")
# axd.set_ylabel("V_Ext [mV]")
for i in range(SIZE):
    ax.plot(amp_spread, currents[i],
            color=next(color), label=neuron_names[i])
    # ax.plot(distance[:len(gather_current[i]['current'].nonzero()[0])],
    #         gather_current[i]['current'][gather_current[i]['current'].nonzero()[0]] / 1000.,
    #         color=next(color), label=names[i])
    # axd.plot(gather_current[i]['v_ext_at_pulse'], label="v_ext" + names[i])
# plt.xticks(np.linspace(0, max_distance, 10))
# plt.locator_params(tight=True)
# if max_current < 0:
#     plt.gca().invert_yaxis()
# plt.legend(loc="upper left")
art = []
lgd = ax.legend(loc=9, prop={'size': 6}, bbox_to_anchor=(0.5, -0.1), ncol=6)
art.append(lgd)
# plt.savefig(os.path.join(output_f, "2d_view_YZ.png"),  dpi=200)

# if max_current > 0:
# plt.savefig(os.path.join(output_f, "sensitivity_" + layer_name + '_' + name_shape_ecog +
#             "_" + str(int(min(amp_spread))) + "." + str(int(max(amp_spread))) + ".png",
#             additional_artists=art, bbox_inches="tight", dpi=300))
# else:
#     plt.savefig("sensitivity_" + layer_name + '_' + name_shape_ecog +
#                 "_negative_" + str(min_distance) + "." + str(max_distance) + ".png", dpi=300)

# [axview.scatter(cells[nc]['xmid'][ap], cells[nc]['ymid'][ap], cells[nc]['zmid'][ap],
#                 '*', c='k') for ap in gather_current[nc]['ap_loc']]
# for i, nrn in enumerate(neurons):
#     axview.text(cells[i]['xmid'][0], cells[i]['ymid'][0], cells[i]['zmid'][0], names[i], fontdict=font_text)

# [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]], [cells[nc]['zmid'][idx]], 'D',
#              c= c_idxs(cells[nc]['v_idxs'].index(idx))) for idx in cells[nc]['v_idxs']]
# [axview.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]],
#              [cells[nc]['zmid'][idx]], 'D', c= 'k') for idx in cells[nc]['v_idxs']]
# ax1.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0],
#          "cell {0}".format(cells[nc]['rank']))
# axview.text(cells[nc]['xmid'][v_idxs[widx]], cells[nc]['ymid'][v_idxs[widx]], cells[nc]['zmid'][v_idxs[widx]],
#             "cell {0}.".format(cells[nc]['rank']) + cells[nc]['name'])

# axview.scatter(source_xs, source_ys, source_zs, c=source_amps)
plt.tight_layout()
plt.show()
