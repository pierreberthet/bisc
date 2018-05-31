import json
import numpy as np
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import global_parameters as g_param
import plotting_convention

source_folder = sys.argv[1]

# data_folder = os.path.join('outputs/fram/', source_folder
# print("source folder is {}".format(source_folder))
cwd = os.getcwd()

params = g_param.parameter(params_fn=source_folder)

# params = {}
# params.filename = json.load(open(os.path.join(source_folder, 'simulation_filenames.json'), 'r'))
# param = json.load(open(os.path.join(source_folder, 'simulation_parameteres.json'), 'r'))


os.chdir(source_folder)
# JSON LOAD #############################################################################
currents = json.load(open(params.filename['current_dump'], 'r'))
ap_loc = json.load(open(params.filename['ap_loc_dump'], 'r'))
c_vext = json.load(open(params.filename['c_vext_dump'], 'r'))
max_vmem = json.load(open(params.filename['max_vmem_dump'], 'r'))
t_max_vmem = json.load(open(params.filename['t_max_vmem_dump'], 'r'))
channels = json.load(open(params.filename['channels_dump'], 'r'))

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

min_current = params.sim['min_stim_current']
max_current = params.sim['max_stim_current']
n_intervals = params.sim['n_intervals']

print("DATA LOADED")
os.chdir(cwd)
SIZE = len(currents)

amp_spread = np.linspace(min_current, max_current, n_intervals)

# PROCESSING ###########################################################################

x_comp = np.zeros(SIZE)
y_comp = np.zeros(SIZE)
z_comp = np.zeros(SIZE)
ratio = np.zeros(SIZE)
x_amp = np.zeros(SIZE)
y_amp = np.zeros(SIZE)
z_amp = np.zeros(SIZE)
ratio_amplitude = np.zeros(SIZE)

for neuron in range(SIZE):
    for seg in range(len(xstart[neuron])):
        x_comp[neuron] += np.absolute(xend[neuron][seg] - xstart[neuron][seg])
        y_comp[neuron] += np.absolute(yend[neuron][seg] - ystart[neuron][seg])
        z_comp[neuron] += np.absolute(zend[neuron][seg] - zstart[neuron][seg])
    # print("Neuron {}: total n seg {}      x comp {}, y comp {}, z comp {}\
    #       ".format(neuron_names[neuron], len(xstart[neuron]), x_comp[neuron], y_comp[neuron], z_comp[neuron]))
    x_amp[neuron] = np.absolute(np.min(xend[neuron]) - np.max(xend[neuron]))
    y_amp[neuron] = np.absolute(np.min(yend[neuron]) - np.max(yend[neuron]))
    z_amp[neuron] = np.absolute(np.min(zend[neuron]) - np.max(zend[neuron]))
    ratio_amplitude = ((x_amp + y_amp) / 2.) / z_amp
    ratio[neuron] = (x_comp[neuron] + y_comp[neuron] / 2.) / z_comp[neuron]


# for i in range(SIZE):
#     print("Neuron {} RATIO {} AP {}".format(neuron_names[i], ratio[i], np.count_nonzero(ap_loc[i])))


nn = np.asarray(neuron_names)
unique_type = np.zeros(len(np.unique(nn)))
# for cell_type in np.unique(nn):

# get all unique attributes
channel_list = []
for i in range(SIZE):
    for attribute in channels[str(i)].keys():
        if attribute not in channel_list:
            channel_list.append(attribute)

ap = np.asarray(ap_loc)


# FGIURES #############################################################################

# get the number of AP per channel types:
# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(channel_list))))
xticks = np.zeros(len(np.unique(nn)))

# fig = plt.figure(title='number of channels')

fig = plt.figure()
all_channels = np.zeros(len(channel_list))
by_cell_type = np.zeros(len(np.unique(nn)))
row = 5
column = len(channel_list) // row + 1
for c, chan in enumerate(channel_list):
    ax = plt.subplot(row, column, c + 1, title=chan)
    # color = iter(plt.cm.rainbow(np.linspace(0, 1, len(channel_list))))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

    for ct, cell_type in enumerate(np.unique(nn)):
        col = next(color)

        here = np.argwhere(nn == cell_type)
        i_temp = []  # keep indices of valid neurons (those who contain chan)
        for i in here:
            if chan in channels[str(i[0])].keys():
                i_temp.append(i[0])
                all_channels[c] += channels[str(i[0])][chan]
                # plot the number of channels of channel type == chan, for all individual neurons, ranked by neuron type.
                ax.scatter(i[0], channels[str(i[0])][chan], c=col, label=cell_type)
        # ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean([np.count_nonzero(channels[str(h)][chan]) for h in i_temp]),
        #             yerr=np.std([np.count_nonzero(channels[str(h)][chan]) for h in i_temp]), fmt='x',
        #             color=col, label=cell_type)

        ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean([channels[str(h)][chan] for h in i_temp]),
                    yerr=np.std([channels[str(h)][chan] for h in i_temp]), fmt='x',
                    color=col, label=cell_type)


        xticks[ct] = (here[0][0] + here[-1][0]) // 2

    ax.axes.set_xticks(xticks)
    ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical', 'fontsize': 'xx-small'})

plt.show()


# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(channel_list))))
xticks = np.zeros(len(np.unique(nn)))
# fig = plt.figure(title='number of AP')
fig = plt.figure()
all_channels = np.zeros(len(channel_list))
row = 5
column = len(channel_list) // row + 1
for c, chan in enumerate(channel_list):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

    ax = plt.subplot(row, column, c + 1, title=chan)
    for ct, cell_type in enumerate(np.unique(nn)):
        here = np.argwhere(nn == cell_type)
        col = next(color)
        i_temp = []  # keep indices of valid neurons (those who contain chan)
        for i in here:
            if chan in channels[str(i[0])].keys():
                i_temp.append(i[0])

                all_channels[c] += np.count_nonzero(ap[i[0]])
                # plot the number of channels of channel type == chan, for all individual neurons, ranked by neuron type.
                ax.scatter(i, np.count_nonzero(ap[i[0]]), c=col, label=cell_type)
        ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean([np.count_nonzero(ap[h]) for h in i_temp]),
                    yerr=np.std([np.count_nonzero(ap[h]) for h in i_temp]), fmt='x',
                    color=col, label=cell_type)
        xticks[ct] = (here[0][0] + here[-1][0]) // 2

    ax.axes.set_xticks(xticks)
    ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical', 'fontsize': 'xx-small'})

























plt.show()















# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# fig = plt.figure(figsize=[15, 15])
# fig.subplots_adjust(wspace=.6)
# ax = plt.subplot(331, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("ratio [x,y] / z")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), ratio[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(ratio[here]), yerr=np.std(ratio[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)

# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(332, title="Geometry averaged by neuron types")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("ratio [x,y] / z")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)

#     ax.errorbar(i, np.mean(ratio[here]), yerr=np.std(ratio[here]), fmt='x',
#                 color=next(color), label=cell_type)

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     xticks = np.arange(len(np.unique(nn)))
# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)


# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(333, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("ratio [x,y] / z")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, ratio[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)


# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(334, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("X component")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), x_comp[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(x_comp[here]), yerr=np.std(x_comp[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)

# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(335, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("Y component")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), y_comp[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(y_comp[here]), yerr=np.std(y_comp[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)


# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(336, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("Z component")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), z_comp[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(z_comp[here]), yerr=np.std(z_comp[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)



# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(337, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("X component")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, x_comp[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)
# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(338, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("Y component")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, y_comp[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)
# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(339, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("Z component")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, z_comp[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)


# plt.tight_layout()


# # SPATIAL AMPLITUDE###########################################3


# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# fig = plt.figure(figsize=[15, 15])
# fig.subplots_adjust(wspace=.6)
# ax = plt.subplot(331, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("amplitude ratio [x,y] / z")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), ratio_amplitude[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(ratio_amplitude[here]), yerr=np.std(ratio_amplitude[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# ax.axes.set_ylim(bottom=np.min(ratio_amplitude) * .9, top=np.max(ratio_amplitude) * 1.1)

# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(332, title="Geometry averaged by neuron types")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("amplitude ratio [x,y] / z")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)

#     ax.errorbar(i, np.mean(ratio_amplitude[here]), yerr=np.std(ratio_amplitude[here]), fmt='x',
#                 color=next(color), label=cell_type)

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     xticks = np.arange(len(np.unique(nn)))
# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# ax.axes.set_ylim(bottom=np.min(ratio_amplitude) * .9, top=np.max(ratio_amplitude) * 1.1)


# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(333, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("amplitude ratio [x,y] / z")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, ratio_amplitude[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# ax.axes.set_ylim(bottom=np.min(ratio_amplitude) * .9, top=np.max(ratio_amplitude) * 1.1)


# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(334, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("X amplitude")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), x_amp[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(x_amp[here]), yerr=np.std(x_amp[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)

# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(335, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("Y amplitude")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), y_amp[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(y_amp[here]), yerr=np.std(y_amp[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)


# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(nn)))))

# ax = plt.subplot(336, title="Geometry")
# # axd = ax.twinx()
# ax.set_xlabel("neuron type")
# ax.set_ylabel("Z amplitude")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i, cell_type in enumerate(np.unique(nn)):
#     here = np.argwhere(nn == cell_type)
#     col = next(color)
#     ax.scatter(range(here[0][0], here[-1][0] + 1), z_amp[here],
#                color=col, label=cell_type)
#     ax.errorbar((here[0][0] + here[-1][0]) // 2, np.mean(z_amp[here]), yerr=np.std(z_amp[here]), fmt='x',
#                 color=col, label=cell_type)

#     xticks[i] = (here[0][0] + here[-1][0]) // 2

# ax.axes.set_xticks(xticks)
# ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)



# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(337, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("X amplitude")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, x_amp[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)
# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(338, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("Y amplitude")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, y_amp[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)
# color = iter(plt.cm.rainbow(np.linspace(0, 1, SIZE)))

# ax = plt.subplot(339, title="Geometry and activation")
# # axd = ax.twinx()
# ax.set_xlabel("activation over range of currents [%]")
# ax.set_ylabel("Z amplitude")
# # axd.set_ylabel("V_Ext [mV]")
# xticks = np.zeros(len(np.unique(nn)))
# for i in range(SIZE):

#     ax.scatter(np.count_nonzero(ap_loc[i]) * 100 / n_intervals, z_amp[i], color=next(color))

#     # xticks[i] = (here[0][0] + here[-1][0]) // 2
#     # xticks = np.arange(len(np.unique(nn)))
# # ax.axes.set_xticks(xticks)
# # ax.axes.set_xticklabels(np.unique(nn), fontdict={'rotation': 'vertical'})
# # ax.axes.set_ylim(bottom=np.min(ratio) * .9, top=np.max(ratio) * 1.1)


# plt.tight_layout()












# plt.show()

