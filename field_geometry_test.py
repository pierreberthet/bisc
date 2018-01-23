#!/usr/bin/env python
'''
Simulation of electrical stimulations on neurons.
Determine the threshold of current delivery needed to elicitate an AP on a neuron/axon at various depths.
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# from GeneticOptimization import MEAutility as MEA
from os.path import join
import utils
import neuron
import plotting_convention
# Parameters for the external field
sigma = 0.3
# source_xs = np.array([-50, -50, -15, -15, 15, 15, 50, 50])
# source_ys = np.array([-50, 50, -15, 15, 15, -15, -50, 50])
# source_xs = np.array([-50, 0, 50, 0, 0])
# source_ys = np.array([0, 50, 0, -50, 0])
# source_zs = np.ones(len(source_xs))

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
# stim_amp = 1.
# n_stim_amp = -stim_amp / 4
# source_geometry = np.array([0, 0, 0, 0, stim_amp])  # monopole
# source_geometry = np.array([-stim_amp, 0, stim_amp, 0, 0])  # dipole
# source_geometry = np.array([-stim_amp / 4, -stim_amp / 4, -stim_amp / 4, -stim_amp / 4, stim_amp])
# source_geometry = np.array([stim_amp, stim_amp, stim_amp, stim_amp, -stim_amp])

# source_geometry = np.array([-1, -1, 1, 1, 1, 1, -1, -1])

polarity, n_elec, positions = utils.create_array_shape('circle', 25)

amp = (.7 * 10**6) / n_elec  # mA
voltage = 5000

cortical_surface_height = 20

source_amps = np.multiply(polarity, amp)
ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1],
                                     positions[2] + cortical_surface_height, sigma)
x_extent = 500
z_extent = 1000
y_extent = 500
space_resolution = 500
depth_check = -350

# v_field_ext_xz = np.zeros((space_resolution, space_resolution))
# v_field_ext_xy2 = np.zeros((space_resolution, space_resolution))
# xf = np.linspace(-x_extent, x_extent, space_resolution)
# zf = np.linspace(-x_extent * 2, cortical_surface_height, space_resolution)
# for xidx, x in enumerate(xf):
#     for zidx, z in enumerate(zf):
#         v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(x, 0, z)
#         # v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(0, x, z)
# v_field_ext_xy = np.zeros((space_resolution, space_resolution))
# xf2 = np.linspace(-z_extent, z_extent, space_resolution)
# yf2 = np.linspace(-z_extent, z_extent, space_resolution)
# for xidx, x in enumerate(xf2):
#     for yidx, y in enumerate(yf2):
#         v_field_ext_xy[xidx, yidx] = ExtPot.ext_field(x, y, cortical_surface_height)

# xf3 = np.linspace(-x_extent, x_extent, space_resolution)
# yf3 = np.linspace(-x_extent, x_extent, space_resolution)
# for xidx, x in enumerate(xf3):
#     for yidx, y in enumerate(yf3):
#         v_field_ext_xy2[xidx, yidx] = ExtPot.ext_field(x, y, depth_check)

v_field_ext_xz, dd_field_xz = utils.external_field(ExtPot, space_resolution=space_resolution, x_extent=x_extent,
											   y_extent=None, z_extent=z_extent, z_top=cortical_surface_height,
											   axis='xz', dderivative=True, plan=None)

v_field_ext_xy, dd_field_xy = utils.external_field(ExtPot, space_resolution=space_resolution, x_extent=x_extent,
											   y_extent=x_extent, z_extent=None, z_top=cortical_surface_height,
											   axis='xy', dderivative=True, plan=None)

v_field_ext_yz, dd_field_yz = utils.external_field(ExtPot, space_resolution=space_resolution, x_extent=x_extent, y_extent=y_extent,
											   z_extent=z_extent, z_top=cortical_surface_height,
											   axis='yz', dderivative=True, plan=None)

v_field_ext_xy2 = utils.external_field(ExtPot, space_resolution=space_resolution, x_extent=x_extent, y_extent=y_extent, z_extent=None,
								   z_top=cortical_surface_height, axis='xy', dderivative=False, plan=depth_check)



# FIGURES

fig = plt.figure(figsize=[18, 7])
fig.suptitle("External Potential")

ax1 = plt.subplot(131, title="V_ext", xlabel="x [$\mu$m]", ylabel='z [$\mu$m]')
vmin = -2500
vmax = 2500
# vmin = np.min([np.min(v_field_ext_xz), np.min(v_field_ext_xy)])
# vmax = np.max([np.max(v_field_ext_xz), np.max(v_field_ext_xy)])
logthresh = 0
# maxlog = int(np.ceil(np.log10(vmax)))
# minlog = int(np.ceil(np.log10(-vmin)))
# tick_locations = ([-(10 ** x) for x in xrange(minlog, -logthresh - 1, -1)] +
#                   [0.0] + [(10**x) for x in xrange(-logthresh, maxlog + 1)])
imshow_dict = dict(origin='lower', interpolation='nearest',
                   cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax,
                   norm=matplotlib.colors.SymLogNorm(10**-logthresh))

# cax = plt.axes([0.4, 0.1, 0.01, 0.33])
# cb = plt.colorbar(img1)
# cb.set_ticks(tick_locations)
# cb.set_label('mV', labelpad=-10)
source_xs = positions[0]
source_ys = positions[1]

img1 = ax1.imshow(v_field_ext_xz.T,
                  extent=[-x_extent, x_extent,
                          -z_extent, cortical_surface_height],
                  **imshow_dict)


ax1.scatter(source_xs, np.ones(len(source_xs)) * cortical_surface_height, c=source_amps, s=100, vmin=-1.4,
            vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

[ax1.scatter(source_xs[i], cortical_surface_height,
             marker='+', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]
[ax1.scatter(source_xs[i], cortical_surface_height,
             marker='_', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]

ax2 = plt.subplot(132, title="V_ext at z = " + str(depth_check) + " $\mu$m", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
img2 = ax2.imshow(v_field_ext_xy2.T,
                  extent=[-x_extent, x_extent,
                          -y_extent, y_extent],
                  **imshow_dict)
# ax2.scatter(source_xs, np.ones(len(source_xs)) * -300, c=source_amps, s=100, vmin=-1.4,
#             vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)


ax3 = plt.subplot(133, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
ax3.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
            edgecolor='k', lw=2, cmap=plt.cm.bwr)
[ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]
[ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]

img3 = ax3.imshow(v_field_ext_xy.T,
                  extent=[-x_extent, x_extent,
                          -y_extent, y_extent],
                  **imshow_dict)
plt.colorbar(img3, label="mV")
plt.savefig("electrode_geometry_test.png")

plt.show()













# img1 = ax1.imshow(v_field_ext_xz.T,
# [ax1.scatter(source_xs[i], source_zs[i],
#              marker='_', s='o' in np.where(source_amps < 0)[0]]
#                   **imshow_dict)


# ax1.scatter(source_xs, np.ones(len(source_xs)) * cortical_surface_height, c=source_amps, s=100, vmin=-1.4,
#             vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

# [ax1.scatter(source_xs[i], cortical_surface_height,
#              marker='+', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]
# [ax1.scatter(source_xs[i], cortical_surface_height,
#              marker='_', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]

# ax2 = plt.subplot(132, title="V_ext at z = " + str(depth_check) + " $\mu$m", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
# img2 = ax2.imshow(v_field_ext_xy2.T,
# [ax3.scatter(source_xs[i], x_extentr=x_extent'k') for i in np.where(source_amps > 0)[0]]
# [ax3.scatter(source_xs[i], x_extentr=x_extent'k') for i in np.where(source_amps < 0)[0]]
#                   **imshow_dict)
# # ax2.scatter(source_xs, np.ones(len(source_xs)) * -300, c=source_amps, s=100, vmin=-1.4,
# #             vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)


# ax3 = plt.subplot(133, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
# ax3.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
#             edgecolor='k', lw=2, cmap=plt.cm.bwr)
# [ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]
# [ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]

# img3 = ax3.imshow(v_field_ext_xy.T,
#                   extent=[-z_extent, z_extent,
#                           -z_extent, z_extent],
#                   **imshow_dict)
# plt.colorbar(img3, label="mV")
# plt.savefig("electrode_geometry_test.png")

# plt.show()
