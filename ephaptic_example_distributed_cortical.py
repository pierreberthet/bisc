#!/usr/bin/env python
import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import LFPy


class ImposedPotentialField:
    """Class to make the imposed external from given current sources.

    Parameters
    ----------
    source_amps : array, list
        Amplitudes of current sources in nA.
    source_xs : array, list
        x-positions of current sources
    source_ys : array, list
        x-positions of current sources
    source_zs : array, list
        x-positions of current sources
    sigma : float
        Extracellular conductivity in S/m, defaults to 0.3 S/m

    """
    def __init__(self, source_amps, source_xs, source_ys, source_zs, sigma=0.3):

        self.source_amps = np.array(source_amps)
        self.source_xs = np.array(source_xs)
        self.source_ys = np.array(source_ys)
        self.source_zs = np.array(source_zs)
        self.num_sources = len(source_amps)

        self.sigma = sigma

    def ext_field(self, x, y, z):
        """Returns the external field at positions x, y, z"""
        ef = 0
        for s_idx in range(self.num_sources):
            ef += self.source_amps[s_idx] / (2 * np.pi * self.sigma * np.sqrt(
                (self.source_xs[s_idx] - x) ** 2 +
                (self.source_ys[s_idx] - y) ** 2 +
                (self.source_zs[s_idx] - z) ** 2))
        return ef


# Define cell parameters
cell_params = {
    'morphology' : join('cells', 'cells', 'j4a.hoc'), # from Mainen & Sejnowski, J Comput Neurosci, 1996
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150.,        # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'passive' : True,   # turn on NEURONs passive mechanism for all sections
    'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
    'nsegs_method' : 'lambda_f', # spatial discretization method
    'lambda_f' : 100.,           # frequency where length constants are computed
    'dt' : 2.**-4,      # simulation time step size
    'tstart' : 0.,      # start time of simulation, recorders start at t=0
    'tstop' : 100.,     # stop simulation at 100 ms.
    'extracellular': True,
}
cell = LFPy.Cell(**cell_params)
cell.set_rotation(x=4.99, y=-4.33, z=3.14)

cortical_surface_height = np.max(cell.zend) + 20

# Parameters for the external field
sigma = 0.3
amp = 100000.
start_time = 20
source_xs = np.array([-70, -70, -10, -10, 10, 10, 70, 70])
source_ys = np.array([-70, 70, -10, 10, 10, -10, -70, 70])
source_zs = np.ones(len(source_xs)) * cortical_surface_height
source_amps = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * amp

ExtPot = ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)

# Make step function to represent turning on current
n_tsteps = int(cell_params["tstop"] / cell_params["dt"] + 1)
t = np.arange(n_tsteps) * cell_params["dt"]
pulse = np.zeros(n_tsteps)
start_idx = np.argmin(np.abs(t - start_time))
pulse[start_idx:] = 1.

# Find external potential field at all cell positions as a function of time
v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid
                                    ).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)


# Insert external potential at cell
cell.insert_v_ext(v_cell_ext, t)
cell.simulate(rec_imem=True, rec_vmem=True)

# Make cross-sections of external field for plotting purposes
plot_field_length = 400
v_field_ext_xz = np.zeros((100, 100))
xf = np.linspace(-plot_field_length, plot_field_length, 100)
zf = np.linspace(-plot_field_length, cortical_surface_height, 100)
for xidx, x in enumerate(xf):
    for zidx, z in enumerate(zf):
        v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(x, 0, z)

v_field_ext_xy = np.zeros((100, 100))
xf = np.linspace(-plot_field_length, plot_field_length, 100)
yf = np.linspace(-plot_field_length, plot_field_length, 100)
for xidx, x in enumerate(xf):
    for yidx, y in enumerate(yf):
        v_field_ext_xy[xidx, yidx] = ExtPot.ext_field(x, y, cortical_surface_height)

plt.close('all')

fig = plt.figure(figsize=[12, 8])
fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9,
                    bottom=0.1, right=0.98, left=0.05)

ax_side = plt.subplot(221, aspect=1, frameon=False, xticks=[],
                      title='Sources and imposed potential\n(side view)',
                      ylim=[-plot_field_length, cortical_surface_height],
                      xlabel='x ($\mu$m)', ylabel='z ($\mu$m)')
ax_top = plt.subplot(223, aspect=1, frameon=False, xticks=[],
                     title='Sources and imposed potential\n(top view)',
                     ylim=[-plot_field_length, plot_field_length],
                     xlabel='x ($\mu$m)', ylabel='y ($\mu$m)')
ax_vmem = plt.subplot(224, title='Somatic membrane potential',
                      ylabel='mV', xlabel="Time (ms)")
ax_curr = plt.subplot(222, ylim=[-5 - amp / 1000. - 5, amp / 1000. + 5],
                      ylabel='$\mu$A',
                      title='Injected currents', xlabel="Time (ms)")

# Plot cell morphology
cell_clr = 'k'
for idx in range(cell.totnsegs):
    if idx == 0:
        # Plot soma (idx = 0) as ball
        ax_side.plot(cell.xmid[idx], cell.zmid[idx], 'o',
                     ms=8, c=cell_clr, mec='none')
        ax_top.plot(cell.xmid[idx], cell.ymid[idx], 'o',
                    ms=8, c=cell_clr, mec='none')
    else:
        # Plot dendrites as lines
        ax_side.plot([cell.xstart[idx], cell.xend[idx]],
                     [cell.zstart[idx], cell.zend[idx]], c=cell_clr, lw=2)
        ax_top.plot([cell.xstart[idx], cell.xend[idx]],
                    [cell.ystart[idx], cell.yend[idx]], c=cell_clr, lw=2)

# Plot current sources
ax_side.scatter(source_xs, source_zs, c=source_amps, s=100, vmin=-1.4,
                vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)
ax_top.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
               edgecolor='k', lw=2, cmap=plt.cm.bwr)

ax_side.scatter(source_xs[np.where(source_amps < 0)],
                source_zs[np.where(source_amps < 0)],
                c="k", s=50, marker='_', lw=2, clip_on=False)
ax_side.scatter(source_xs[np.where(source_amps > 0)],
                source_zs[np.where(source_amps > 0)],
                c="k", s=50, marker='+', lw=2, clip_on=False)
ax_top.scatter(source_xs[np.where(source_amps < 0)],
               source_ys[np.where(source_amps < 0)],
               c="k", s=50, marker='_', lw=2, clip_on=False, zorder=100)
ax_top.scatter(source_xs[np.where(source_amps > 0)],
               source_ys[np.where(source_amps > 0)],
               c="k", s=50, marker='+', lw=2, clip_on=False, zorder=100)

# Plot imposed fields with logarithmic strength for both positive and negative
logthresh = 0
vmax = 100
vmin = -100
maxlog = int(np.ceil(np.log10(vmax)))
minlog = int(np.ceil(np.log10(-vmin)))
tick_locations = ([-(10**x) for x in xrange(minlog, -logthresh-1, -1)]
                + [0.0]
                + [(10**x) for x in xrange(-logthresh, maxlog+1)])

imshow_dict = dict(origin='lower', interpolation='nearest',
                  cmap=plt.cm.bwr, vmin=vmin, vmax=vmax,
                  norm=matplotlib.colors.SymLogNorm(10**-logthresh))

img1 = ax_side.imshow(v_field_ext_xz.T,
                      extent=[-plot_field_length, plot_field_length,
                              -plot_field_length, cortical_surface_height],
                      **imshow_dict)
img2 = ax_top.imshow(v_field_ext_xy.T,
                     extent=[-plot_field_length, plot_field_length,
                             -plot_field_length, plot_field_length],
                     **imshow_dict)
cax = plt.axes([0.4, 0.1, 0.01, 0.33])
cb = plt.colorbar(img1, cax=cax)
cb.set_ticks(tick_locations)
cb.set_label('mV', labelpad=-10)

# Plot membrane potential and current pulses
ax_vmem.plot(t, cell.vmem[0, :], c=cell_clr, lw=3)
ax_curr.plot(t, pulse * amp / 1000, lw=3, c='r')
ax_curr.plot(t, -pulse * amp / 1000, lw=3, c='b')

for ax in [ax_side, ax_top, ax_vmem, ax_curr]:
    ax.set_yticks(ax.get_yticks()[::2])

plt.savefig('distributed_surface_current_source.png')
