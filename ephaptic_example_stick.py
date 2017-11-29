#!/usr/bin/env python

import LFPy
import numpy as np
import matplotlib.pyplot as plt

# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology' : 'stick.hoc',
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    'passive_parameters':dict(g_pas=1/30000., e_pas=-65),
    'v_init' : -65.,    # initial crossmembrane potential
    'passive' : True,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 500.,
    'dt' : 2.**-7,   # [ms] dt's should be in powers of 2 for both,
    'tstart' : 0.,    # start time of simulation, recorders start at t=0
    'tstop' : 3.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
    "extracellular": True,
}

cell = LFPy.Cell(**cell_parameters)
cell.set_pos(z=-500)
n_tsteps = int(cell.tstop / cell.dt + 1)
t = np.arange(n_tsteps) * cell.dt

# Make a linear external field
ext_field = np.vectorize(lambda z: 0. + z/np.max(cell.zmid) * 10.)

pulse = np.zeros(n_tsteps)
pulse[100:] = 1.

# Calculate time dependent field for each cell compartment
v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
v_cell_ext[:, :] = ext_field(cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
cell.insert_v_ext(v_cell_ext, t)

zs = np.linspace(np.min(cell.zmid), np.max(cell.zmid), 4)
cell_plot_idxs = [cell.get_closest_idx(0, 0, z) for z in zs]

# This function is used to color code compartments for plotting
cell_pos_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(zs)) / (np.max(zs) - np.min(zs)))


# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_vmem=True)

# Plot results
plt.close('all')
fig = plt.figure(figsize=[12, 5])
fig.subplots_adjust(wspace=0.6, top=0.83)
fig.suptitle("Stick cell in linear external potential")

ax1 = plt.subplot(151, title="Stick cell", aspect=1, frameon=False, xlim=[-100, 100], xlabel="x [$\mu$m]", ylabel="y [$\mu$m]")
ax2 = plt.subplot(152, title="External potential\n(Ue)", xlim=[0, cell.tstop], xlabel="Time [ms]", ylabel="mV")
ax3 = plt.subplot(153, title="Membrane potential\n(Vm)", sharex=ax2, xlabel="Time [ms]", ylabel="[mV]")
ax4 = plt.subplot(154, title="Ui = Ue + Vm", sharex=ax3, xlabel="Time [ms]", ylabel="[mV]")
ax5 = plt.subplot(155, title="Transmembrane currents", sharex=ax3, xlabel="Time [ms]", ylabel="nA")

[ax1.plot([cell.xstart[idx], cell.xend[idx]],
          [cell.zstart[idx], cell.zend[idx]], '-',
          c='k', clip_on=False) for idx in range(cell.totnsegs)]

for num, idx in enumerate(cell_plot_idxs):
    ax1.plot(cell.xmid[idx], cell.zmid[idx], 'D', c=cell_pos_clr(cell.zmid[idx]))
    ax2.plot(cell.tvec, v_cell_ext[idx], c=cell_pos_clr(cell.zmid[idx]))
    ax3.plot(cell.tvec, cell.vmem[idx, :], c=cell_pos_clr(cell.zmid[idx]))
    ax4.plot(cell.tvec, v_cell_ext[idx] + cell.vmem[idx, :], c=cell_pos_clr(cell.zmid[idx]))
    ax5.plot(cell.tvec, cell.imem[idx, :], c=cell_pos_clr(cell.zmid[idx]))

plt.savefig('example_stick.png')
