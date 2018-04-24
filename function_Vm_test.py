#!/usr/bin/env python
import matplotlib
matplotlib.use('TkAgg')
import neuron
import LFPy
import numpy as np
import matplotlib.pyplot as plt
import utils
import os

output_f = '/media/erebus/oslo/code/darpa/bisc/outputs/'
name_model = 'hallermann_soma_axon_myelin'
name_shape_ecog = 'monopole'

# Define cell parameters
# BALL AND STICK
if name_model == 'ball_and_stick':
    cell_parameters = {          # various cell parameters,
        'morphology': 'morphologies/ball_and_stick.hoc',
        'cm': 1.0,         # membrane capacitance
        'Ra': 150,        # axial resistance
        'passive_parameters': dict(g_pas=1 / 30000., e_pas=-65),
        'v_init': -65.,    # initial crossmembrane potential
        'passive': True,   # switch on passive mechs
        'nsegs_method': 'lambda_f',
        'lambda_f': 500.,
        'dt': 2.**-7,   # [ms] dt's should be in powers of 2 for both,
        'tstart': 0.,    # start time of simulation, recorders start at t=0
        'tstop': 3.,   # stop simulation at 200 ms. These can be overridden
                            # by setting these arguments i cell.simulation()
        "extracellular": True,
    }

# HALLERMANN STICK
if name_model[:10] == 'hallermann':
    folder = "morphologies/cell_hallermann_myelin"
    # folder = "morphologies/cell_hallermann_unmyelin"
    # folder = "morphologies/simple_axon_hallermann"
    # folder = "morphologies/HallermannEtAl2012"
    neuron.load_mechanisms(os.path.join(folder))
    # morph = 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    # morph = join(folder, '28_04_10_num19.hoc') # HallermannEtAl2012
    # morph = join('morphologies', 'axon.hoc') # Mainen&Sejnowski, 1996
    # morph = join(folder, 'cell_simple.hoc')
    morph = os.path.join(folder, 'cell_simple.hoc')
    custom_code = [os.path.join(folder, 'Cell parameters.hoc'),
                   os.path.join(folder, 'charge.hoc'),
                   os.path.join(folder, 'pruning.hoc')]
    # Define cell parameters
    cell_parameters = {          # various cell parameters,
        'morphology': morph,  # simplified neuron model from HallermannEtAl2012
        # rm': 30000.,      # membrane resistance
        'cm': 1.0,         # membrane capacitance
        'Ra': 150,        # axial resistance
        # 'passive_parameters':dict(g_pas=1/30., e_pas=-65),
        'v_init': -85.,    # initial crossmembrane potential
        # 'e_pas': -65.,     # reversal potential passive mechs
        'passive': False,   # switch on passive mechs
        'nsegs_method': 'lambda_f',
        'lambda_f': 300.,
        'dt': 2.**-4,   # [ms] dt's should be in powers of 2 for both,
        'tstart': -50.,    # start time of simulation, recorders start at t=0
        'tstop': 50.,   # stop simulation at 200 ms. These can be overridden
                            # by setting these arguments in cell.simulation()
        "extracellular": True,
        "pt3d": True,
        'custom_code': custom_code}




# Make a linear external field
# ext_field = np.vectorize(lambda z: 0. + z/np.max(cell.zmid) * 10.)


sigma = 0.3
polarity, n_elec, positions = utils.create_array_shape(name_shape_ecog, 25)
dura_height = 50
displacement_source = 50

SIZE = 1
n_intervals = 21

distance = np.linspace(0, 500, n_intervals)

source_xs = positions[0]
source_ys = positions[1] + displacement_source
# source_ys = positions[1]
source_zs = positions[2] + dura_height

min_current = -200 * 10**3  # uA
max_current = 200 * 10**3  # uA

n_tsteps = int(cell_parameters['tstop'] / cell_parameters['dt'] + 1)

pulse = np.zeros(n_tsteps)
pulse[100:] = 1.
index = 130  # pulse start at 100:


amp_spread = np.linspace(min_current, max_current, n_intervals)


all_currents = []
all_cvext = []

for idx, dis in enumerate(distance):
    current = []
    c_vext = []
    ap_loc = []
    for amp in amp_spread:

        cell = LFPy.Cell(**cell_parameters)
        cell.set_pos(z=-np.max(cell.zend) - dis)
        n_tsteps = int(cell.tstop / cell.dt + 1)
        t = np.arange(n_tsteps) * cell.dt

        source_amps = np.multiply(polarity, amp)
        ExtPot = utils.ImposedPotentialField(source_amps, positions[0], positions[1] + displacement_source,
                                             positions[2] + dura_height, sigma)

        v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))

        v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid
                                            ).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

        # Calculate time dependent field for each cell compartment
        # v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
        # v_cell_ext[:, :] = ext_field(cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
        cell.insert_v_ext(v_cell_ext, t)
        cell.simulate(rec_imem=True, rec_vmem=True)

        current.append(cell.vmem[0][index])  # SOMA
        c_vext.append(cell.v_ext[0][index])  # SOMA
        # ap_loc
    all_currents.append(current)
    all_cvext.append(c_vext)



# zs = np.linspace(np.min(cell.zmid), np.max(cell.zmid), 4)
# cell_plot_idxs = [cell.get_closest_idx(0, 0, z) for z in zs]

# # This function is used to color code compartments for plotting
# cell_pos_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(zs)) / (np.max(zs) - np.min(zs)))


# Run simulation, electrode object argument in cell.simulate
# print("running simulation...")

# Plot results
# plt.close('all')
fig = plt.figure(figsize=[12, 6])
# fig = plt.figure()
# fig.subplots_adjust(wspace=0.6, top=0.83)
# fig.suptitle("Vmem = f(I) in Stick cell")

ax1 = plt.subplot(131, title=name_model, aspect='auto', frameon=False, xlim=[-100, 100], xlabel="x [$\mu$m]", ylabel="y [$\mu$m]")
ax2 = plt.subplot(132, title="External potential\n(Ue)", xlabel="I [uA]", ylabel="mV")
ax3 = plt.subplot(133, title="Membrane potential\n(Vm)", xlabel="I [uA]", ylabel="[mV]")
# ax4 = plt.subplot(154, title="Ui = Ue + Vm", sharex=ax3, xlabel="Time [ms]", ylabel="[mV]")
# ax5 = plt.subplot(155, title="Transmembrane currents", sharex=ax3, xlabel="Time [ms]", ylabel="nA")


color = iter(plt.cm.rainbow(np.linspace(0, 1, n_intervals)))

[ax1.plot([cell.xstart[idx], cell.xend[idx]],
          [cell.zstart[idx], cell.zend[idx]], '-',
          c='k', clip_on=False) for idx in range(cell.totnsegs)]

ax1.plot(cell.xmid, cell.zmid, 'D', c='k')
ax1.scatter(source_xs, source_zs, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
            edgecolor='k', lw=2, cmap=plt.cm.bwr)
[ax1.scatter(source_xs[i], source_zs[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps > 0)[0]]
[ax1.scatter(source_xs[i], source_zs[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps < 0)[0]]

for i, dis in enumerate(distance):
    current_color = next(color)
    ax2.plot(amp_spread / 1000, all_cvext[i], c=current_color, label=int(dis))
    ax2.legend()
    ax3.plot(amp_spread / 1000, all_currents[i], c=current_color)
    # ax4.plot(cell.tvec, v_cell_ext[idx] + cell.vmem[idx, :], c=current_color)
    # ax5.plot(cell.tvec, cell.imem[idx, :], c=current_color)
    art = []
    lgd = ax2.legend(title="distance [$\mu$m]", fancybox=True, loc=9, prop={'size': 8}, bbox_to_anchor=(1.1, -0.1), ncol=6)
    art.append(lgd)
# plt.tight_layout()
plt.savefig(os.path.join(output_f, 'Vm_function_I_'+ name_model +'_' + name_shape_ecog + '.png'), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
plt.show()
