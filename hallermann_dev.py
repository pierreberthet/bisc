#!/usr/bin/env python
'''
Implementation of Hallermann et al 2012 neuron model on a simple stick model
'''
import LFPy
import numpy as np
import os
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from GeneticOptimization import MEAutility as MEA
import surface_electrodes
from os.path import join
import neuron

# plt.interactive(1)
plt.close('all')


################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################
#folder = "morphologies/cell_hallermann_myelin"
folder = "morphologies/cell_hallermann_unmyelin"
#folder = "morphologies/simple_axon_hallermann"
#folder = "morphologies/HallermannEtAl2012"
neuron.load_mechanisms(join(folder))

# Define cell parameters
cell_parameters = {          # various cell parameters,
    # 'morphology' : 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    #'morphology' : join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    'morphology' : join(folder,'cell_simple.hoc'), # simplified neuron model from HallermannEtAl2012
    #'morphology' : join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    #rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    # 'passive_parameters':dict(g_pas=1/30., e_pas=-65),
    'v_init' : -85.,    # initial crossmembrane potential
    # 'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : False,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 1000.,
    'dt' : 2.**-4,   # [ms] dt's should be in powers of 2 for both,
    'tstart' : -50.,    # start time of simulation, recorders start at t=0
    'tstop' : 100.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
    "extracellular": True,
    'custom_code': [join(folder, 'Cell parameters.hoc'),
                    join(folder, 'charge.hoc')]
                    #,join(folder, 'pruning.hoc')]
}


cell = LFPy.Cell(**cell_parameters)
#cell.set_rotation(x=np.pi/2)
#cell.set_rotation(y=np.pi/2)
#cell.set_rotation(z=np.pi/2)
cell.set_pos(x=50)
n_tsteps = int(cell.tstop / cell.dt + 1)



print("number of segments: ", cell.totnsegs)

t = np.arange(n_tsteps) * cell.dt

pulse_start = 500
pulse_duration = 400

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start+pulse_duration)] = 1.


linear = False
synapse = False
mea = False 
surface_stim = True 

green_field= False


if mea:
    N_side=10
    mea = MEA.SquareMEA(dim=N_side, pitch=10, x_plane=-10)
    amp = 5e3 # current {Pierre: in nA?}
    mea.reset_currents()
    mea[int(round(N_side/2))][int(round(N_side/2))].set_current(amp)
#    for xx in range(3):
#        for yy in range(3):
#            if xx != 1 or yy != 1:
#                mea[int(round(N_side/2))-1+xx][int(round(N_side/2))-1+yy].set_current(-amp/8)
    currents_mea = mea.get_currents().reshape(N_side,N_side)
    print 'MEA currents'
    for ii in range(N_side):
        print ["%6.f" %xx for xx in currents_mea[ii]]       
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_cell_ext[:,:] = np.transpose(mea.compute_field(\
            np.transpose(np.array([cell.xmid, cell.ymid, cell.zmid]))) * pulse.reshape(n_tsteps,1))
        # mea.compute_field(np.transpose(np.array([cell.xmid, cell.ymid, cell.zmid]))).reshape(cell.totnsegs, 1) * pulse[pulse_start:(pulse_start+pulse_duration)].reshape(1, n_tsteps)
        #{Pierre: 3d field for a 3d model?}

if linear:
    ext_field = np.vectorize(lambda z: -20 + z/np.max(cell.zmid) * 40)
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_cell_ext[:, :] = ext_field(cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

    v_field_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_field_ext = ext_field(np.linspace(np.min(cell.zmid), np.max(cell.zmid), cell.totnsegs)).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
#    v_field_ext = np.zeros((50, n_tsteps))
#    v_field_ext = ext_field(np.linspace(np.min(cell.zmid), np.max(cell.zmid), 50)).reshape(50, 1) * pulse.reshape(1, n_tsteps)

if synapse:
    #Synaptic parameters, corresponding to a NetCon synapse built into NEURON
    synapseParameters = {
        'idx' : cell.get_closest_idx(.3*np.min(cell.zmid)),               # insert synapse on index "idx",  "0" being the soma
        'e' : 0.,                # reversal potential of synapse
        'syntype' : 'Exp2Syn',   # conductance based double-exponential synapse
        'tau1' : 1.0,            # Time constant, rise
        'tau2' : 1.0,            # Time constant, decay
        'weight' : 0.05,         # Synaptic weight
        'record_current' : True, # Will enable synapse current recording
    }
    
    
    #attach synapse with parameters and set spike time
    synapse = LFPy.Synapse(cell, **synapseParameters)
    synapse.set_spike_times(np.array([10]))
    
    d = cell.diam[0] * 1e-6  # meter
    Rm = 1 / cell_parameters["passive_parameters"]["g_pas"]  # Ohm / cm2
    Ra = cell_parameters["Ra"] * 1e2  # Ohm meter
    cm = cell_parameters["cm"] * 1e-6  # F / cm2
    
    tau_m = Rm * cm * 1000  # ms
    lmda = np.sqrt(d * Rm / (4 * Ra)) * 1e6  # uM
    print tau_m, lmda


if surface_stim:
    cortical_surface_height = np.max(cell.zend) +20 
    
    # Parameters for the external field
    sigma = 0.3
    amp = -400.
    source_xs = np.array([-70, -70, -10, -10, 10, 10, 70, 70])
    source_ys = np.array([-70, 70, -10, 10, 10, -10, -70, 70])
    source_zs = np.ones(len(source_xs)) * cortical_surface_height #- 141
    #source_amps = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * amp
    source_amps = np.array([0, 0, 1, 1, 1, 1, 0, 0]) * amp
    
    ExtPot = surface_electrodes.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
    
    # Find external potential field at all cell positions as a function of time
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
    
#zs = [-750, -500, -100, 0, 100]
zs = [int(.95*np.min(cell.zmid)), int(.5*np.min(cell.zmid)), int(.1934*np.min(cell.zmid)), 0, int(.95*np.max(cell.zmid))]
v_idxs = [cell.get_closest_idx(0, 0, z) for z in zs]
#v_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(zs)) / (np.max(zs) - np.min(zs)))
v_clr = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.zend)) / (np.max(np.abs(cell.zmid) - np.min(np.abs(cell.zmid)))))

v_field_ext_stick = np.zeros((len(zs), n_tsteps))
#v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
v_field_ext_stick = v_cell_ext[v_idxs]

# Insert external potential at cell
cell.insert_v_ext(v_cell_ext, t)


# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_vmem=True)

#############################
################ FIGURES ####
if green_field:
    v_field_ext_xz = np.zeros((201, 201))
    v_field_ext_yz = np.zeros((201, 201))
    xf = np.linspace(1, 201, 201)
    yf = np.linspace(-100, 100, 201)
    zf = np.linspace(-100, 100, 201)
    for zidx, z in enumerate(zf):
        for xidx, x in enumerate(xf):
            if linear:
                v_field_ext_xz[xidx, zidx] = tb.linear_field(z, a=a, b=b)
           # if poly:
           #     v_field_ext_xz[xidx, zidx] = tb.poly_field(z, a=a, b=b, c=c, d=d, e=e)
            if mea:
                v_field_ext_xz[xidx, zidx] = mea.compute_field(np.array([x, 0, z]))

    vmax = np.max(v_field_ext_xz)
    vmin = np.min(v_field_ext_xz)


fig = plt.figure(figsize=[18, 7])
fig.subplots_adjust(wspace=0.6)

# fig.suptitle(r"$\lambda$ = {} $\mu$m; $\tau_m$ = {} ms".format(lmda, tau_m))
# ax3 = plt.subplot(152, title="Ue", ylim=[-200, 1200], xlim=[0, cell.tstop], xlabel="Time [ms]", ylabel="y [$\mu$m]")
ax3 = plt.subplot(162, title="Ue", xlim=[0, cell.tstop], xlabel="Time [ms]", ylabel="[mV]")
# img = plt.imshow(v_field_ext, extent=[0, cell.tvec[-1], np.min(cell.zmid), np.max(cell.zmid)],
#                  origin='bottom', cmap=plt.cm.bwr, aspect='auto', vmin=-np.max(np.abs((v_cell_ext))), vmax=np.max(np.abs((v_cell_ext))))
# plt.colorbar(img)
[ax3.plot(cell.tvec, v_field_ext_stick[num], c=v_clr(cell.zmid[idx])) for num, idx in enumerate(v_idxs)]

#ax2 = plt.subplot(161, title="Cell model", aspect=1, frameon=False, xlim=[-100, 100], xlabel="x [$\mu$m]", ylabel="z [$\mu$m]")
ax2 = plt.subplot(161, title="Cell model", aspect=1, projection='3d',  xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]")
#[plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
[ax2.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
          c='k', clip_on=False) for idx in range(cell.totnsegs)]
[ax2.plot([cell.xmid[idx]], [cell.ymid[idx]], [cell.zmid[idx]], 'D', c=v_clr(cell.zmid[idx])) for idx in v_idxs]
[ax2.scatter(source_xs, source_ys, source_zs, c=source_amps)]
if green_field:
    img1 = ax2.imshow(v_field_ext_xz.T,
                  extent=[-400, 400, -200, 200],
                  origin='lower',
                  interpolation='nearest',
                  cmap='Greens',
                  vmin=vmin,
                  vmax=vmax
                  )
    # ,
    #               norm=matplotlib.colors.SymLogNorm(10**-logthresh))
    cb = plt.colorbar(img1, ax=ax2, shrink=.8)   #, ticks=tick_locations)
    #cb.set_label('mV', y=1.12, rotation='vertical', x=.5)
    cb.set_label('mV', fontsize=9)

ax1 = plt.subplot(163, title="Vm", xlabel="Time [ms]", ylabel="[mV]")
[ax1.plot(cell.tvec, cell.vmem[idx, :], c=v_clr(cell.zmid[idx])) for idx in v_idxs]

ax1 = plt.subplot(164, title="Ui = Ue + Vm", sharex=ax3, xlabel="Time [ms]", ylabel="[mV]")
[ax1.plot(cell.tvec, v_field_ext_stick[num] + cell.vmem[idx, :], c=v_clr(cell.zmid[idx])) for num, idx in enumerate(v_idxs)]

ax5 = plt.subplot(165, title="(Vm_n+1 - Vm_n) / dt", xlabel="Time [ms]", ylabel="[mV / t]", xlim=[pulse_start*cell.dt-5, (pulse_start+pulse_duration)*cell.dt+15])
[ax5.plot(cell.tvec[:-1], (cell.vmem[idx, 1:] - cell.vmem[idx, :-1]) / cell.dt, c=v_clr(cell.zmid[idx])) for idx in v_idxs]


ax6 = plt.subplot(166, title="Im", sharex=ax3, xlabel="Time [ms]", ylabel="[nA]")
[ax6.plot(cell.tvec, cell.imem[idx, :], c=v_clr(cell.zmid[idx])) for idx in v_idxs]


plt.savefig(folder+'_.png')
plt.show()
