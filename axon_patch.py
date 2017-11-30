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
import utils
import neuron
from mpi4py import MPI

#initialize the MPI interface
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

#number of units
n_cells = SIZE
cell_id = RANK

print("cell {0} of {1}").format(cell_id, n_cells)

#set the numpy random seeds
global_seed = 1234
np.random.seed(global_seed)

# plt.interactive(1)
plt.close('all')


COMM.Barrier()

################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################

folder = "morphologies/cell_hallermann_myelin"
#folder = "morphologies/cell_hallermann_unmyelin"
#folder = "morphologies/simple_axon_hallermann"
#folder = "morphologies/HallermannEtAl2012"
neuron.load_mechanisms(join(folder))

if RANK==0:
    # 'morphology' : 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    #'morphology' : join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    #'morphology' : join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder,'cell_simple.hoc')

if RANK==1:
    # 'morphology' : 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    #'morphology' : join(folder, '28_04_10_num19.hoc'), # HallermannEtAl2012
    #'morphology' : join('morphologies', 'axon.hoc'), # Mainen&Sejnowski, 1996
    morph = join(folder,'cell_simple_long.hoc')


# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology' : morph, # simplified neuron model from HallermannEtAl2012
    #rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    # 'passive_parameters':dict(g_pas=1/30., e_pas=-65),
    'v_init' : -85.,    # initial crossmembrane potential
    # 'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : False,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 500.,
    'dt' : 2.**-4,   # [ms] dt's should be in powers of 2 for both,
    'tstart' : -50.,    # start time of simulation, recorders start at t=0
    'tstop' : 100.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
    "extracellular": True,
    'custom_code': [join(folder, 'Cell parameters.hoc'),
                    join(folder, 'charge.hoc')]
                    #,join(folder, 'pruning.hoc')]
}

#assign cell positions
x_cell_pos = np.linspace(-150., 1000., n_cells)
y_cell_pos = np.linspace(-100., 100., n_cells)
#z_cell_pos = np.linspace(-100., 0., n_cells)

#re-seed the random number generator
#cell_seed = global_seed + cell_id
#np.random.seed(cell_seed)


cell = LFPy.Cell(**cell_parameters)

    
cell.set_rotation(y=cell_id*np.pi/2)
#cell.set_rotation(y=np.pi/2)
#cell.set_pos(y=y_cell_pos[cell_id])
#cell.set_rotation(x=np.pi/2)
#cell.set_rotation(y=cell_id*np.pi/2)
#cell.set_rotation(z=np.pi/2)
#cell.set_rotation(x=0, y=0, z=z_rotation[cell_id])
cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id])


n_tsteps = int(cell.tstop / cell.dt + 1)



print("number of segments: ", cell.totnsegs)

t = np.arange(n_tsteps) * cell.dt

pulse_start = 500
pulse_duration = 400

pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start+pulse_duration)] = 1.

cortical_surface_height = np.max(cell.zend) + 150 

# Parameters for the external field
sigma = 0.3
#source_xs = np.array([-70, -70, -10, -10, 10, 10, 70, 70])
source_xs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#source_ys = np.array([-70, 70, -10, 10, 10, -10, -70, 70])
source_ys = np.array([0, 0, 0, 0, 0, 0, 0, 0])
source_zs = np.ones(len(source_xs)) * cortical_surface_height #- 141
#source_amps = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * amp

total_n_runs = 8 
if RANK == 0:
    cells = []
glb_vmem = []
glb_vext = []
spread = np.linspace( -8000, -50000, total_n_runs)
num=0

if 'my[0]' in cell.allsecnames:
    zs = [cell.get_idx('apic[0]')[0], cell.get_idx('soma')[0], cell.get_idx('axon[0]')[0], cell.get_idx('node[0]')[0],cell.get_idx('my[0]')[int(len(cell.get_idx('my[0]'))/2)], cell.get_idx('node[1]')[len(cell.get_idx('axon[1]'))-1], cell.get_idx(cell.allsecnames[len(cell.allsecnames)-1])[0] ]
else:
    zs = [cell.get_idx('apic[0]')[0], cell.get_idx('soma')[0], cell.get_idx('axon[0]')[0], cell.get_idx('axon[1]')[0], cell.get_idx('axon[1]')[len(cell.get_idx('axon[1]'))-1]    ]

v_idxs = zs

#zs = [int(.95*np.min(cell.zmid)), int(.5*np.min(cell.zmid)), int(.1934*np.min(cell.zmid)), 0, int(.95*np.max(cell.zmid))]
#zs = [-750, -500, -100, 0, 100]
#v_idxs = [cell.get_closest_idx(0, 0, z) for z in zs]

#v_clr_z = lambda z: plt.cm.jet(1.0 * (z - np.min(zs)) / (np.max(zs) - np.min(zs)))
#v_clr_x = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.xend)) / (np.max(np.abs(cell.xmid) - np.min(np.abs(cell.xmid)))))
#v_clr_y = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.yend)) / (np.max(np.abs(cell.ymid) - np.min(np.abs(cell.ymid)))))
#v_clr_z = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.zend)) / (np.max(np.abs(cell.zmid) - np.min(np.abs(cell.zmid)))))
c_idxs = lambda z: plt.cm.jet(1.* z / len(v_idxs) )


for s in spread:
    
    amp = s
    source_amps = np.array([0, 0, 0, 0, 1, 0, 0, 0]) * amp
    ExtPot = surface_electrodes.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
    
    # Find external potential field at all cell positions as a function of time
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)
    
    
    v_field_ext_stick = np.zeros((len(zs), n_tsteps))
    #v_field_ext_stick = ext_field(zs).reshape(len(zs), 1) * pulse.reshape(1, n_tsteps)
    v_field_ext_stick = v_cell_ext[v_idxs]
    
    # Insert external potential at cell
    cell.insert_v_ext(v_cell_ext, t)
    glb_vext.append(v_cell_ext)
    
    
    # Run simulation, electrode object argument in cell.simulate
    if RANK==0:
        print "running simulation {0} of {1}".format(num+1, total_n_runs)
    cell.simulate(rec_imem=True, rec_vmem=True)
    glb_vmem.append(cell.vmem)
    #cells.append(cell)
    cell = LFPy.Cell(**cell_parameters)
    
    cell.set_pos(x=x_cell_pos[cell_id], y=y_cell_pos[cell_id])
    cell.set_rotation(y=cell_id*np.pi/2)
    num += 1
    COMM.Barrier()

# plot 3d view of evolution of vmem as a function of stimulation amplitude (legend)
COMM.Barrier()
widx = 3 #to select in len(v_idxs)

if RANK==0:
    #print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
    #single_cells = [utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK)]
    cells.append(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, widx, RANK))
    for i_proc in range(1, SIZE):
       #single_cells = np.r_[single_cells, COMM.recv(source=i_proc)]
       cells.append(COMM.recv(source=i_proc))
else:
    #print len(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, RANK))
    COMM.send(utils.built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, widx, RANK), dest=0)

COMM.Barrier()


##### FIGURES #####

if RANK==0:
    
    #fig = plt.figure()
    fig = plt.figure(figsize=[18, 7])
    fig.suptitle("Evolution of Vmem with increasing current stimulations")
    fig.subplots_adjust(wspace=0.6)


    ax2 = plt.subplot(132, projection='3d', title="Vmem evolution cell "+str(cells[0]['rank']), xlabel="t [ms]", ylabel="Vext [mV]", zlabel="Vmem [mV]", ylim=[np.min([cells[0]['glb_vext'][0][v_idxs[widx]][int(pulse_start+pulse_duration/2)], cells[1]['glb_vext'][0][v_idxs[widx]][int(pulse_start+pulse_duration/2)]]), np.max([cells[0]['glb_vext'][num-1][v_idxs[widx]][int(pulse_start+pulse_duration/2)], cells[1]['glb_vext'][num-1][v_idxs[widx]][int(pulse_start+pulse_duration/2)]])], zlim=[np.min([np.min(cells[0]['glb_vmem'][0][v_idxs[widx]]), np.min(cells[1]['glb_vmem'][0][v_idxs[widx]])]), np.max([np.max(cells[0]['glb_vmem'][num-1][v_idxs[widx]]), np.max(cells[1]['glb_vmem'][num-1][v_idxs[widx]])])])
    color=iter(plt.cm.rainbow(np.linspace(0,1,total_n_runs)))
    for i in range(num):
        #ax2.plot_wireframe(t, glb_vext[i][v_idxs[0]], glb_vmem[i][v_idxs[0]], cmap=plt.cm.bwr )
        ax2.plot_wireframe(t, np.sign(np.min(glb_vext[i][v_idxs[widx]][pulse_start:(pulse_start+pulse_duration)]))*round( np.max(np.abs(glb_vext[i][v_idxs[widx]])), 2) , glb_vmem[i][v_idxs[widx]],  color=next(color) )

    ax3 = plt.subplot(133, projection='3d', title="Vmem evolution cell "+str(cells[0]['rank']), xlabel="t [ms]", ylabel="Vext [mV]", zlabel="Vmem [mV]", ylim=[np.min([cells[0]['glb_vext'][0][v_idxs[widx]][int(pulse_start+pulse_duration/2)], cells[1]['glb_vext'][0][v_idxs[widx]][int(pulse_start+pulse_duration/2)]]), np.max([cells[0]['glb_vext'][num-1][v_idxs[widx]][int(pulse_start+pulse_duration/2)], cells[1]['glb_vext'][num-1][v_idxs[widx]][int(pulse_start+pulse_duration/2)]])], zlim=[np.min([np.min(cells[0]['glb_vmem'][0][v_idxs[widx]]), np.min(cells[1]['glb_vmem'][0][v_idxs[widx]])]), np.max([np.max(cells[0]['glb_vmem'][num-1][v_idxs[widx]]), np.max(cells[1]['glb_vmem'][num-1][v_idxs[widx]])])])
    color=iter(plt.cm.rainbow(np.linspace(0,1,total_n_runs)))
    for i in range(num):
        #ax3.plot_wireframe(t, glb_vext[i][v_idxs[0]], glb_vmem[i][v_idxs[0]], cmap=plt.cm.bwr )
        ax3.plot_wireframe(t, np.sign(np.min(cells[1]['glb_vext'][i][v_idxs[widx]][pulse_start:(pulse_start+pulse_duration)]))*round( np.max(np.abs(cells[1]['glb_vext'][i][v_idxs[widx]])), 2) , cells[1]['glb_vmem'][i][v_idxs[widx]],  color=next(color) )
    
    
    ##### ? yinfo = np.zeros(total_n_runs)
    
    
    ax1 = plt.subplot(131, title="3D view", aspect=1, projection='3d',  xlabel="x [$\mu$m]", ylabel="y [$\mu$m]", zlabel="z [$\mu$m]", xlim=[-1000,1000], ylim=[-1000, 1000], zlim=[-800, 200])
    #[ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
    #          c='k', clip_on=False) for idx in range(cell.totnsegs)]
    for nc in range(0,SIZE):
        [ax1.plot([cells[nc]['xstart'][idx], cells[nc]['xend'][idx]], [cells[nc]['ystart'][idx], cells[nc]['yend'][idx]], [cells[nc]['zstart'][idx], cells[nc]['zend'][idx]], '-',
                  c='k', clip_on=False) for idx in range(cells[nc]['totnsegs'])]
        [ax1.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]], [cells[nc]['zmid'][idx]], 'D', c= c_idxs(cells[nc]['v_idxs'].index(idx))) for idx in cells[nc]['v_idxs']]
        [ax1.plot([cells[nc]['xmid'][idx]], [cells[nc]['ymid'][idx]], [cells[nc]['zmid'][idx]], 'D', c= 'k') for idx in cells[nc]['v_idxs']]
        #ax1.text(cells[nc]['xmid'][0], cells[nc]['ymid'][0], cells[nc]['zmid'][0], "cell {0}".format(cells[nc]['rank']))
        ax1.text(cells[nc]['xmid'][v_idxs[widx]], cells[nc]['ymid'][v_idxs[widx]], cells[nc]['zmid'][v_idxs[widx]], "cell {0}.".format(cells[nc]['rank']) + cells[nc]['name'])
    
    ax1.scatter(source_xs, source_ys, source_zs, c=source_amps)
    
    #ax.axes.set_yticks(yinfo)
    #ax.axes.set_yticklabels(yinfo)
    plt.show()
