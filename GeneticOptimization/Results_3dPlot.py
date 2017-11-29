'''
Visualize and Analyze results from Monte Carlo simulation
'''

import SimulateMEA as sim
import MEAutility as MEA
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy import stats
import pickle
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import StimSim_toolbox as tb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import colors as mpl_colors
try:
        with open('objs.pickle') as f:
                [complexity,
                 performances,
                 currents,
                 neurons,
                 v_grid,
                 x_vec,
                 y_vec,
                 z_vec,
                 mea,
                 target_neurons,
                 surround_neurons,
                 unit,
                 bound,
                 v_grid_dv2_z,
                 v_grid_dv2_y] = pickle.load(f)

except:
        # Sample Scenario
        N_target = 1
        N_surround = 4
        Ngen = 300
        Nstall = 100
        N_sim = 1
        monopolar_current = -10000
        vm_targ = 1.5
        vm_surr = 0.2

        # meaParam: N_side, pitch, current_step, max_current, monopolaramp
        meaParam = [4, 15, 2, 20, monopolar_current]
        # neuronParam: close_separation, xlim, n_target, n_surround, axon_length, discrete
        neuronParam = [15, [5, 15], [5, 30], N_target, N_surround, 15, 3, 15]
        # fitParam: vm_target, vm_surround, alpha_target_surround, alpha_energy_sparsity
        fitParam = [vm_targ, vm_surr, 0.4, 0.5]
        # gaParam: NGEN, CXPB, MUTPB, PBEST, NSTALL
        gaParam = [300, 0.8, 0.1, 0.04, 100]
        # mutselParam: muGauss, sdGauss, pgauss, pzero, tourn_size
        mutselParam = [0, 4000, 0.2, 0.2, 3]

        s = sim.SimulateScenario()

        [complexity,
         performances,
         currents,
         neurons] = s.simulate_scenario(meaParam,
                                        neuronParam,
                                        fitParam,
                                        gaParam,
                                        mutselParam,
                                        verbose=True)
        pitch = 15
        mea = MEA.SquareMEA(dim=4, pitch=pitch)
        target_neurons = neurons[0]
        surround_neurons = neurons[1]

        mea.set_current_matrix(currents[0])

        unit = 1
        bound = abs(mea[0][0].position[1]) + pitch
        x_vec = np.arange(1, bound, unit)
        y_vec = np.arange(-bound, bound, unit)
        z_vec = np.arange(-bound, bound, unit)

        x, y, z = np.meshgrid(x_vec, y_vec, z_vec)

        v_grid = np.zeros((len(y_vec), len(z_vec)))

        # maintain matrix orientation (row - z, column - y, [0,0] - top left corner)
        z_vec = z_vec[::-1]

        for ii in range(len(z_vec)):
                for jj in range(len(y_vec)):
                    v_grid[ii, jj] = mea.compute_field(np.array([15, y_vec[jj], z_vec[ii]]))
                    # print np.array([10, y_vec[jj], z_vec[ii]])

        v_grid_dv2_z = np.zeros((len(y_vec), len(z_vec)))
        v_grid_dv2_y = np.zeros((len(y_vec), len(z_vec)))

        tract_image = (max(y_vec)-min(y_vec)) / (len(y_vec) - 1)

        for ii in range(len(y_vec)):
                v_grid_dv2_z[:, ii] = np.gradient(
                        np.gradient(v_grid[:, ii])) / tract_image**2
                v_grid_dv2_y[ii, :] = np.gradient(
                        np.gradient(v_grid[ii, :])) / tract_image**2

        with open('objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([complexity,
                             performances,
                             currents,
                             neurons,
                             v_grid,
                             x_vec,
                             y_vec,
                             z_vec,
                             mea,
                             target_neurons,
                             surround_neurons,
                             unit,
                             bound,
                             v_grid_dv2_z,
                             v_grid_dv2_y], f)


'''3d plot'''
plt.close('all')

fig = plt.figure(figsize=[6, 16])
gs = gridspec.GridSpec(9,
                       10,
                       hspace=0.,
                       wspace=0.)
fig.subplots_adjust(left=0.01, right=.8, top=1., bottom=0.01)
elev = 30
azim = -60
dist = 10
# Add surface
y_plane, z_plane = np.meshgrid(y_vec, z_vec)

v_grid_orig = np.zeros((len(y_vec), len(z_vec)))

# maintain matrix orientation (row - z, column - y, [0,0] - top left corner)

for ii in range(len(z_vec)):
    for jj in range(len(y_vec)):
        v_grid_orig[ii, jj] = mea.compute_field(np.array(
            [15, y_plane[ii][jj], z_plane[ii][jj]]))
        
#ax1 = fig.add_subplot(311, projection='3d')
ax1 = fig.add_subplot(gs[0:3, 0:9], projection='3d')
#ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=2)

ax1.view_init(elev=elev, azim=azim)
surf1 = ax1.plot_surface(y_plane,
                         z_plane,
                         v_grid_orig,
                         cmap=cm.coolwarm,
                         alpha=0.3,
                         zorder=0,
                         antialiased=True)
# ax1.contour(y_plane,
#             z_plane,
#             v_grid_orig,
#             cmap=cm.coolwarm,
#             extend3d=True,)

ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
# Get rid of the panes
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax1.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.dist = 10..
cax1 = fig.add_subplot(gs[1, 9:])
cbar_ax1 = fig.colorbar(surf1, cax=cax1)
cbar_ax1.set_label('mV', rotation=270)

ax2 = fig.add_subplot(gs[3:6, 0:9], projection='3d')
cax2 = fig.add_subplot(gs[4, 9:])
ax2.view_init(elev=elev, azim=azim)
ax2.set_xlim3d(-30, 30)
ax2.set_ylim3d(-30, 30)
ax2.set_zlim3d(0, 30)
ax2.dist = 10..
soma_length = 3.
soma_radius = 1.
axon_length = 15.
axon_radius = .2
n_points = 20.
for neur in range(len(target_neurons)):
        direction = target_neurons[neur].align_dir
        #1,2,0
        direction = np.array([direction[1], direction[2], direction[0]])
        color = 'r'
        #	soma
        soma_pos = target_neurons[neur].soma_pos
        soma_pos = np.array([soma_pos[1], soma_pos[2], soma_pos[0]])
        axon_pos = target_neurons[neur].soma_pos
        axon_pos = np.array([axon_pos[1], axon_pos[2], axon_pos[0]])
                            
        # get poly soma
        soma_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                                   direction=direction,
                                                   length=soma_length,
                                                   radius=soma_radius,
                                                   n_points=n_points,
                                                   facecolor=color,
                                                   alpha=1.)
        # get poly axon
        axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                                   direction=direction,
                                                   length=axon_length,
                                                   radius=axon_radius,
                                                   n_points=n_points,
                                                   facecolor=color,
                                                   alpha=1.)
        for crt_poly3d in soma_poly3d:
                ax2.add_collection3d(crt_poly3d)
        for crt_poly3d in axon_poly3d:
                ax2.add_collection3d(crt_poly3d)

for neur in range(len(surround_neurons)):
        direction = surround_neurons[neur].align_dir
        direction = np.array([direction[1], direction[2], direction[0]])
        color = '0.5'
        #	soma
        soma_pos = surround_neurons[neur].soma_pos
        soma_pos = np.array([soma_pos[1], soma_pos[2], soma_pos[0]])
        axon_pos = surround_neurons[neur].soma_pos
        axon_pos = np.array([axon_pos[1], axon_pos[2], axon_pos[0]])
        # get poly soma
        soma_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                                   direction=direction,
                                                   length=soma_length,
                                                   radius=soma_radius,
                                                   n_points=n_points,
                                                   facecolor=color,
                                                   alpha=1.)
        # get poly axon
        axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                                   direction=direction,
                                                   length=axon_length,
                                                   radius=axon_radius,
                                                   n_points=n_points,
                                                   facecolor=color,
                                                   alpha=1.)
        for crt_poly3d in soma_poly3d:
                ax2.add_collection3d(crt_poly3d)
        for crt_poly3d in axon_poly3d:
                ax2.add_collection3d(crt_poly3d)

verts = []
elec_size = 5
for e in range(mea.number_electrode):
        yy = [mea.electrodes[e].position[1] - elec_size,
              mea.electrodes[e].position[1] - elec_size,
              mea.electrodes[e].position[1] + elec_size,
              mea.electrodes[e].position[1] + elec_size]
        zz = [mea.electrodes[e].position[2] + elec_size,
              mea.electrodes[e].position[2] - elec_size,
              mea.electrodes[e].position[2] - elec_size,
              mea.electrodes[e].position[2] + elec_size]
        xx = [0, 0, 0, 0]
        verts.append(list(zip(yy, zz, xx)))

jet = plt.get_cmap('jet')
colors = mea.get_currents() / np.max(np.abs(mea.get_current_matrix())) + 1
curr = ax2.add_collection3d(Poly3DCollection(verts,
#                                            zorder=1,
                                             alpha=0.8,
                                             color=jet(colors)))
currents = mea.get_currents()/1000

m = cm.ScalarMappable(cmap=cm.jet)
bounds = np.round(np.linspace(np.min(currents), np.max(currents), 7))
norm = mpl_colors.BoundaryNorm(bounds, cm.jet)
m.set_array(currents)
cbar_ax2 = plt.colorbar(m, cax=cax2, norm=norm, boundaries=bounds)
cbar_ax2.set_label('mA', rotation=270)
# ghost_axis = ax2.scatter(xx, yy, zz, color=jet(colors))
# cax2 = fig.add_subplot(gs[4, 9:])
# fig.colorbar(ghost_axis, cax=cax2)
# ghost_axis.axis('off')
# cmap = plt.cm.jet
# norm = mpl.colors.BoundaryNorm(colors, cmap)

# cb = mpl.colorbar.ColorbarBase(ax2,
#                                cmap=cax2,
#                                norm=norm)

ax2.set_xticklabels([])
ax2.set_yticklabels([])
        
# Get rid of the panes
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax2.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax2.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax2.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))

# ax2.set_xlabel('Y [um]')
# ax2.set_ylabel('Z [um]')
ax2.set_zlabel('Z [um]')

# last axis
ax0 = fig.add_subplot(gs[6:9, 0:9], projection='3d')

ax0.view_init(elev=elev, azim=azim)
ax0.dist = 10..
# plot data points.
# ax0.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
#             [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
#             marker='o', c='b', s=50, zorder=2)
ax0.set_xlabel('y ($\mu$m)', fontsize=20)
ax0.set_ylabel('z ($\mu$m)', fontsize=20)
ax0.xaxis.set_tick_params(labelsize=15, width=5)
ax0.yaxis.set_tick_params(labelsize=15, width=5)


#ax0.set_xticklabels([])
#ax0.set_yticklabels([])
ax0.set_zticks([])
# Get rid of the panes
ax0.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax0.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax0.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax0.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax0.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax0.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))

ax0.grid(b=False)

ax0.set_zlim3d(0, 0.1)

# overlay neuron soma and axon (projection on (x=10,y,z))
for neur in range(len(target_neurons)):
    axon_terminal = target_neurons[neur].get_axon_end()
    neuron_proj_ext = ax0.plot([target_neurons[neur].soma_pos[1],
                                axon_terminal[1]],
                               [target_neurons[neur].soma_pos[2],
                                axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = ax0.plot([target_neurons[neur].soma_pos[1],
                                axon_terminal[1]],
                               [target_neurons[neur].soma_pos[2],
                                axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
    trg = ax0.scatter(target_neurons[neur].soma_pos[1],
                      target_neurons[neur].soma_pos[2],
                      marker='^', c='r', s=500, zorder=2)


for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = ax0.plot([surround_neurons[neur].soma_pos[1],
                                axon_terminal[1]],
                               [surround_neurons[neur].soma_pos[2],
                                axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', '0.5', 'linewidth', 3.0)
    neuron_proj_int = ax0.plot([surround_neurons[neur].soma_pos[1],
                                axon_terminal[1]],
                               [surround_neurons[neur].soma_pos[2],
                                axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    surr = ax0.scatter(surround_neurons[neur].soma_pos[1],
                       surround_neurons[neur].soma_pos[2],
                       marker='^',
                       c='#888888',
                       s=400,
                       zorder=2)

CS = ax0.contourf(y_vec,
                  z_vec,
                  v_grid_orig,
                  zdir='z',
                  offset=-0.0001,
                  cmap=cm.coolwarm)

#plt.xlim([np.min(y_vec), np.max(y_vec)])
#plt.ylim([np.min(z_vec), np.max(z_vec)])

#cbar = plt.colorbar() # draw colorbar
#cbar.set_label('V (mV)', fontsize=20, rotation=270, labelpad=20)
# plt.legend((trg, surr),
#            ('Target', 'Surround'),
#            scatterpoints=1,
#            ncol=1,
#            loc='upper right',
#            fontsize=18)

#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('Results_3d.pdf')
plt.show()
