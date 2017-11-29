'''

Visualize and Analyze results from Monte Carlo simulation

'''

import SimulateMEA as sim
import MEAutility as MEA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors as mpl_colors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
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

#plt.ion()
simulated = False
labelsize = 50
ticksize = 40
labelsize3d= 25
ticksize3d = 12
labelpadcbar1 = 20
labelpadcbar2 = 30
labelpadxy = 25
labelpadz = 5
lwtarg = 10
lwsurr = 8
lwth = 5
max_curr = 20

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

	# meaParam: N_side, pitch, current_step, max_current, monopolaelev = 30ramp
	meaParam = [4, 15, 2, 20, monopolar_current]
	# neuronParam: close_separation, xlim, n_target, n_surround, axon_length, discrete, trg_above_thresh
	neuronParam = [15, [5, 15], [5, 30], N_target, N_surround, 15, 15, 1]
	# fitParam: vm_target, vm_surround, alpha_target_surround, alpha_energy_sparsity
	fitParam = [vm_targ, vm_surr, 0.4, 0.5]
	# gaParam: NGEN, CXPB, MUTPB, PBEST, NSTALL
	gaParam = [Ngen, 0.8, 0.1, 0.04, Nstall]
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
		v_grid_dv2_z[:, ii] = np.gradient(np.gradient(v_grid[:, ii])) / tract_image**2
		v_grid_dv2_y[ii, :] = np.gradient(np.gradient(v_grid[ii, :])) / tract_image**2

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

'''Plots'''

'''3d plot'''


fig = plt.figure(figsize=[10, 16])
gs = gridspec.GridSpec(27,
                       30,
                       hspace=0.,
                       wspace=0.)
fig.subplots_adjust(left=0.01, right=.9, top=1., bottom=0.01)
elev = 50
# elev = 22
azim = -45
dist = 10
# Add surface
y_plane, z_plane = np.meshgrid(y_vec, z_vec)

v_grid_orig = np.zeros((len(y_vec), len(z_vec)))

# maintain matrix orientation (row - z, column - y, [0,0] - top left corner)

for ii in range(len(z_vec)):
    for jj in range(len(y_vec)):
        v_grid_orig[ii, jj] = mea.compute_field(np.array(
            [15, y_plane[ii][jj], z_plane[ii][jj]]))


'''AX1'''
ax1 = fig.add_subplot(gs[0:9, 0:27], projection='3d')
cax1 = fig.add_subplot(gs[1:8, 28:29])
ax1.view_init(elev=elev, azim=azim)
ax1.set_xlim3d(-30, 30)
ax1.set_ylim3d(-30, 30)
ax1.set_zlim3d(-0.1, 30)

# Plot electrodes
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

yz = []

for e in range(mea.number_electrode):
        crt_yz = [mea.electrodes[e].position[1] - elec_size,
                  mea.electrodes[e].position[2] - elec_size]
        yz.append(crt_yz)
        
jet = plt.get_cmap('jet')
colors = mea.get_currents() / (2*max_curr*1000) + 0.5
mea_collection = []
for crt_yz, crt_color in zip(yz, colors):
        rec = Rectangle(crt_yz,
                        elec_size*2.,
                        elec_size*2.,
                        facecolor=jet(crt_color),
                        edgecolor=jet(crt_color),
                        alpha=0.7)
        ax1.add_patch(rec)
        art3d.pathpatch_2d_to_3d(rec, z=0, zdir='z')
# curr = ax1.add_collection3d(Poly3DCollection(verts,
#                                              #                                            zorder=1,
#                                              alpha=0.7,
#                                              color=jet(colors)),
#                             zs=-1,
#                             zdir='z')
currents = mea.get_currents() / 1000

m = cm.ScalarMappable(cmap=cm.jet)
bounds = np.arange(-max_curr, max_curr+1,2)
norm = mpl_colors.BoundaryNorm(bounds, cm.jet)
m.set_array(currents)
cbar_ax1 = plt.colorbar(m, cax=cax1, norm=norm, boundaries=bounds, alpha=0.7)
cbar_ax1.set_label('$\mu$A', fontsize=labelsize3d, labelpad=labelpadcbar1, rotation=0) #rotation=270,
cbar_ax1.ax.tick_params(labelsize=ticksize3d)

# ax2.dist = 10..
soma_length = 3.
soma_radius = 1.5
axon_length = 15.
axon_radius = .2
n_points = 20.
for neur in range(len(target_neurons)):
    direction = target_neurons[neur].align_dir
    # 1,2,0
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
                                               alpha=1.,
                                               flatten_along_zaxis=False)
    # get poly axon
    axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                               direction=direction,
                                               length=axon_length,
                                               radius=axon_radius,
                                               n_points=n_points,
                                               facecolor=color,
                                               alpha=1.,
                                               flatten_along_zaxis=False)
    for crt_poly3d in soma_poly3d:
        ax1.add_collection3d(crt_poly3d)
    for crt_poly3d in axon_poly3d:
        ax1.add_collection3d(crt_poly3d)

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
        ax1.add_collection3d(crt_poly3d)
    for crt_poly3d in axon_poly3d:
        ax1.add_collection3d(crt_poly3d)


ax1.set_xticklabels([])
ax1.set_yticklabels([])

# Get rid of the panes
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax1.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))

# ax1.set_xlabel('Y [um]')
# ax1.set_ylabel('Z [um]')
ax1.set_zlabel('x ($\mu$m)', fontsize=labelsize3d, labelpad=labelpadz)
ax1.tick_params(labelsize=ticksize3d)

'''AX2'''

# ax1 = fig.add_subplot(311, projection='3d')
ax2 = fig.add_subplot(gs[9:18, 0:27], projection='3d')# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)

ax2.view_init(elev=elev, azim=azim)
surf1 = ax2.plot_surface(y_plane,
                         z_plane,
                         v_grid_orig,
                         cmap=cm.coolwarm,
                         alpha=0.3,
                         zorder=0,
                         antialiased=True)
# ax2.contour(y_plane,
#             z_plane,
#             v_grid_orig,
#             cmap=cm.coolwarm,
#             extend3d=True,)

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
# Get rid of the panes
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax2.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax2.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax2.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))
# ax2.dist = 10..
cax2 = fig.add_subplot(gs[10:17, 28:29])
cbar_ax2 = fig.colorbar(surf1, cax=cax2)
cbar_ax2.set_label('mV', fontsize=labelsize3d, labelpad=labelpadcbar2, rotation=0)
cbar_ax2.ax.tick_params(labelsize=ticksize3d)



'''AX3'''
# last axis
ax3 = fig.add_subplot(gs[18:25, 0:27], projection='3d')

ax3.view_init(elev=elev, azim=azim)
# ax3.dist = 10..
# plot data points.
# ax3.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
#             [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
#             marker='o', c='b', s=50, zorder=2)
ax3.set_xlabel('y ($\mu$m)', fontsize=labelsize3d, labelpad=labelpadxy)
ax3.set_ylabel('z ($\mu$m)', fontsize=labelsize3d, labelpad=labelpadxy)
ax3.xaxis.set_tick_params(labelsize=ticksize3d, width=5)
ax3.yaxis.set_tick_params(labelsize=ticksize3d, width=5)

# ax3.set_xticklabels([])
# ax3.set_yticklabels([])
ax3.set_zticks([])
# Get rid of the panes
ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax3.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax3.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax3.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))

ax3.grid(False)

#ax3.set_zlim3d(-0.1, 0.5)

CS = ax3.contourf(y_vec,
                  z_vec,
                  v_grid_orig,
                  zdir='z',
                  offset=-0.05,
                  cmap=cm.coolwarm,
                  alpha=0.5,
                  zorder=0)

for neur in range(len(target_neurons)):
    direction = target_neurons[neur].align_dir
    # 1,2,0
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
                                               alpha=1.,
                                               flatten_along_zaxis=True)
    # get poly axon
    axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                               direction=direction,
                                               length=axon_length,
                                               radius=axon_radius,
                                               n_points=n_points,
                                               facecolor=color,
                                               alpha=1.,
                                               flatten_along_zaxis=True)
    for crt_poly3d in soma_poly3d:
        ax3.add_collection3d(crt_poly3d)
    for crt_poly3d in axon_poly3d:
        ax3.add_collection3d(crt_poly3d)


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
                                               alpha=1.,
                                               flatten_along_zaxis=True)
    # get poly axon
    axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
                                               direction=direction,
                                               length=axon_length,
                                               radius=axon_radius,
                                               n_points=n_points,
                                               facecolor=color,
                                               alpha=1.,
                                               flatten_along_zaxis=True)
    for crt_poly3d in soma_poly3d:
        ax3.add_collection3d(crt_poly3d)
    for crt_poly3d in axon_poly3d:
        ax3.add_collection3d(crt_poly3d)

ax3.set_autoscalez_on(True)
ax3.set_zlim3d(-0.1, 0.1)
# for neur in range(len(target_neurons)):
#     axon_terminal = target_neurons[neur].get_axon_end()
#     neuron_proj_ext = ax3.plot([target_neurons[neur].soma_pos[1], axon_terminal[1]], [target_neurons[neur].soma_pos[2], axon_terminal[2]], [0, 0])
#     plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 5.0, 'zorder', 50)
#     neuron_proj_int = ax3.plot([target_neurons[neur].soma_pos[1], axon_terminal[1]],
#                                [target_neurons[neur].soma_pos[2], axon_terminal[2]], [0, 0])
#     plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 4.5, 'zorder', 50)
#     # neuron_proj_int = plt.plot([target_neurons[neur].soma_pos[1], axon_terminal[1]], [target_neurons[neur].soma_pos[2], axon_terminal[2]])
#     # plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
#     # plt.scatter(target_neurons[neur].soma_pos[1], target_neurons[neur].soma_pos[2], marker='^', c='k', s=600)
#     # plt.scatter(target_neurons[neur].soma_pos[1], target_neurons[neur].soma_pos[2], marker='^', c='r', s=500)


# for neur in range(len(surround_neurons)):
#     axon_terminal = surround_neurons[neur].get_axon_end()
#     neuron_proj_ext = ax3.plot([surround_neurons[neur].soma_pos[1], axon_terminal[1]], [surround_neurons[neur].soma_pos[2], axon_terminal[2]], [0, 0])
#     plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 5.0, 'zorder', 50)
#     neuron_proj_int = ax3.plot([surround_neurons[neur].soma_pos[1], axon_terminal[1]],
#                                [surround_neurons[neur].soma_pos[2], axon_terminal[2]], [0, 0])
#     plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 4.5, 'zorder', 50)
#     # neuron_proj_int = plt.plot([surround_neurons[neur].soma_pos[1], axon_terminal[1]], [surround_neurons[neur].soma_pos[2], axon_terminal[2]])
#     # plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
#     # plt.scatter(surround_neurons[neur].soma_pos[1], surround_neurons[neur].soma_pos[2], marker='^', c='k', s=400)
#     # plt.scatter(surround_neurons[neur].soma_pos[1], surround_neurons[neur].soma_pos[2], marker='^', c='#888888', s=400)
#plt.show()    
plt.savefig('Results.pdf')    
#
# soma_radius = .005
# axon_radius = .005
#
# for neur in range(len(target_neurons)):
#     direction = target_neurons[neur].align_dir
#     # 1,2,0
#     direction = np.array([direction[1], direction[2], 0])
#     color = 'r'
#     #	soma
#     soma_pos = target_neurons[neur].soma_pos
#     soma_pos = np.array([soma_pos[1], soma_pos[2], 0])
#     axon_pos = target_neurons[neur].soma_pos
#     axon_pos = np.array([axon_pos[1], axon_pos[2], 0])
#
#     # get poly soma
#     soma_poly3d = tb.get_polygons_for_cylinder(soma_pos,
#                                                direction=direction,
#                                                length=soma_length,
#                                                radius=soma_radius,
#                                                n_points=n_points,
#                                                facecolor=color,
#                                                alpha=1.)
#     # get poly axon
#     axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
#                                                direction=direction,
#                                                length=axon_length,
#                                                radius=axon_radius,
#                                                n_points=n_points,
#                                                facecolor=color,
#                                                alpha=1.)
#     for crt_poly3d in soma_poly3d:
#         ax3.add_collection3d(crt_poly3d)
#     for crt_poly3d in axon_poly3d:
#         ax3.add_collection3d(crt_poly3d)
#
# for neur in range(len(surround_neurons)):
#     direction = surround_neurons[neur].align_dir
#     direction = np.array([direction[1], direction[2], 0])
#     color = '0.5'
#     #	soma
#     soma_pos = surround_neurons[neur].soma_pos
#     soma_pos = np.array([soma_pos[1], soma_pos[2], 0])
#     axon_pos = surround_neurons[neur].soma_pos
#     axon_pos = np.array([axon_pos[1], axon_pos[2], 0])
#     # get poly soma
#     soma_poly3d = tb.get_polygons_for_cylinder(soma_pos,
#                                                direction=direction,
#                                                length=soma_length,
#                                                radius=soma_radius,
#                                                n_points=n_points,
#                                                facecolor=color,
#                                                alpha=1.)
#     # get poly axon
#     axon_poly3d = tb.get_polygons_for_cylinder(soma_pos,
#                                                direction=direction,
#                                                length=axon_length,
#                                                radius=axon_radius,
#                                                n_points=n_points,
#                                                facecolor=color,
#                                                alpha=1.)
#     for crt_poly3d in soma_poly3d:
#         ax3.add_collection3d(crt_poly3d)
#     for crt_poly3d in axon_poly3d:
#         ax3.add_collection3d(crt_poly3d)
#
#
#





'''Second derivative along neuron'''

der2filter = [1, -2, 1]
axon_length = 15

samples = 15.0
tract_up = float(axon_length)/(samples-1)
steps = np.linspace(0, axon_length , int(samples))
axon = np.linspace(0, axon_length, int(samples))
axon2 = np.linspace(0, axon_length,int(samples-2))

fig2 = plt.figure()
ax2 = plt.subplot(111,
                  xlim=[0, axon_length],
                  ylim=[-2, 4])
ax2.set_xlabel('Neuron ($\mu$m)', fontsize=labelsize)
ax2.set_ylabel('AF mV$^2$/$\mu$m$^2$', fontsize=labelsize)
locs = range(-1,4)
ax2.set_yticks(locs)
ax2.tick_params(axis='y', which='major', labelsize=ticksize)
ax2.tick_params(axis='x', which='major', labelsize=ticksize)
# plt.title('AF along neurons', fontsize = 20)

print 'TARGET NEURON(S)'
for neur in range(len(target_neurons)):

    x_target_upsampled = np.array([target_neurons[neur].soma_pos + st * target_neurons[neur].align_dir for st in steps])

    v_axon_up = mea.compute_field(x_target_upsampled)
    dv2_axon = np.convolve(v_axon_up, der2filter, 'valid') / tract_up**2

    ax2.plot(axon2, dv2_axon, c='r', ls='-', lw=lwtarg, label='Target', zorder=10)

    print 'target n: ', neur+1

print 'SURROUNDING NEURON(S)'
for neur in range(len(surround_neurons)):

    x_surround_upsampled = np.array([surround_neurons[neur].soma_pos + st * surround_neurons[neur].align_dir for st in steps])

    v_axon_up = mea.compute_field(x_surround_upsampled)
    dv2_axon = np.convolve(v_axon_up, der2filter, 'valid') / tract_up**2
    if neur==0:
        surrline = ax2.plot(axon2, dv2_axon, c='grey', ls='--', lw=lwsurr, label='Surround', zorder=10)
    else:
        surrline = ax2.plot(axon2, dv2_axon, c='grey', ls='--', lw=lwsurr, zorder=10)
    print 'surround n: ', neur+1

# ax2.yaxis.grid(True, linestyle='-', which='major', color='grey',
#                    alpha=0.7)

# Thresholds

ax2b = ax2.twinx()
ax2b.axhline(y=1.5, c="b", ls=':', lw=lwth, alpha=1, zorder=0)
ax2b.axhline(y=0.5, c="b", ls=':', lw=lwth, alpha=1, zorder=0)
#ax2b.set_axisbelow(True)
# ax2b.set_ylabel('Thresholds', color='b', fontsize=labelsize)

locs = [0.5, 1.5]
ax2b.set_yticks(locs)
ax2b.set_ylim([-2, 4])
ax2b.tick_params(axis='y', which='major', labelsize=ticksize)
for tl in ax2b.get_yticklabels():
    tl.set_color('b')

# ax2.set_aspect(2)
# ax2b.set_aspect(2)

leg = ax2.legend(fontsize=labelsize)

ax2.grid(True)


if simulated:

    # Getting back the objects:

    # file string
    file_name = os.path.abspath(
        'Simulation_Output/'
        'Simulation_Nsim_1000_Ngen_300_Nstall_100_Ntrg_1_Nsrr_4_Mono_'
        '-10000_vtarg_1.5_vsurr_0.5_trgAboveThresh_1_discrete_15.pickle')

    with open(file_name) as f:  # Python 3: open(..., 'rb')
        obj_field_name, obj_field = pickle.load(f)

    axon_dist = obj_field[0]
    div = obj_field[1]
    overlap = obj_field[2]
    ga_null = obj_field[3]
    ga_mean_curr = obj_field[4]

    ga_max_targ = obj_field[5]
    ga_max_surr = obj_field[6]
    ga_min_targ = obj_field[7]
    ga_min_surr = obj_field[8]

    ga_mean_targ = obj_field[9]
    ga_mean_surr = obj_field[10]
    ga_median_targ = obj_field[11]
    ga_median_surr = obj_field[12]
    ga_sd_targ = obj_field[13]
    ga_sd_surr = obj_field[14]

    mono_max_targ = obj_field[15]
    mono_max_surr = obj_field[16]
    mono_min_targ = obj_field[17]
    mono_min_surr = obj_field[18]

    mono_mean_targ = obj_field[19]
    mono_mean_surr = obj_field[20]
    mono_median_targ = obj_field[21]
    mono_median_surr = obj_field[22]
    mono_sd_targ = obj_field[23]
    mono_sd_surr = obj_field[24]

    bi_max_targ = obj_field[25]
    bi_max_surr = obj_field[26]
    bi_min_targ = obj_field[27]
    bi_min_surr = obj_field[28]

    bi_mean_targ = obj_field[29]
    bi_mean_surr = obj_field[30]
    bi_median_targ = obj_field[31]
    bi_median_surr = obj_field[32]
    bi_sd_targ = obj_field[33]
    bi_sd_surr = obj_field[34]


    '''Linear Regression'''

    slope_ga_surr, intercept_ga_surr, r_value_ga_surr, p_value_ga_surr, std_err_ga_surr = \
        stats.linregress(overlap, np.max(ga_max_surr, 1))

    slope_mono_surr, intercept_mono_surr, r_value_mono_surr, p_value_mono_surr, std_err_mono_surr = \
        stats.linregress(overlap, np.max(mono_max_surr, 1))

    slope_bi_surr, intercept_bi_surr, r_value_bi_surr, p_value_bi_surr, std_err_bi_surr = \
        stats.linregress(overlap, np.max(bi_max_surr, 1))

    X_overlap = np.linspace(0, np.max(overlap), num=50)
    Y_ga = slope_ga_surr*X_overlap + intercept_ga_surr
    Y_mono = slope_mono_surr*X_overlap + intercept_mono_surr
    Y_bi = slope_bi_surr*X_overlap + intercept_bi_surr

    sim = range(len(ga_mean_targ))

    fig1 = plt.figure()
    ax1 = plt.subplot(111,
                      xlim = [-10, len(ga_mean_targ) + 10])

    plt.xlabel('# scenario', fontsize=20)
    plt.ylabel('dv2', fontsize=20)
    plt.title('Maximum surround AF', fontsize=labelsize)


    mono = ax1.scatter(sim, np.max(mono_max_surr, 1), c='c', marker='*', s=80)
    bi = ax1.scatter(sim, np.max(bi_max_surr, 1), c='g', marker='^', s=80)
    ga = ax1.scatter(sim, np.max(ga_max_surr, 1), c='r', marker='o', s=50)

    ax1.axhline(y=1.5 ,xmin=-10,xmax=len(ga_mean_targ) + 10, c="k", linewidth=5, zorder=0)
    ax1.axhline(y=0.5 ,xmin=-10,xmax=len(ga_mean_targ) + 10, c="k", linewidth=2, zorder=5)

    plt.legend((ga, mono, bi),
               ('GA', 'Monopolar', 'Bipolar'),
               scatterpoints=1,
               ncol=1,
               fontsize=18)

    plt.grid()

    fig4 = plt.figure()
    ax7 = plt.subplot(111)
    # plt.ylabel('AF mV$^2$/$\mu$m$^2$', fontsize=ticksize)
    #plt.title('Maximum Surround AF', fontsize=ticksize)
    labels = ['GA', 'MONO', 'BI']
    ga_points = np.max(ga_max_surr, 1)[np.where(np.max(ga_max_surr, 1)<10)]
    mono_points = np.max(mono_max_surr, 1)[np.where(np.max(mono_max_surr, 1) < 10)]
    bi_points = np.max(bi_max_surr, 1)[np.where(np.max(bi_max_surr, 1) < 10)]
    bp = ax7.boxplot([ga_points, mono_points, bi_points],
                labels=labels, widths = 0.6, notch=0)#,vert=2, whis=1.5)
    plt.setp(bp['boxes'], color='black', lw = 2.5)
    plt.setp(bp['whiskers'], color='black', lw = 2.5)
    plt.setp(bp['caps'], color = 'black', lw = 2.5)
    plt.setp(bp['fliers'], color='black', marker='o', markersize=6)
    plt.setp(bp['medians'], color = 'OrangeRed', linewidth = 3.5)
    ## change the style of fliers and their fill
    # for flier in bp['fliers']:
    #     flier.set(marker='o', color='#ffffff', alpha=1, markersize=8)

    ax7.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.8)
    ax7.set_axisbelow(True)
    # ax7.set_xlabel('Stimulation Approach', fontsize=ticksize)
    ax7.set_ylabel('AF mV$^2$/$\mu$m$^2$', fontsize=labelsize)
    xtickNames = plt.setp(ax7, xticklabels=labels)
    plt.setp(xtickNames, rotation=45, fontsize=labelsize)
    ax7.tick_params(axis='y', which='major', labelsize=ticksize)

    ax7.axhline(y=1.5, c="b", ls='-', lw=2, alpha=0.8)
    ax7.set_ylim([-1, 10.1])

    ax7.set_aspect(0.1)


    # number of outliers
    print len(np.where(np.max(ga_max_surr, 1)>=10)[0])
    print len(np.where(np.max(mono_max_surr, 1) >= 10)[0])
    print len(np.where(np.max(bi_max_surr, 1) >= 10)[0])

    #trg
    print np.mean(ga_mean_targ)
    print np.mean(ga_median_targ)
    print np.mean(ga_max_targ)
    print np.mean(ga_min_targ)
    print np.mean(ga_sd_targ)
    print len(np.where(ga_max_targ > 1.5)[0])/10.

    print np.mean(mono_mean_targ)
    print np.mean(mono_median_targ)
    print np.mean(mono_max_targ)
    print np.mean(mono_min_targ)
    print np.mean(mono_sd_targ)
    print len(np.where(mono_max_targ > 1.5)[0])/10.

    print np.mean(bi_mean_targ)
    print np.mean(bi_median_targ)
    print np.mean(bi_max_targ)
    print np.mean(bi_min_targ)
    print np.mean(bi_sd_targ)
    print len(np.where(bi_max_targ > 1.5)[0])/10.
    
    #surr
    print np.mean(np.max(ga_mean_surr,1))
    print np.mean(np.max(ga_median_surr, 1))
    print np.mean(np.max(ga_max_surr, 1))
    print np.mean(np.max(ga_min_surr, 1))
    print np.mean(np.max(ga_sd_surr, 1))
    print len(np.where(ga_max_surr > 1.5)[0])/10.

    print np.mean(np.max(mono_mean_surr, 1))
    print np.mean(np.max(mono_median_surr, 1))
    print np.mean(np.max(mono_max_surr, 1))
    print np.mean(np.max(mono_min_surr, 1))
    print np.mean(np.max(mono_sd_surr, 1))
    print len(np.where(mono_max_surr > 1.5)[0])/10.

    print np.mean(np.max(bi_mean_surr, 1))
    print np.mean(np.max(bi_median_surr, 1))
    print np.mean(np.max(bi_max_surr, 1))
    print np.mean(np.max(bi_min_surr, 1))
    print np.mean(np.max(bi_sd_surr, 1))
    print len(np.where(bi_max_surr > 1.5)[0]) / 10.


    print 'Total current: ', (16-np.mean(ga_null))*np.mean(ga_mean_curr)
    print len(np.where(ga_null==15)[0])/10.
    print len(np.where(ga_null >12)[0]) / 10.

    plt.figure()
    ax3 = plt.subplot(121)
    ax3.hist(ga_null)
    ax4 = plt.subplot(122)
    ax4.hist((16-ga_null)*ga_mean_curr, color='g')

    print np.median((16-ga_null)*ga_mean_curr)




#
# # Sample Scenario
# N_target = 1
# N_surround = 4
# Ngen = 1
# Nstall = 100
# N_sim = 5
# monopolar_current = -10000
# vm_targ = 1.5
# vm_surr = 0.2
#
# # meaParam: N_side, pitch, current_step, max_current, monopolaramp
# meaParam = [4, 15, 2, 20, monopolar_current]
# # neuronParam: close_separation, xlim, n_target, n_surround, axon_length, discrete
# neuronParam = [15, [15, 30], [5, 30], N_target, N_surround, 15, 3, 15]
# # fitParam: vm_target, vm_surround, alpha_target_surround, alpha_energy_sparsity
# fitParam = [vm_targ, vm_surr, 0.4, 0.5]
# # gaParam: NGEN, CXPB, MUTPB, PBEST, NSTALL
# gaParam = [Ngen, 0.8, 0.1, 0.04, 100]
# # mutselParam: muGauss, sdGauss, pgauss, pzero, tourn_size
# mutselParam = [0, 4000, 0.2, 0.2, 3]
#
# s = sim.SimulateScenario()
#
# [complexity, performances, currents, neurons] = s.simulate_scenario(meaParam, neuronParam,
#                                                                     fitParam, gaParam, mutselParam, verbose=True)
# pitch = 15
# mea = MEA.SquareMEA(dim=4, pitch=pitch)
# target_neurons = neurons[0]
# surround_neurons = neurons[1]
#
# mea.set_current_matrix(currents[0])
#
# unit = 1
# bound = abs(mea[0][0].position[1]) + pitch
# x_vec = np.arange(1, bound, unit)
# y_vec = np.arange(-bound, bound, unit)
# z_vec = np.arange(-bound, bound, unit)
#
# x, y, z = np.meshgrid(x_vec, y_vec, z_vec)
#
# v_grid = np.zeros((len(y_vec), len(z_vec)))
#
# # maintain matrix orientation (row - z, column - y, [0,0] - top left corner)
# z_vec = z_vec[::-1]
#
# for ii in range(len(z_vec)):
#     for jj in range(len(y_vec)):
#         v_grid[ii, jj] = mea.compute_field(np.array([15, y_vec[jj], z_vec[ii]]))
#         # print np.array([10, y_vec[jj], z_vec[ii]])
#
# v_grid_dv2_z = np.zeros((len(y_vec), len(z_vec)))
# v_grid_dv2_y = np.zeros((len(y_vec), len(z_vec)))
#
# tract_image = (max(y_vec)-min(y_vec)) / (len(y_vec) - 1)
#
# for ii in range(len(y_vec)):
#     v_grid_dv2_z[:, ii] = np.gradient(np.gradient(v_grid[:, ii])) / tract_image**2
#     v_grid_dv2_y[ii, :] = np.gradient(np.gradient(v_grid[ii, :])) / tract_image**2
