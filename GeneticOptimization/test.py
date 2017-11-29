from MEAutility import *
from scipy.interpolate import griddata

import matplotlib.pyplot as plt


mea = SquareMEA(dim=10, pitch=10, x_plane=0)

# mea.set_random_currents(amp=1e4)

mea[2][2].current = 1e4
mea[2][3].current = 1e4
mea[3][2].current = 1e4
mea[3][3].current = 1e4

mea[6][6].current = -1e4
mea[6][7].current = -1e4
mea[7][6].current = -1e4
mea[7][7].current = -1e4

curr_mat = mea.get_current_matrix()

print "Accessing electrode in i=5, j=7: position - ", mea[5][7].position, " current - ", mea[5][7].current


''' Test neuron '''
soma = np.array([15, 0, 0])
align = np.array([0, 1, 1])
neur = GeometricNeuron(soma, align, length=40)
xp = neur.get_axon_points()

axon = np.linspace(0, neur.length, neur.points)
v_axon = mea.compute_field(xp)
dv_axon = np.gradient(v_axon)
dv2_axon = np.gradient(dv_axon)

fig1 = plt.figure()
plt.subplot(3, 1, 1)
plt.plot(axon, v_axon, 'b.-')
plt.title('V along axon')
plt.ylabel('mV')

plt.subplot(3, 1, 2)
plt.plot(axon, dv_axon, 'g.-')
plt.title('dV along axon')
plt.ylabel('mV/um')

plt.subplot(3, 1, 3)
plt.plot(axon, dv2_axon, 'r.-')
plt.title('dV2 along axon')
plt.ylabel('mV2/um2')


# Create mesh
unit = 5
bound = 80
x_vec = np.arange(1, bound, unit)
y_vec = np.arange(-bound, bound, unit)
z_vec = np.arange(-bound, bound, unit)

x, y, z = np.meshgrid(x_vec, y_vec, z_vec)
y_plane, z_plane = np.meshgrid(y_vec, z_vec)

positions = np.vstack([y_plane.ravel(), z_plane.ravel()])
pos = np.append(positions, [10*np.ones(positions.shape[1])], 0)
pos = np.transpose(pos)

print 'Computing Field on ', pos.shape[0], ' x-y points'
v_p = mea.compute_field(pos)

v_grid = np.zeros((len(y_vec), len(z_vec)))

for ii in range(len(y_vec)):
    for jj in range(len(z_vec)):
        v_grid[ii, jj] = mea.compute_field(np.array([10, y_vec[ii], z_vec[jj]]))

# v_grid = griddata((pos[:, 0], pos[:, 1]), v_p, (y_vec[:, None], z_vec[None, :]), method='cubic')
#
# print 'Computed: ', len(v_p), ' field values and gridded to ', v_grid.shape

fig2 = plt.figure()
CS = plt.contour(y_vec, z_vec, v_grid/1000, linewidths=1, colors='k')
CS = plt.contourf(y_vec, z_vec, v_grid/1000, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('Potential in proximity of electrodes: z = 10 um')

# overlay neuron soma and axon (projection on (x=10,y,z))
axon_terminal = neur.get_axon_end()
neuron_proj_ext = plt.plot([soma[1], axon_terminal[1]], [soma[2], axon_terminal[2]])
plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 5.0)
neuron_proj_int = plt.plot([soma[1], axon_terminal[1]], [soma[2], axon_terminal[2]])
plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 4.0)

plt.scatter(soma[1], soma[2], marker='^', c='k', s=600)
plt.scatter(soma[1], soma[2], marker='^', c='r', s=500)


plt.matshow(mea.get_current_matrix())
plt.colorbar()

plt.show()



