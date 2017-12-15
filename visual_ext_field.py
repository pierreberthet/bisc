import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt


# Parameters for the external field
sigma = 0.3
# source_xs = np.array([-50, -50, -10, -10, 10, 10, 50, 50])
# source_ys = np.array([-50, 50, -10, 10, 10, -10, -50, 50])
cortical_surface_height = 50

source_xs = np.array([-50, 0, 50, 0, 0])
source_ys = np.array([0, 50, 0, -50, 0])

# source_geometry = np.array([0, 0, 1, 1, 1, 1, 0, 0])
stim_amp = 1.
n_stim_amp = -stim_amp / 4
source_geometry = np.array([-stim_amp / 4, -stim_amp / 4, -stim_amp / 4, -stim_amp / 4, stim_amp])
source_zs = np.ones(len(source_xs)) * cortical_surface_height


# Stimulation Parameters:
max_current = -100000.   # mA
current_resolution = 100
# amp_range = np.exp(np.linspace(1, np.log(max_current), current_resolution))
amp_range = np.linspace(1, max_current, current_resolution)
amp = amp_range[0]

for loop in range(current_resolution):
    # loop for various geometries
    source_amps = source_geometry * amp_range[loop]
    ExtPot = utils.ImposedPotentialField(source_amps, source_xs, source_ys, source_zs, sigma)
    plot_field_length = 500
    v_field_ext_xz = np.zeros((100, 100))
    xf = np.linspace(-plot_field_length, plot_field_length, 100)
    zf = np.linspace(-plot_field_length, cortical_surface_height, 100)
    for xidx, x in enumerate(xf):
        for zidx, z in enumerate(zf):
            v_field_ext_xz[xidx, zidx] = ExtPot.ext_field(x, 0, z)

    vmin = -1000
    vmax = 1000
    logthresh = 0

    fig = plt.figure(figsize=[18, 7])
    fig.suptitle('CURRENT = {0} mA'.format((amp_range[loop]) / 1000.))
    fig.subplots_adjust(wspace=.6)

    imshow_dict = dict(origin='lower', interpolation='nearest',
                       cmap=plt.cm.bwr, vmin=vmin, vmax=vmax,
                       norm=matplotlib.colors.SymLogNorm(10**-logthresh))

    ax2 = plt.subplot(131, title="V_ext", xlabel="x [$\mu$m]", ylabel='z [$\mu$m]')
    img1 = ax2.imshow(v_field_ext_xz.T,
                      extent=[-plot_field_length, plot_field_length,
                              -plot_field_length, cortical_surface_height],
                      **imshow_dict)
    # cax = plt.axes([0.4, 0.1, 0.01, 0.33])
    # cb = plt.colorbar(img1)
    # cb.set_ticks(tick_locations)
    # cb.set_label('mV', labelpad=-10)

    ax2.scatter(source_xs, np.ones(len(source_xs)) * cortical_surface_height, c=source_amps, s=100, vmin=-1.4,
                vmax=1.4, edgecolor='k', lw=2, cmap=plt.cm.bwr, clip_on=False)

    [ax2.scatter(source_xs[i], np.ones(len(source_xs))[i] * cortical_surface_height,
                 marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)]
    [ax2.scatter(source_xs[i], np.ones(len(source_xs))[i] * cortical_surface_height,
                 marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)]

    ax3 = plt.subplot(122, title="Current source geometry", xlabel="x [$\mu$m]", ylabel='y [$\mu$m]')
    ax3.scatter(source_xs, source_ys, c=source_amps, s=100, vmin=-1.4, vmax=1.4,
                edgecolor='k', lw=2, cmap=plt.cm.bwr)
    [ax3.scatter(source_xs[i], source_ys[i], marker='_', s=50, lw=2, c='k') for i in np.where(source_amps > 0)]
    [ax3.scatter(source_xs[i], source_ys[i], marker='+', s=50, lw=2, c='k') for i in np.where(source_amps < 0)]

    v_field_ext_xy = np.zeros((100, 100))
    xf = np.linspace(-plot_field_length, plot_field_length, 100)
    yf = np.linspace(-plot_field_length, plot_field_length, 100)
    for xidx, x in enumerate(xf):
        for yidx, y in enumerate(yf):
            v_field_ext_xy[xidx, yidx] = ExtPot.ext_field(x, y, cortical_surface_height)

    img2 = ax3.imshow(v_field_ext_xy.T,
                      extent=[-plot_field_length, plot_field_length,
                              -plot_field_length, plot_field_length],
                      **imshow_dict)
    cax = plt.axes([0.335, 0.26, 0.01, 0.45])
    cb = plt.colorbar(img2, cax=cax)
    # cb.set_ticks(tick_locations)
    cb.set_label('mV', labelpad=-10)
    plt.savefig("outputs/visual_ext_field" + str(loop) + ".png")
    plt.close()
