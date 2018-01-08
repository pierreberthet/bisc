# import LFPy as lfpy
import numpy as np
import matplotlib.pyplot as plt


def built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, widx, rank):
    '''
    Return a dict of array useful for plotting in parallel simulation
    (cell objet can not be communicated directly between thread).
    '''
    # c_idxs = [plt.cm.jet( 1.*v_idxs.index(idx) / len(v_idxs) for idx in v_idxs  )]
    return {'totnsegs': cell.totnsegs, 'glb_vext': glb_vext, 'glb_vmem': glb_vmem, 'v_idxs': v_idxs,
            'name': cell.get_idx_name(v_idxs[widx])[1], 'rank': rank,
            'xstart': cell.xstart, 'ystart': cell.ystart, 'zstart': cell.zstart,
            'xmid': cell.xmid, 'ymid': cell.ymid, 'zmid': cell.zmid,
            'xend': cell.xend, 'yend': cell.yend, 'zend': cell.zend}


def built_for_mpi_space(cell, rank, extra=None):
    '''
    Return a dict of array useful for plotting cells in 3D space, in parallel simulation
    (cell objet can not be communicated directly between thread).
    '''
    return {'totnsegs': cell.totnsegs, 'rank': rank, 'vmem': cell.vmem, 'vext': cell.v_ext, 'extra': extra,
            'xstart': cell.xstart, 'ystart': cell.ystart, 'zstart': cell.zstart,
            'xmid': cell.xmid, 'ymid': cell.ymid, 'zmid': cell.zmid,
            'xend': cell.xend, 'yend': cell.yend, 'zend': cell.zend}


def return_first_spike_time_and_idx(vmem):
    '''
    Return the index of the segment where Vmem first crossed the threshold (Usually set at -20 mV.
    If many segments crossed threshold during a unique time step, it returns the index of the segment
    with the most depolarized membrane value.
    Also contains the time step when this occurred.
    '''
    if np.max(vmem) < -20:
        print "No spikes detected"
        return [None, None]
    for t_idx in range(1, vmem.shape[1]):
        if np.max(vmem.T[t_idx]) > -20:
            print np.argmax(vmem.T[t_idx])
            return [t_idx, np.argmax(vmem.T[t_idx])]
    # crossings = []
    # for comp_idx in range(vmem.shape[0]):
    #     for t_idx in range(1, vmem.shape[1]):
    #         if vmem[comp_idx, t_idx - 1] < -20 <= vmem[comp_idx, t_idx]:
    #             crossings.append([t_idx, comp_idx])
    # crossings = np.array(crossings)
    # first_spike_comp_idx = np.argmin(crossings[:, 0])
    # return crossings[first_spike_comp_idx]


def reposition_stick_horiz(cell, x=0, y=0, z=0):
    '''
    Only for stick models?
    rotate the cell model by a 1/4 of a turn, from perpendicular to the x axis to parallel,
    and reposition the cell along the x-axis in order to have it centered around the provided x position, or 0.
    '''
    l_cell = np.max(cell.zstart) + np.abs(np.min(cell.zend))
    # l_cell = cell.length
    cell.set_pos(x=x + int(l_cell / 2), y=y, z=z)
    cell.set_rotation(y=np.pi / 2)
    return


def reposition_stick_flip(cell, x=0, y=0, z=0):
    '''
    Only for stick models?
    Flip upside down the cell model, thus keeping it perpendicular to the x axis,
    and reposition the cell to avoid any displacement on the z-axis.
    (zmin and zend are unchanged)
    '''
    l_cell = np.max(cell.zstart) + np.abs(np.min(cell.zend))
    cell.set_pos(x=x, y=y, z=z - l_cell)
    cell.set_rotation(y=np.pi)
    return


def create_bisc_array():
    '''
    to be edited from create_array_shape()
    '''

    return bisc_array


def compute_dderivative():
    '''
    Compute the double derivative of the electric field
    '''

    return dderivative


def plot_segment_vmem(cell, seg_name):

    plt.figure()
    for name in cell.allsecnames:
        print name
        plt.plot(cell.tvec, cell.vmem[cell.get_idx(name)[0]])

    plt.show()

    return


def get_limit(multid_array):
    ma = multid_array[0][0]
    mi = ma

    for i in range():
        if np.min() < mi:
            mi = np.min()

        if np.max() > ma:
            ma = np.max()

    return mi, ma


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
            ef += self.source_amps[s_idx] / (4 * np.pi * self.sigma * np.sqrt(
                (self.source_xs[s_idx] - x) ** 2 +
                (self.source_ys[s_idx] - y) ** 2 +
                (self.source_zs[s_idx] - z) ** 2))
        return ef


def sanity_vext(vext, t):
    vxmin = 0
    vxmax = 0
    for i in range(len(t)):
        for j in range(len(vext)):
            lmin = np.min(vext[j][i])
            lmax = np.max(vext[j][i])
            if lmin < vxmin:
                vxmin = lmin
            if lmax > vxmax:
                vxmax = lmax
    return [vxmin, vxmax]


def create_array_shape(shape=None, pitch=None):
    '''
    create current sources of various shapes.
    {To implement: specifiy number as argument}
    '''
    if not pitch:
        pitch = 15

    if shape:  # create monopole
        if shape == 'dipole':
            n_elec = 2
            polarity = np.array([-1, 1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-10, 10])
            source_ys = np.array([0, 0])

        elif shape == 'line':
            n_elec = 4
            polarity = np.array([-1, 1, 1, -1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-2 * pitch, -pitch, pitch, 2 * pitch])
            source_ys = np.array([0, 0, 0, 0])

        elif shape == 'multipole':
            n_elec = 8
            polarity = np.array([-1, -1, 1, 1, 1, 1, -1, -1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-50, -50, -10, -10, 10, 10, 50, 50])
            source_ys = np.array([-50, 50, -10, 10, 10, -10, -50, 50])

        elif shape == 'quadrupole':
            n_elec = 5
            polarity = np.array([-1 / 4, -1 / 4, 1, -1 / 4, -1 / 4])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-20, 0, 0, 0, 20])
            source_ys = np.array([0, 20, 0, -20, 0])

        elif shape == 'monopole':
            n_elec = 1
            polarity = np.array([1])
            source_xs = np.array([0])
            source_ys = np.array([0])
            source_zs = np.array([0])

        elif shape == 'monosquare':
            n_elec = 9
            pos = 1.
            neg = -pos / (n_elec - 1)
            polarity = np.array([neg, neg, neg, neg, pos, neg, neg, neg, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, -pitch, -pitch, 0, 0, 0, pitch, pitch, pitch])
            source_ys = np.array([pitch, 0, -pitch, pitch, 0, -pitch, pitch, 0, -pitch])

        elif shape == 'across':
            n_elec = 9
            pos = 1. / 5
            neg = -1 / 4
            polarity = np.array([neg, pos, neg, pos, pos, pos, neg, pos, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, -pitch, -pitch, 0, 0, 0, pitch, pitch, pitch])
            source_ys = np.array([pitch, 0, -pitch, pitch, 0, -pitch, pitch, 0, -pitch])

        elif shape == 'minicross':
            n_elec = 4
            pos = 1. / 2
            neg = -pos
            polarity = np.array([neg, pos, pos, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch * 2, 0, 0, pitch * 2])
            source_ys = np.array([0, -pitch, pitch, 0])

        elif shape == 'circle':
            n_elec = 50
            polarity = []  # np.ones(n_elec)
            source_zs = np.zeros(n_elec * 2)    # double to account for positive and negative
            size_bisc = 1000
            xs = np.cos(np.linspace(-np.pi, np.pi, n_elec))
            ys = np.sin(np.linspace(-np.pi, np.pi, n_elec))
            x_mesh = range(-size_bisc / 2, (size_bisc + pitch) / 2, pitch)
            y_mesh = range(-size_bisc / 2, (size_bisc + pitch) / 2, pitch)

            r_i = 300  # um, radius of the inner circle
            r_e = r_i + pitch  # um, radius of the external circle
            if 2 * np.pi * r_i / n_elec < np.sqrt(pitch ** 2 + pitch ** 2):
                print "spatial resolution is too big, please change the number of sources or r"
                return
            source_xs = []
            source_ys = []
            for x in xs * r_i:
                source_xs.append(x_mesh[np.argmin(abs(x_mesh - x))])
                polarity.append(1.)
            for y in ys * r_i:
                source_ys.append(y_mesh[np.argmin(abs(y_mesh - y))])
            for x in xs * r_e:
                source_xs.append(x_mesh[np.argmin(abs(x_mesh - x))])
                polarity.append(-1.)
            for y in ys * r_e:
                source_ys.append(y_mesh[np.argmin(abs(y_mesh - y))])
            n_elec = n_elec * 2

        else:
            print("Unknown geometry for the current source arrangement")
            return

    position = [source_xs, source_ys, source_zs]
    return polarity, n_elec, position
