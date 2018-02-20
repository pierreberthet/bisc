import LFPy as lfpy
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
# import neuron


'''
Conventions:
position = [um]
current = [nA]
voltage = [mV]
'''


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


def built_for_mpi_space(cell, rank, extra1=None, extra2=None):
    '''
    Return a dict of array useful for plotting cells in 3D space, in parallel simulation
    (cell objet can not be communicated directly between thread).
    '''
    return {'totnsegs': cell.totnsegs, 'rank': rank, 'vmem': cell.vmem, 'vext': cell.v_ext, 'extra1': extra1,
            'xstart': cell.xstart, 'ystart': cell.ystart, 'zstart': cell.zstart, 'extra2': extra2,
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
            return [t_idx, np.argmax(vmem.T[t_idx])]


def spike_soma(cell):
    ''' Return the index of the compartment of the soma where Vmem first crossed the threshold (Usually set at -20 mV.
    If many compartments crossed threshold during a unique time step, it returns the index of the compartment
    with the most depolarized membrane value. Also contains the time step when this occurred.'''
    idx = cell.get_idx('soma')
    if np.max(cell.vmem[idx]) < -20:
        print "No spikes detected"
        return [None, None]
    else:
        if idx.size == 1:
            t_ap = np.where(cell.vmem[idx[0]] > -20)[0][0]
            return [t_ap, idx[0]]
        else:
            t_ap = np.min(np.where(cell.vmem[idx] > -20)[1])
            return [t_ap, idx[np.argmax(cell.vmem[idx].T[t_ap])]]


def spike_segments(cell):
    ''' Return the index for all segments where Vmem crossed the threshold (Usually set at -20 mV.)
    If many compartments within a segment crossed threshold during a unique time step,
    it returns the index of the compartment with the most depolarized membrane value.
    Also contains the time step when this occurred.'''
    spike_time_loc = {}
    for seg in cell.allsecnames:
        idx = cell.get_idx(seg)
        if np.max(cell.vmem[idx]) < -20:
            spike_time_loc[seg] = [None, None]
        else:
            if idx.size == 1:
                t_ap = np.where(cell.vmem[idx[0]] > -20)[0][0]
                spike_time_loc[seg] = [t_ap, idx[0]]
            else:
                t_ap = np.min(np.where(cell.vmem[idx] > -20)[1])
                spike_time_loc[seg] = [t_ap, idx[np.argmax(cell.vmem[idx].T[t_ap])]]
    return spike_time_loc


def spike_compartments(cell):
    spike_time_comp = np.zeros(cell.totnsegs)
    for idx in range(cell.totnsegs):
        if np.max(cell.vmem[idx]) < -20:
            # print "No spikes detected"
            spike_time_comp[idx] = None
        else:
            spike_time_comp[idx] = np.where(cell.vmem[idx] > -20.)[0][0]
    return spike_time_comp


def ap_dromic(cell):
    '''If Vmem crossed a predefined threshold (here -20 mV), returns True if the AP is orthodromic,
    False if antidromic, and None if the threshold was not crossed'''
    ais = cell.get_idx('axon[0]')[0]
    axon = np.where(cell.vmem[ais] > -20)[0]
    soma = np.where(cell.vmem[0] > -20)[0]
    if axon.size == 0 or soma.size == 0:
        return None
    if np.argmin(soma) < np.argmin(axon):
        print("ORTHODROMIC AP soma {}, axon {}").format(soma[0], axon[0])
        return True
    else:
        print("ANTIDROMIC AP soma {}, axon {}").format(soma[0], axon[0])
        return False


def dendritic_spike(cell):
    '''NOT SURE IF THIS IS VALID'''
    '''If Vmem crossed a predefined threshold (here -20 mV), returns True if the AP was triggered by dendritic activation,
    False otherwise, and None if the threshold was not crossed'''
    assert (cell.get_idx('dend').any() or cell.get_idx('apic').any()), "NO DENDRITES (apical OR basal)"
    soma = np.where(cell.vmem[0] > -20)[0]
    t_dend = None
    t_apic = None
    if soma.size == 0:
        return None
    dendritic = False
    if cell.get_idx('dend').any():
        # print("no apical dendrites found!!! (no 'apic'?)")
        idx = cell.get_idx('dend[0]')[0]
        if idx.size == 1:
            dend = np.where(cell.vmem[cell.get_idx('dend[0]')[0]] > - 20)[0]
            if soma[0] > dend[0]:
                dendritic = True
                t_dend = dend[0]
        else:
            dend = np.min(np.where(cell.vmem[cell.get_idx('dend[0]')[0]] > - 20))[1]
            if soma[0] > dend:
                dendritic = True
                t_dend = dend

    if cell.get_idx('apic').any():
        # print("no apical dendrites found!!! (no 'apic'?)")
        idx = cell.get_idx('apic[0]')[0]
        if idx.size == 1:
            apic = np.where(cell.vmem[cell.get_idx('apic[0]')[0]] > - 20)[0]
            if soma[0] > apic[0]:
                dendritic = True
                t_apic = apic[0]
        else:
            apic = np.min(np.where(cell.vmem[cell.get_idx('apic[0]')[0]] > - 20))[1]
            if soma[0] > apic:
                dendritic = True
                t_apic = apic

    if dendritic:
        print("DENDRITIC activation in cell {}, soma {}, dend {} apic {}").format(cell.allsecnames[0][:cell.allsecnames[0].find('.')], soma[0], t_dend, t_apic)
    else:
        print("NON DENDRITIC activation in cell {}, soma {}, dend {} apic {}").format(cell.allsecnames[0][:cell.allsecnames[0].find('.')],soma[0], t_dend, t_apic)

    return dendritic


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


def reposition_cell_flip(cell, x=0, y=0, z=0):
    '''
    Flip upside down the cell model, thus keeping it perpendicular to the x axis,
    and reposition the cell to avoid any displacement on the z-axis.
    (zmin and zend should be unchanged)
    '''
    # l_cell = np.max(cell.zend) + np.abs(np.min(cell.zend))
    cell.set_rotation(y=np.pi)
    cell.set_pos(x=x, y=y, z=cell.zend[0] + np.abs(np.max(cell.zend)))
    return


def create_bisc_array():
    '''
    to be edited from create_array_shape()
    '''
    bisc_array = []
    return bisc_array


def external_field(ExtPot, space_resolution=500, x_extent=500, y_extent=500, z_extent=500,
                   z_top=0, axis='xz', dderivative=False, plan=None):
    '''
    Create an external field and optionally compute its double derivative, along a specified plane (2D).
    '''
    if plan is None:
        plan = 0

    if axis == 'xz':
        v_field_ext = np.zeros((space_resolution, space_resolution))
        xf = np.linspace(-x_extent, x_extent, space_resolution)
        zf = np.linspace(-z_extent, z_top, space_resolution)
        for xidx, x in enumerate(xf):
            for zidx, z in enumerate(zf):
                v_field_ext[xidx, zidx] = ExtPot.ext_field(x, plan, z)

    if axis == 'xy':
        v_field_ext = np.zeros((space_resolution, space_resolution))
        xf = np.linspace(-x_extent, x_extent, space_resolution)
        yf = np.linspace(-y_extent, y_extent, space_resolution)
        for xidx, x in enumerate(xf):
            for yidx, y in enumerate(yf):
                v_field_ext[xidx, yidx] = ExtPot.ext_field(x, y, plan)

    if axis == 'yz':
        v_field_ext = np.zeros((space_resolution, space_resolution))
        yf = np.linspace(-y_extent, y_extent, space_resolution)
        zf = np.linspace(-z_extent, z_top, space_resolution)
        for yidx, y in enumerate(yf):
            for zidx, z in enumerate(zf):
                v_field_ext[yidx, zidx] = ExtPot.ext_field(plan, y, z)

    if dderivative:
        '''
        Compute the double derivative of the electric field
        '''

        # derivative of V in z-direction
        if axis == 'xz':
            d_v_field_ext = np.zeros((v_field_ext.shape[0], v_field_ext.shape[1] - 1))
            dz = zf[1] - zf[0]

            for zidx in range(len(zf) - 1):
                d_v_field_ext[:, zidx] = (v_field_ext[:, zidx + 1] - v_field_ext[:, zidx]) / dz

            # double derivative of V in z-direction
            dd_v_field_ext = np.zeros((v_field_ext.shape[0], d_v_field_ext.shape[1] - 1))

            for zidx in range(len(zf) - 2):
                dd_v_field_ext[:, zidx] = (d_v_field_ext[:, zidx + 1] - d_v_field_ext[:, zidx]) / dz

        elif axis == 'xy':
            d_v_field_ext = np.zeros((v_field_ext.shape[0], v_field_ext.shape[1] - 1))
            dy = yf[1] - yf[0]

            for yidx in range(len(yf) - 1):
                d_v_field_ext[:, yidx] = (v_field_ext[:, yidx + 1] - v_field_ext[:, yidx]) / dy

            # double derivative of V in z-direction
            dd_v_field_ext = np.zeros((v_field_ext.shape[0], d_v_field_ext.shape[1] - 1))

            for yidx in range(len(yf) - 2):
                dd_v_field_ext[:, yidx] = (d_v_field_ext[:, yidx + 1] - d_v_field_ext[:, yidx]) / dy

        elif axis == 'yz':
            d_v_field_ext = np.zeros((v_field_ext.shape[0], v_field_ext.shape[1] - 1))
            dz = zf[1] - zf[0]

            for zidx in range(len(zf) - 1):
                d_v_field_ext[:, zidx] = (v_field_ext[:, zidx + 1] - v_field_ext[:, zidx]) / dz

            # double derivative of V in z-direction
            dd_v_field_ext = np.zeros((v_field_ext.shape[0], d_v_field_ext.shape[1] - 1))

            for zidx in range(len(zf) - 2):
                dd_v_field_ext[:, zidx] = (d_v_field_ext[:, zidx + 1] - d_v_field_ext[:, zidx]) / dz

        return v_field_ext, d_v_field_ext, dd_v_field_ext
    else:
        return v_field_ext


def plot_segment_vmem(cell, seg_name):

    plt.figure()
    for name in cell.allsecnames:
        print name
        plt.plot(cell.tvec, cell.vmem[cell.get_idx(name)[0]])

    plt.show()

    return


def get_minmax(multid_array):
    ma = multid_array[0][0]
    mi = ma
    if ma is None:
        print("! None values in the array !")

    ma = np.max(multid_array)
    mi = np.min(multid_array)

    # for i in range():
    #     if np.min() < mi:
    #         mi = np.min()

    #     if np.max() > ma:
    #         ma = np.max()

    return mi, ma


def init_neurons_epfl(layer, n_cells):
    '''Return a list of neurons from the EPFL models '''
    # working dir
    neurons = glob(os.path.join('morphologies/hoc_combos_syn.1_0_10.allzips', layer + '*'))
    assert len(neurons) > n_cells, "More threads than available neuron models"
    print("Found {} {} neuron models. Keeping {}.").format(len(neurons), layer, n_cells)
    neurons = neurons[:n_cells]

    # Write function to sample from the total pool of neurons and model types with one layer

    # for n in range(2, n_cells + 1):
    #     # load only some layer 5 pyramidal cell types
    #     neurons += glob(os.path.join('morphologies/hoc_combos_syn.1_0_10.allzips', 'L5_MC*'))[:1]
    #     neurons += glob(os.path.join('morphologies/hoc_combos_syn.1_0_10.allzips', 'L5_LBC*'))[:1]
    #     neurons += glob(os.path.join('morphologies/hoc_combos_syn.1_0_10.allzips', 'L5_NBC*'))[:1]

    return neurons


def get_epfl_model_name(list_models, short=True):
    name = []
    for i, model in enumerate(list_models):
        name.append(model[(model.rfind('/') + 1):])
        if short:
            name[i] = name[i][:(name[i][name[i].find('_') + 1:].find('_') + name[i].find('_') + 1)]

    return name


def set_z_layer(layer_name):
    '''Returns a random depth from a normal distribution centered around the median depth of the specified layer.
    Values from DeFelipe et al. 2002. STD set to +/- 1/14 of the layer thickness.'''
    if layer_name == 'L1':
        return np.random.normal(-116, 16)
    if layer_name == 'L23':
        return np.random.normal(-770, 75)
    if layer_name == 'L4':
        return np.random.normal(-1445, 20)
    if layer_name == 'L5':
        return np.random.normal(-1866, 39)
    if layer_name == 'L6':
        return np.random.normal(-2382, 34)
    else:
        print("lAYER NAME NOT RECOGNIZED. Possible options are 'L1', 'L23', 'L4', 'L5', 'L6'.")
        return None


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

    def ext_field_v(self, x, y, z):
        """Returns the external field at positions x, y, z"""
        ef = 0
        for s_idx in range(self.num_sources):
            ef += self.source_amps[s_idx] / (4 * np.pi * np.sqrt(
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


def test_linear(axis='xz', dim=[-200, 200, 0, -1000], a=1e-3, b=.1, res=201):
    '''
    Returns an array with a linear field with the specified dimensions, starting at 0.
    '''
    v_field_ext = np.zeros((abs(dim[0] - dim[1]), abs(dim[2] - dim[3])))
    if axis == 'xz':
        xf = np.linspace(dim[0], dim[1], abs(dim[0] - dim[1]))
        zf = np.linspace(dim[2], dim[3], abs(dim[2] - dim[3]))
        for zidx, z in enumerate(zf):
            for xidx, x in enumerate(xf):
                v_field_ext[xidx, zidx] = a * z + b
    if axis == 'yz':
        yf = np.linspace(dim[0], dim[1], abs(dim[0] - dim[1]))
        zf = np.linspace(dim[2], dim[3], abs(dim[2] - dim[3]))
        for zidx, z in enumerate(zf):
            for yidx, y in enumerate(yf):
                v_field_ext[yidx, zidx] = a * z + b
    if axis == 'xy':
        xf = np.linspace(dim[0], dim[1], abs(dim[0] - dim[1]))
        yf = np.linspace(dim[2], dim[3], abs(dim[2] - dim[3]))
        for yidx, y in enumerate(yf):
            for xidx, x in enumerate(xf):
                v_field_ext[xidx, yidx] = a * y + b

    # toplot = np.array((np.linspace(dim[0], dim[1], res), np.linspace(dim[2], dim[3], res), np.zeros(res)))
    # for x in res:
    #     for y in res:
    #         toplot
    return v_field_ext


def half_test_linear(axis='xz', dim=[-200, 200, 0, -1000], a=1e-3, b=.1, res=201):
    '''
    TEST Returns an array with a linear field with the specified dimensions, starting at 0.
    '''
    v_field_ext = np.zeros((abs(dim[0] - dim[1]), abs(dim[2] - dim[3])))
    if axis == 'xz':
        xf = np.linspace(dim[0], dim[1], abs(dim[0] - dim[1]))
        zf = np.linspace(dim[2], dim[3], abs(dim[2] - dim[3]))
        for zidx, z in enumerate(zf):
            if z > dim[3] / 2.:
                for xidx, x in enumerate(xf):
                    v_field_ext[xidx, zidx] = a * z + b
            else:
                for xidx, x in enumerate(xf):
                    v_field_ext[xidx, zidx] = a * (dim[3] - z) + b
    if axis == 'yz':
        yf = np.linspace(dim[0], dim[1], abs(dim[0] - dim[1]))
        zf = np.linspace(dim[2], dim[3], abs(dim[2] - dim[3]))
        for zidx, z in enumerate(zf):
            for yidx, y in enumerate(yf):
                v_field_ext[yidx, zidx] = a * z + b
    if axis == 'xy':
        xf = np.linspace(dim[0], dim[1], abs(dim[0] - dim[1]))
        yf = np.linspace(dim[2], dim[3], abs(dim[2] - dim[3]))
        for yidx, y in enumerate(yf):
            for xidx, x in enumerate(xf):
                v_field_ext[xidx, yidx] = a * y + b

    # toplot = np.array((np.linspace(dim[0], dim[1], res), np.linspace(dim[2], dim[3], res), np.zeros(res)))
    # for x in res:
    #     for y in res:
    #         toplot
    return v_field_ext


def half_linear_field(cell, pulse_start, pulse_end, n_tsteps, axis, a=1e-3, b=.1):
    '''
    0 mV @ x, y or z = 0, depending on the axis selected for the field.
    '''
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    if axis == 'x':
        v_cell_ext[:, pulse_start:pulse_end] = np.array([(a * x + b) for x in cell.xmid]).reshape(cell.totnsegs, 1)
    elif axis == 'y':
        v_cell_ext[:, pulse_start:pulse_end] = np.array([(a * y + b) for y in cell.ymid]).reshape(cell.totnsegs, 1)
    elif axis == 'z':
        for i, z in enumerate(cell.zmid):
            if z > np.min(cell.zmid) / 2.:
                v_cell_ext[i, pulse_start:pulse_end] = a * z + b
            else:
                v_cell_ext[i, pulse_start:pulse_end] = a * (np.min(cell.zmid) - z) + b
        # v_cell_ext[:, pulse_start:pulse_end] = np.array([(a * z + b) for z in cell.zmid]).reshape(cell.totnsegs, 1)

    return v_cell_ext


def linear_field(cell, pulse_start, pulse_end, n_tsteps, axis, a=1e-3, b=.1):
    '''
    0 mV @ x, y or z = 0, depending on the axis selected for the field.
    '''
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    if axis == 'x':
        v_cell_ext[:, pulse_start:pulse_end] = np.array([(a * x + b) for x in cell.xmid]).reshape(cell.totnsegs, 1)
    elif axis == 'y':
        v_cell_ext[:, pulse_start:pulse_end] = np.array([(a * y + b) for y in cell.ymid]).reshape(cell.totnsegs, 1)
    elif axis == 'z':
        v_cell_ext[:, pulse_start:pulse_end] = np.array([(a * z + b) for z in cell.zmid]).reshape(cell.totnsegs, 1)

    return v_cell_ext


def clamp_ends(cell, pulse_start, pulse_end, voltage=-70., axis='z'):
    '''
    Clamp the extremities of a cell
    Should only be used for ball and stick models, otherwise to update
    Pay attention to the orientation, here the cell is assumed to be
    horizontally oriented along the corresponding x/y/z axis.
    '''
    pointprocesses = {'pptype': 'SEClamp', 'amp1': voltage, 'dur1': pulse_end}
    if axis == 'x':
        lfpy.StimIntElectrode(cell, np.argmax(cell.xend), **pointprocesses)
        lfpy.StimIntElectrode(cell, np.argmin(cell.xend), **pointprocesses)
    elif axis == 'y':
        lfpy.StimIntElectrode(cell, np.argmax(cell.yend), **pointprocesses)
        lfpy.StimIntElectrode(cell, np.argmin(cell.yend), **pointprocesses)
    elif axis == 'z':
        lfpy.StimIntElectrode(cell, np.argmax(cell.zend), **pointprocesses)
        lfpy.StimIntElectrode(cell, np.argmin(cell.zend), **pointprocesses)
    return


def create_array_shape(shape=None, pitch=None, names=False):
    '''
    create current sources of various shapes.
    {To implement: specifiy number as argument}
    '''
    if not pitch:
        pitch = 25

    if shape:  # create monopole
        if shape == 'dipole':
            n_elec = 2
            polarity = np.array([-1, 1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-10, 10])
            source_ys = np.array([0, 0])

        elif shape == '3dots':
            n_elec = 3
            polarity = np.array([-1 / 2., 1, -1 / 2.])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, 0, pitch])
            source_ys = np.array([0, 0, 0])

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

        elif shape == 'multipole2':
            n_elec = 8
            polarity = np.array([-1, -1, 1, 1, 1, 1, -1, -1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-100, -100, -10, -10, 10, 10, 100, 100])
            source_ys = np.array([-100, 100, -10, 10, 10, -10, -100, 100])

        elif shape == 'multipole3':
            n_elec = 16
            polarity = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-200, -200, 0, -200, -50, -50, -10, -10, 10, 10, 50, 50, 0, 200, 200, 200])
            source_ys = np.array([-200, 0, -200, 200, -50, 50, -10, 10, 10, -10, 50, -50, 200, 0, -200, 200])

        elif shape == 'quadrupole':
            n_elec = 5
            polarity = np.array([-1 / 4., -1 / 4., 1, -1 / 4., -1 / 4.])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, 0, 0, 0, pitch])
            source_ys = np.array([0, pitch, 0, -pitch, 0])

        elif shape == 'monopole':
            n_elec = 1
            polarity = np.array([1])
            source_xs = np.array([0])
            source_ys = np.array([0])
            source_zs = np.array([0])

        elif shape == 'plausible_monopole':  # #################### To complete
            n_elec = 9
            n_p = 5  # number of pitch times the return electrodes are away from the source
            pos = 1.
            neg = -pos / (n_elec - 1.)
            polarity = np.array([neg, neg, neg, neg, pos, neg, neg, neg, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-n_p * pitch, -n_p * pitch, -n_p * pitch,
                                  0, 0, 0, n_p * pitch, n_p * pitch, n_p * pitch])
            source_ys = np.array([n_p * pitch, 0, -n_p * pitch, n_p * pitch,
                                  0, -n_p * pitch, n_p * pitch, 0, -n_p * pitch])

        elif shape == 'twosquare':
            n_elec = 8
            polarity = np.array([-1, -1, 1, 1, 1, 1, -1, -1])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-(pitch * 2), -(pitch * 2), -pitch, -pitch, pitch, pitch, (pitch * 2), (pitch * 2)])
            source_ys = np.array([-(pitch * 2), (pitch * 2), -pitch, pitch, pitch, -pitch, -(pitch * 2), (pitch * 2)])

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
            pos = 1.
            neg = -5. / 4
            polarity = np.array([neg, pos, neg, pos, pos, pos, neg, pos, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, -pitch, -pitch, 0, 0, 0, pitch, pitch, pitch])
            source_ys = np.array([pitch, 0, -pitch, pitch, 0, -pitch, pitch, 0, -pitch])

        elif shape == 'bcross':
            n_elec = 5
            pos = 1.
            neg = -3. / 2
            polarity = np.array([neg, pos, pos, pos, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, 0, 0, 0, pitch])
            source_ys = np.array([0, -pitch, 0, pitch, 0])

        elif shape == 'minicross':
            n_elec = 4
            pos = 1.
            neg = -pos
            polarity = np.array([neg, pos, pos, neg])
            source_zs = np.zeros(n_elec)
            # source_xs = np.array([-2 * pitch, 0, 0, pitch * 2])
            source_xs = np.array([-2 * pitch, 0, 0, pitch * 2])
            source_ys = np.array([0, -pitch, pitch, 0])

        elif shape == 'stick':
            n_elec = 8
            pos = 1.
            neg = -pos
            polarity = np.array([neg, neg, pos, pos, pos, pos, neg, neg])
            source_zs = np.zeros(n_elec)
            source_xs = np.array([-pitch, 0, 0, 0, 0, 0, 0, pitch])
            source_ys = np.array([0, -3 * pitch, -2 * pitch, -pitch, pitch, 2 * pitch, 3 * pitch, 0])

        elif shape == 'circle':
            n_elec = 12
            polarity = []  # np.ones(n_elec)
            source_zs = np.zeros(n_elec * 2)    # double to account for positive and negative
            size_bisc = 1000
            xs = np.cos(np.linspace(-np.pi, np.pi, n_elec))
            ys = np.sin(np.linspace(-np.pi, np.pi, n_elec))
            x_mesh = range(-size_bisc / 2, (size_bisc + pitch) / 2, pitch)
            y_mesh = range(-size_bisc / 2, (size_bisc + pitch) / 2, pitch)

            r_i = 75.  # um, radius of the inner circle
            r_e = r_i + 2 * pitch  # um, radius of the external circle
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
            # source_xs = []
            # source_ys = []
            # for x in xs:
            #     source_xs.append(np.multiply(x, r_i))
            #     polarity.append(1.)
            # for y in ys:
            #     source_ys.append(np.multiply(y, r_i))
            # for x in xs:
            #     source_xs.append(np.multiply(x, r_e))
            #     polarity.append(-1.)
            # for y in ys:
            #     source_ys.append(np.multiply(y, r_e))

            n_elec = n_elec * 2
            # need to convert lists to arrays
            source_xs = np.asarray(source_xs)
            source_ys = np.asarray(source_ys)
            source_zs = np.asarray(source_zs)

        else:
            print("Unknown geometry for the current source arrangement")
            return

    position = [source_xs, source_ys, source_zs]
    return polarity, n_elec, position


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within
    Arguments
    ---------
    f : file, mode 'r'
    Returns
    -------
    templatename : str
    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print 'template {} found!'.format(templatename)
            continue

    return templatename


def get_sections_number(cell):
    '''
    Returns the number of different sections and their names.
    '''
    nlist = []
    for i, name in enumerate(cell.allsecnames):
        if name != 'soma':
            name = name[:name.rfind('[')]

        if name not in nlist:
            nlist.append(name)
    n = len(nlist)
    return n, nlist
