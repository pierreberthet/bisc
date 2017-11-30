import LFPy as lfpy
import numpy as np
import matplotlib.pyplot as plt







def built_for_mpi_comm(cell, glb_vext, glb_vmem, v_idxs, widx, rank):
    '''
    Return a dict of array useful for plotting in parallel simulation (cell objet can not be communicated directly between thread).
    '''
    #c_idxs = [plt.cm.jet( 1.*v_idxs.index(idx) / len(v_idxs) for idx in v_idxs  )]
    #v_clr_z = lambda z: plt.cm.jet(1.0 * (z - np.min(cell.zend)) / (np.max(np.abs(cell.zmid) - np.min(np.abs(cell.zmid)))))
    return {'totnsegs':cell.totnsegs, 'glb_vext':glb_vext, 'glb_vmem':glb_vmem, 'v_idxs':v_idxs, 'name':cell.get_idx_name(v_idxs[widx])[1], 'rank':rank,\
            'xstart':cell.xstart, 'ystart':cell.ystart, 'zstart':cell.zstart,\
            'xmid':cell.xmid, 'ymid':cell.ymid, 'zmid':cell.zmid,\
            'xend':cell.xend, 'yend':cell.yend, 'zend':cell.zend }



def built_for_mpi_space(cell, rank):
    '''
    Return a dict of array useful for plotting cells in 3D space, in parallel simulation (cell objet can not be communicated directly between thread).
    '''
    return {'totnsegs':cell.totnsegs, 'rank':rank,\
            'xstart':cell.xstart, 'ystart':cell.ystart, 'zstart':cell.zstart,\
            'xmid':cell.xmid, 'ymid':cell.ymid, 'zmid':cell.zmid,\
            'xend':cell.xend, 'yend':cell.yend, 'zend':cell.zend }


def return_first_spike_time_and_idx(vmem):
    crossings = []
    if np.max(vmem) < -20:
        print "No spikes detected"
        return [None, None]
 
    for comp_idx in range(vmem.shape[0]):
        for t_idx in range(1, vmem.shape[1]):
            if vmem[comp_idx, t_idx - 1] < -20 <= vmem[comp_idx, t_idx]:
                crossings.append([t_idx, comp_idx])
    crossings = np.array(crossings)
    first_spike_comp_idx = np.argmin(crossings[:, 0])

    return crossings[first_spike_comp_idx]


def create_bisc_array():
 
    return bisc_array


def compute_dderivative():
    
    return dderivative



def plot_segment_vmem(cell, seg_name):
    
    plt.figure()
    for name in cell.allsecnames:
        print name
        plt.plot(cell.tvec, cell.vmem[cell.get_idx(name)[0]]) 

    plt.show()


    return







