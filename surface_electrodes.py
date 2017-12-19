    import numpy as np


class ImposedPotentialField:

    """Class to make the imposed external from given current sources.

    Parameters
    -----
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
            ef += self.source_amps[s_idx] / (2 * np.pi * self.sigma * np.sqrt( (self.source_xs[s_idx] - x) ** 2 + (self.source_ys[s_idx] - y) ** 2 +(self.source_zs[s_idx] - z) ** 2))
        
        return ef

































