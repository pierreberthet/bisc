
"""

Sample Current Spatial Patterns

"""

from MEAutility import *
import matplotlib.pyplot as plt


N_side = 10
pitch = 10
amp = 30

mea = SquareMEA(dim=N_side, pitch=pitch)

# RANDOM

mea.set_random_currents(amp=amp)

# MONOPOLAR (almost centered)

mea.reset_currents()
mea[int(round(N_side/2))][int(round(N_side/2))].set_current(amp)

plt.matshow(mea.get_current_matrix())

# BIPOLAR

mea.reset_currents()
mea[int(round(N_side/2))][int(round(N_side/2))].set_current(amp)
mea[int(round(N_side/2))-1][int(round(N_side/2))].set_current(-amp)

plt.matshow(mea.get_current_matrix())

# MULTIPOLAR
mea[int(round(N_side/2))][int(round(N_side/2))].set_current(amp)

for xx in range(3):
    for yy in range(3):
        if xx != 1 or yy != 1:
            mea[int(round(N_side/2))-1+xx][int(round(N_side/2))-1+yy].set_current(-amp/8)

mea.reset_currents()

plt.matshow(mea.get_current_matrix())

# SINC
mea.reset_currents()

x = np.linspace(-2, 2, 9)
xx = np.outer(x, x)

sinc_weights = np.sinc(xx)
plt.matshow(sinc_weights)
plt.show()