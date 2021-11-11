#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
In this example, we have a waveguide under index modulations or gain-loss modulations. Comment/uncomment the corresponding sections to see the effect of each.
"""

import numpy as np
from tdyno import TDyno


# ============   set up simulation   ==============
# initiate a simulation
tdyno = TDyno()

# geometry of the solving space
dx = 0.1
xmin = -10.
xmax = 30.
dy = 0.1
ymin = -4.
ymax = 4.

# background permittivity and permeability. They fill the space.
epsi_bg = 1.
mu_bg = 1.

# choose polarization
polarization = 'Ez'  # either 'Hz' or 'Ez'

# time step, if simulation diverges, change into a smaller number
dt = 1. / 2. * np.sqrt(dx ** 2 + dy ** 2)

# Total time steps for each simulation.
# After it stops, you can click "play/pause" to continue for another Nt.
Nt = 20000

# solving mode
mode = 'HDE'  # for non-magnetic (permeability is 1 throughout), use 'HDE', otherwise, use 'BHDE'

# Initialize the solver
tdyno.setup(xmin, xmax, dx, ymin, ymax, dy, epsi_bg, mu_bg, polarization=polarization, dt=dt, Nt=Nt, mode=mode)

# ============  add structures   ============
# waveguide
d_slab = 1.
epsi_w = 3.57**2  # silicon at 1 micron
tdyno.add_structure(shape='rectangle',
                    xmin=xmin, xmax=xmax, ymin=-0.5, ymax=0.5,
                    kind='dielectric',
                    epsi=epsi_w)


# Dynamic modulations can be defined in a rectangle shape. We have the choice of index or gain/loss modulations.
# Index modulation is done by modulating the real part of permittivity.
# Gain/loss modulation is done by modulating the conductivity of the material.
# Arbitrary modulation frequency, wave vector, amplitude and phase can be set.
# Below are two examples of dynamic modulated structures, one in the index and the other in gain/loss.
# In these examples, dynamic modulations couple an even mode and an odd mode of the waveguide.
# Comment/uncomment the corresponding sections to see effects.


# # ------------   index modulations   ------------
# # calculate omega (frequency) and q (wave vector) for modulation
# k0 = 2 * np.pi / d_slab * 0.4989  # even mode
# omega0 = 2 * np.pi / d_slab * 0.165
# k1 = 2 * np.pi / d_slab * 0.3924  # odd mode
# omega1 = 2 * np.pi / d_slab * 0.2126
# m_omega = omega1 - omega0  # modulation frequency
# m_q = ((k0 - k1), 0)  # modulation wave vector
# m_amp = 1.0  # modulation amplitude
# # modulating upper half and lower half of slab with pi phase difference.
# xmin_1l = 0.
# xmax_1l = 20.
# ymin_1l = -0.5
# ymax_1l = 0.
# m_phase_1l = np.pi / 2.  # modulation phase
# tdyno.add_structure(shape='rectangle',
#                     xmin=xmin_1l, xmax=xmax_1l, ymin=ymin_1l, ymax=ymax_1l,
#                     kind='index modulated',
#                     epsi=epsi_w,
#                     m_amp=m_amp, m_omega=m_omega, m_q=m_q, m_phase=m_phase_1l)
# xmin_1u = 0.
# xmax_1u = 20.
# ymin_1u = 0.
# ymax_1u = 0.5
# m_phase_1u = np.pi * 3. / 2.  # modulation phase
# tdyno.add_structure(shape='rectangle',
#                     xmin=xmin_1u, xmax=xmax_1u, ymin=ymin_1u, ymax=ymax_1u,
#                     kind='index modulated',
#                     epsi=epsi_w,
#                     m_amp=m_amp, m_omega=m_omega, m_q=m_q, m_phase=m_phase_1u)
# # # ------------   End of modulations   ------------


# ------------   gain and loss modulations   ------------
# calculate omega (frequency) and q (wave vector) for modulation
k0 = 2 * np.pi / d_slab * 0.4989  # even mode
omega0 = 2 * np.pi / d_slab * 0.165
k1 = 2 * np.pi / d_slab * 0.3924  # odd mode
omega1 = 2 * np.pi / d_slab * 0.2126
m_omega = omega1 - omega0  # modulation frequency
m_q = ((k0 - k1), 0)  # modulation wave vector
m_amp = 1.0  # modulation amplitude
# modulating upper half and lower half of slab with pi phase difference.
xmin_1l = 0.
xmax_1l = 20.
ymin_1l = -0.5
ymax_1l = 0.
m_phase_1l = 0.  # modulation phase
tdyno.add_structure(shape='rectangle',
                    xmin=xmin_1l, xmax=xmax_1l,ymin=ymin_1l, ymax=ymax_1l,
                    kind='gain-loss modulated',
                    epsi=epsi_w,
                    m_amp=m_amp, m_omega=m_omega, m_q=m_q, m_phase=m_phase_1l)
xmin_1u = xmin_1l
xmax_1u = xmax_1l
ymin_1u = 0.
ymax_1u = 0.5
m_phase_1u = np.pi  # modulation phase
tdyno.add_structure(shape='rectangle',
                    xmin=xmin_1u, xmax=xmax_1u,ymin=ymin_1u, ymax=ymax_1u,
                    kind='gain-loss modulated',
                    epsi=epsi_w,
                    m_amp=m_amp, m_omega=m_omega, m_q=m_q, m_phase=m_phase_1u)
# ------------   End of gain and loss modulations   ------------


# ============   Source temporal profiles  ============
omega = 2. * np.pi / d_slab * 0.165
tau = 20.
cw = tdyno.add_source_temporal('cw', omega, tau=tau)

tau = 125
packet = tdyno.add_source_temporal('packet', omega, tau=tau)

# ============   set up sources   ============
# TFSF source
# uncomment next line for left input in even mode
tdyno.add_tfsf_source(xmin=xmin+2.5, xmax=xmax-2.5, ymin=-0.5, ymax=0.5, kx=1., ky=0., amp=1., t_profile=cw, epsi=epsi_w, sides='left')
# uncomment next line for right input in even mode
# tdyno.add_tfsf_source(xmin=xmin+2.5, xmax=xmax-2.5, ymin=-0.5, ymax=0.5, kx=-1., ky=0., amp=1., t_profile=cw, epsi=epsi_w, sides='right')

# ============   PML   ============
tdyno.add_pml(poly_ord=3.5, ratio=1.e-7, npx=15, npy=15)

# ============   monitors   ============
# the point monitors record the field amplitudes at certain locations, then do a weighted sum of the fields.
# For 'Ez' polarization, they records the Ez component, vice versa for 'Hz'.
# After running simulation, can click "save" button in each monitor window to save the respective data.

# point monitor on the left
x_mnt = xmin + 6.
y_mnt = 0.2
tdyno.add_point_monitors(coords=[(x_mnt, y_mnt)], weights=[1.], omega_min=2*np.pi*0.15, omega_max=2*np.pi*0.225, n_omega=200)
# omega_min, omega_max, and n_omega are the frequency range and resolution for the power spectral density of the recorded signal.

# point monitors on the right, looking at even mode only
x_1 = xmax - 6.
y_1 = 0.2
x_2 = x_1
y_2 = -0.3
coords = [(x_1, y_1), (x_2, y_2)]  # positions of the points
wts = [1., 1.]  # do a weighted sum of the recorded fields. By adding field values at symmetric locations, we get the even mode.
tdyno.add_point_monitors(coords=coords, weights=wts, omega_min=2*np.pi*0.15, omega_max=2*np.pi*0.225, n_omega=200)

# point monitors on the right, looking at odd mode only
wts = [1., -1.]  # By subtracting the field values at symmetric locations, we get the odd mode.
tdyno.add_point_monitors(coords=coords, weights=wts, omega_min=2*np.pi*0.15, omega_max=2*np.pi*0.225, n_omega=200)

# ==============   Running simulation   =================
tdyno.run(skipping=5, vmin=-1., vmax=1.)


