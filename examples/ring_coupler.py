#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example contain a ring resonator with input/output coupling waveguides.
"""

import numpy as np
from tdyno import TDyno

# ============   set up simulation   ==============
# initiate a simulation
tdyno = TDyno()

# geometry of the solving space
dx = 0.05
xmin = 0.
xmax = 20.5
dy = 0.05
ymin = 0.
ymax = 11.

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
tdyno.setup(xmin, xmax, dx, ymin, ymax, dy, epsi_bg, mu_bg, polarization=polarization, dt=dt, Nt=Nt, mode=mode)  # this initializes the structure

# ============  add structures   ============
# waveguide 1
tdyno.add_structure(shape='rectangle',
                    xmin=0., xmax=1., ymin=1., ymax=1.5,
                    kind='dielectric',
                    epsi=12.)

# waveguide 2
tdyno.add_structure(shape='rectangle',
                    xmin=0., xmax=1., ymin=ymax - 1.5, ymax=ymax - 1.,
                    kind='dielectric',
                    epsi=12.)

# half circle wedge on left
tdyno.add_structure(shape='wedge',
                    xcenter=1., ycenter=5.5, radius=4.5, angle1=-90., angle2=90., width=0.5,
                    kind='dielectric',
                    epsi=12.)

# center ring
tdyno.add_structure(shape='ring',
                    xcenter=10.25, ycenter=5.5, radius=4.5, width=0.5,
                    kind='dielectric',
                    epsi=12.)

# right waveguide 1
tdyno.add_structure(shape='rectangle',
                    xmin=xmax - 1., xmax=xmax, ymin=1., ymax=1.5,
                    kind='dielectric',
                    epsi=12.)

# right waveguide 2
tdyno.add_structure(shape='rectangle',
                    xmin=xmax - 1., xmax=xmax, ymin=ymax - 1.5, ymax=ymax - 1.,
                    kind='dielectric',
                    epsi=12.)

# half circle wedge on right
tdyno.add_structure(shape='wedge',
                    xcenter=xmax - 1., ycenter=5.5, radius=4.5, width=0.5, angle1=90., angle2=270.,
                    kind='dielectric',
                    epsi=12.)

# ============   Source temporal profiles  ============
omega = 2. * np.pi * 0.257
tau = 10.
cw = tdyno.add_source_temporal('cw', omega, tau=tau)

# ============   set up sources   ============
tdyno.add_tfsf_source(xmin=1., xmax=xmax - 2.5, ymin=1., ymax=1.5, kx=1., ky=0., amp=1., t_profile=cw, sides='left', epsi=12.)

# ===============   PML   =================
tdyno.add_pml(poly_ord=3.5, ratio=1.e-7, npx=15, npy=15)

# ============   numeric dispersion correction  ============
tdyno.use_ndc(True, omega)  # the frequency should be that of the source in use

# ==============   Running simulation   =================
tdyno.run(skipping=5, vmin=-1., vmax=1.)
