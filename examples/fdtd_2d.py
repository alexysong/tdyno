#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song.

## This is an example FDTD simulation. It contains structures in different shapes and materials. Wave is incident by point sources and a Total-field scattered-field (TF/SF) source. CPML boundaries to absorb the scattered waves.

## What you do: run this file. A figure window is shown, which contains a simple graphical user-interface (GUI):

* Click "play/pause" to start simulation. Click again to pause it.
* Click "step" to run simulation by one time step.
* Click "Render" to show the field in a smoothly rendered plot.
* Click "Reset" to clear everything and start over.
* The Axes plots a rectangular solving space.
  * The gray region at the edge is the CPML.
  * The large purple rectangle marks the TF/SF source boundary.
  * There are several structures in the solving space. Different colors mark different materials.
* Near the top of the figure window are controls for video recording
  * Click "Record" to start recording, click it again to stop recording and write to file.
  * Click "Pause" to pause recording. Then you can still run simulation without recording it.
  * FPS and DPI can be changed
  * You can input your choice of file name for the video.
* Simulation time-steps are displayed.

## The structure of a simulation script:
1. Initialize the solver, choose parameters such as polarization, time step length, solving mode, and total time steps.
2. Add structures to the solving space
3. Setup sources
4. Setup PML if needed
5. Setup monitors if needed
6. Run the file. The GUI is shown.
A detailed example of each part is found in this script.

## Units:
All simulations run in natural units, i.e. permittivity, permeability and light speed are 1.

"""

import numpy as np
from tdyno import TDyno

# ============   set up simulation   ==============
# initiate a simulation
tdyno = TDyno()

# geometry of the solving space
dx = 0.05
xmin = 0.
xmax = 10.
dy = 0.05
ymin = 0.
ymax = 10.

# background permittivity and permeability. They fill the space.
epsi_bg = 1.
mu_bg = 1.

# choose polarization
polarization = 'Ez'  # either 'Hz' or 'Ez'

# time step, if simulation diverges, change into a smaller number
dt = 1. / 2.1 * np.sqrt(dx ** 2 + dy ** 2)

# Total time steps for each simulation.
# After it stops, you can click "play/pause" to continue for another Nt.
Nt = 3000

# solving mode
mode = 'HDE'  # for non-magnetic (permeability is 1 throughout), use 'HDE', otherwise, use 'BHDE'

# Initialize the solver
tdyno.setup(xmin, xmax, dx, ymin, ymax, dy, epsi_bg, mu_bg, polarization=polarization, dt=dt, Nt=Nt, mode=mode)

# ============  add structures   ============
# lossless dispersionless dielectric in a rectangle (black)
xmin_d = 3.
xmax_d = 4.5
ymin_d = 3.
ymax_d = 4.5
epsi_d = 12.
mu_d = 1.
tdyno.add_structure(shape='rectangle',
                    xmin=xmin_d, xmax=xmax_d, ymin=ymin_d, ymax=ymax_d,
                    kind='dielectric',
                    epsi=epsi_d, mu=mu_d)

# lossy dispersionless dielectric in a ring (green)
# need the following material parameters
# `conductivity`, positive means loss, negative means gain
xcenter = 3.7
ycenter = 6.4
radius = 0.8
width = 0.3
epsi_b = 2.25
mu_b = 1.
conductivity = 0.5
tdyno.add_structure(shape='ring',
                    xcenter=xcenter, ycenter=ycenter, radius=radius, width=width,
                    kind='dielectric',
                    epsi=epsi_b, mu=mu_b, conductivity=conductivity)

# Lorentz dielectric in a disk  (blue)
# need the following material parameters
# `epsi`, `mu`, permittivity and permeability at high frequencies
# `res_omegas`, the resonance frequencies
# `d_epsi`, the change in permittivity at each resonance, i.e. the resonance strength
# `gammas`, broadening for each resonances
xcenter = 6.4
ycenter = 3.7
radius = 0.8
epsi_Lz = 1.5
mu_Lz = 1.
res_omegas = [2 * np.pi / wl for wl in [8., 4., 12.]]
d_epsi = [12., 11., 2.]
gammas = [2 * np.pi / wl for wl in [12., 10., 3.]]
tdyno.add_structure(shape='disk',
                    xcenter=xcenter, ycenter=ycenter, radius=radius,
                    kind='Lorentz',
                    epsi=epsi_Lz, mu=mu_Lz, res_omegas=res_omegas, d_epsi=d_epsi, gammas=gammas)

# Drude metal in a wedge (yellow)
# need the following parameters
# `epsi`, `mu`, permittivity and permeability at high frequencies
# `res_omegas`, the resonance frequencies
# `gammas`, broadening for the resonances
xcenter = 6.4
ycenter = 6.4
radius = 0.8
width = 0.3
angle1 = 0.
angle2 = 270.
epsi_b = 1.
mu_b = 1.
res_omegas = [2 * np.pi / wl for wl in [1.]]
gammas = [2 * np.pi / wl for wl in [2.]]
tdyno.add_structure(shape='wedge',
                    xcenter=xcenter, ycenter=ycenter, radius=radius, width=width, angle1=angle1, angle2=angle2,
                    kind='metal',
                    epsi=epsi_b, mu=mu_b, res_omegas=res_omegas, gammas=gammas)

# ============   Source temporal profiles  ============
# The temporal profile can be any of the following
# `cw` (continuous wave),
# `pulse` (single Gaussian pulse),
# `packet` (cw wave modulated by a Gaussian profile)

# Gaussian pulse
tau = 2.  # time constant of the pulse
pulse = tdyno.add_source_temporal('pulse', tau=tau)

# cw with single frequency
# initially the wave must ramps up by a Gaussian profile. After reaching the peak, the amplitude stays constant.
wl = 2.  # wavelength
omega = 2. * np.pi / wl  # frequency
phi = np.pi / 2.  # phase
tau = 4.  # how fast the wave ramps up.
t_before = 45.  # total time before reaching steady amplitude
cw = tdyno.add_source_temporal('cw', omega=omega, phase=phi, tau=tau, t_before=t_before)

# a wave packet, harmonic wave (carrier) modulated by a Gaussian profile
wl = 2.  # carrier wavelength
omega = 2. * np.pi / wl  # carrier frequency
t_span = 0.  # after reaching the Gaussian peak, keep amplitude at this level for t_spn amount of time
tau = 4.  # how fast the wave ramps up.
t_before = 25.  # total time before reaching peak amplitude
packet = tdyno.add_source_temporal('packet', omega=omega, t_span=t_span, tau=tau, t_before=t_before)

# ============   set up sources   ============
# After setting up the temporal profile, now set up the source
# A Point source
x_sc = 7.5  # positions
y_sc = 7.5
amp = 100.  # amplitude of the source
tdyno.add_point_source(x=x_sc, y=y_sc, amp=amp, t_profile=pulse)

# A Point source
x_sc = 2.5  # positions
y_sc = 7.5
amp = 20  # amplitude of the source
tdyno.add_point_source(x=x_sc, y=y_sc, amp=amp, t_profile=packet)

# TFSF Source
xmin_ts = 2.  # the corners of the TFSF source
xmax_ts = 8
ymin_ts = 2.
ymax_ts = 8
# kx and ky only determine the wave direction. The magnitude of the wave vector in vacuum is automatically set by the frequency in the temporal profile.
kx = 0.25
ky = 0.5
amp = 0.5  # wave amplitude
# `sides` defaults to be `'all'`. Can also be 'top', 'bottom', 'left' or 'right' for uni-directional plane wave sources.
sides = 'all'
tdyno.add_tfsf_source(xmin=xmin_ts, xmax=xmax_ts, ymin=ymin_ts, ymax=ymax_ts, kx=kx, ky=ky, amp=amp, t_profile=cw, sides=sides)

# ============   PML   ============
poly_order = 3.5  # pml polynomial order
ratio = 1.e-7  # target attenuation ratio
npx = 15  # number of PML cells in x and y direction
npy = 15
tdyno.add_pml(poly_order, ratio, npx=npx, npy=npy)

# ============   numeric dispersion correction  ============
# choose if you need numeric dispersion correction (ndc).
# If True, need to supply the frequency for ndc.
tdyno.use_ndc(True, omega)  # the frequency should be that of the source in use

# ============   Running simulation   ============
skipping = 5  # skip some frames in plotting (save your time)
tdyno.run(skipping=skipping, vmin=-1., vmax=1.)  # vmin and vmax set the data range for plotting

