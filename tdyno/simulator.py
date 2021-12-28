# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from warnings import warn
from tdyno.cnst import Cnst
from tdyno.s2t import S2T
from tdyno.pct import PCT
from tdyno.sc_tp_t import Hm, HP, Gsn
from tdyno.sc2t import PS2, TSS2
from tdyno.sc2t_md import TSSMD2
from tdyno.rt2 import RT2
from tdyno.mnt.mnt_t_fa import MntMltPntAmp
from tdyno.mnt.mnt_t_poyn import Mnt2DSqPoynU
from tdyno.mnt.mnt_t_md import MntWgMdU


# todo: API use frequency instead of omega. (however, k contains 2pi anyway?)

class TDyno:
    def __init__(self):
        """
        Finite-difference time-domain (FDTD) package with dynamic modulations in the refractive index and/or in gain/loss.

        """
        self.c = Cnst()
        self.st: S2T = None
        self.scs = []
        self.pc = None
        self.mnts = []
        self.rt = None  # RT2
        self.polarization = 'Ez'
        self.dt = None
        self.Nt = None
        self.tps = []
        self.if_ndc = False
        self.omg_ndc = None
        self.md = 'HDE'

    def setup(self, xmin, xmax, dx, ymin, ymax, dy, epsi, mu, x=None, y=None, polarization='Ez', dt=None, Nt=5000, mode='BHDE', if_ndc=False, ndc_omega=1.):
        """
        Set up the solving space.

        The space is assumed to be in the x-y plane. z is out-of-plane.

        Parameters
        ----------
        xmin, xmax, dx      :   float
                                x limits and grid size
        ymin, ymax, dy      :   float
                                y limits and grid size
        epsi, mu            :   float
                                background permittivity and permeability
        x, y                :   ndarray
                                1d ndarray, If supplied, override xmin etc.
        dt                  :   float
                                time step in simulation. If simulation diverges, change to a smaller number.
        polarization        :   str
                                ['Ez'|'Hz'], 'Ez' means Ez, Hx, Hy mode. 'Hz' means Hz, Ex, Ey mode.
        Nt                  :   int
                                Total run time. After it stops, you can click "Play/Pause" again to run another Nt time steps.
        mode                :   str
                                ['BHDE'|'HDE']
        if_ndc              :   bool
                                if using numeric dispersion correction
        ndc_omega           :   float
                                frequency for ndc

        Returns
        -------

        """

        if not dt:
            dt = 1. / 2. * np.sqrt(dx ** 2 + dy ** 2)

        self.st: S2T = S2T(xmin, xmax, dx, ymin, ymax, dy, epsi, mu, x=x, y=y)
        self.set_polarization(polarization)
        self.set_dt(dt)
        self.set_Nt(Nt)
        self.solving_mode(mode)
        self.use_ndc(if_ndc, ndc_omega)

    def add_structure(self, shape='rectangle',
                      xmin=None, xmax=None, ymin=None, ymax=None,
                      xcenter=None, ycenter=None, radius=None, angle1=None, angle2=None, width=None,
                      kind='dielectric',
                      epsi=None, mu=None, conductivity=0.,
                      res_omegas=None, d_epsi=None, gammas=None,
                      m_amp=None, m_omega=None, m_q=None, m_phase=0.):

        """
        Add structures to the solving space.

        `shape` can be one of the following:
        'rectangle'|'circular'|'ring'|'wedge'
        For 'rectangle',    specify `xmin`, `xmax`, `ymin`, `ymax`.
        For 'disk',         specify centers `xcenter`, `ycenter`, and `radius`.
        For 'ring',         specify `xcenter`, `ycenter`, outer `radius`, and `width`.
        For 'wedge',        specify `xcenter`, `ycenter`, outer `radius`, `width`, and `angle1` and `angle2`.

        The type of material for the structure is specified in `kind`, which can be one of the following:

        'dielectric'|'Lorentz'|'metal'|'index modulated'|'gain-loss modulated'|‘Lorentz modulated'

        'dielectric' means dispersionless dielectric. It needs the following material parameters:
            `epsi`, `mu`, `conductivity` (optional).

        'Lorentz' means dielectric with Lorentz model. It needs the following material parameters:
            `epsi`, `mu`, `res_omegas`, `d_epsi`, `gammas`.

        'metal' means metallic material with Drude model. It needs the following material parameters:
            `epsi`, `mu`, `res_omegas`, `gammas`.

        'index modulated' means dynamic modulation in the refractive index. It needs the following material parameters:
            `epsi`, `mu`, `m_amp`, `m_omega`, `m_q`, `m_phase`.

        'loss-gain modulated' means dynamic modulation in the loss and gain. It needs the following material parameters:
            `epsi`, `mu`, 'conductivity' (optional), `m_amp`, `m_omega`, `m_q`, `m_phase`.

        'Lorentz modulated' means Lorentz model with modulated resonance strengh. It needs the following material parameters:
            `epsi`, `mu`, `res_omegas`, `d_epsi`, `gammas`, `m_amp`, `m_omega`, `m_q`, `m_phase`.

        Parameters
        ----------
        shape                   :   str
                                    ['rectangle'|'disk'|'ring'|'wedge'】
        xmin, xmax, ymin, ymax  :   float
                                    four corners of the structure. If shape is rectangular, these four parameters should be supplied.
        xcenter, ycenter        :   float
                                    x and y center of structure, for use in circular and ring
        radius                  :   float
        angle1, angle2          :   float
                                    start and end angle of wedge, in degrees, between 0 to 360
        width                   :   float
                                    width of ring or wedge
        kind                    :   str
                                    ['dielectric'|'Lorentz'|'metal'|'index modulated'|'loss-gain modulated'|‘Lorentz modulated']
        epsi, mu                :   float or Tuple[float, float, float] or 3x3 ndarray
                                    The permittivity and permeability.
                                    For Lorentz dielectric and Metal material, these two are the values at the high-frequency limit.
        res_omegas              :   list[float]
                                    resonance frequencies
        d_epsi                  :   list[float]
                                    permittivity change at each resonance, i.e. oscillator strength
        Gammas                  :   list[float]
                                    Broadening of each resonance
        conductivity            :   float
                                    loss (positive number) or gain (negative) in the material
        m_amp                   :   float or list[float]
                                    modulation amplitude
        m_omega                 :   float or list[float]
                                    modulation frequency
        m_q                     :   Tuple[float, float] or list[Tuple[float, float]]
                                    (qx, qy), wave vector of modulation.
        m_phase                 :   float or list[float]
                                    modulation phase. By default, cos modulation.

        Returns
        -------

        """
        if shape == 'rectangle':
            shape = 'rct'
        elif shape == 'disk':
            shape = 'ccl'
        elif shape == 'ring':
            shape = 'rng'
        elif shape == 'wedge':
            shape = 'wdg'
        else:
            warn('The shape of the added structure is not recognized', UserWarning)

        if kind == 'dielectric':
            if not conductivity:
                kind = 'sp'
            else:
                kind = 'lg'
        elif kind == 'Lorentz':
            kind = 'Lz'
        elif kind == 'metal':
            kind = 'Dr'
        elif kind == 'index modulated':
            kind = 'dmri'
        elif kind == 'gain-loss modulated':
            kind = 'dmlg'
        elif kind == 'Lorentz modulated':
            kind = 'dmLz'
        else:
            warn('Material type not recognized.', UserWarning)

        self.st.a_b(shp=shape,
                    xmin_b=xmin, xmax_b=xmax, ymin_b=ymin, ymax_b=ymax,
                    xc_b=xcenter, yc_b=ycenter,
                    r_b=radius,
                    a_1=angle1, a_2=angle2,
                    w_b=width,
                    knd=kind,
                    epsi_b=epsi, mu_b=mu,
                    epsi_infty_b=epsi, mu_infty_b=mu,
                    omgs_rsn=res_omegas, dlts_epsi=d_epsi, Gms=gammas,
                    sgm=conductivity,
                    m_a=m_amp, m_o=m_omega, m_q=m_q, m_p=m_phase)

    def set_polarization(self, polarization):
        """
        Choose polarization of simulation.

        Parameters
        ----------
        polarization        :   str
                                ['Ez'|'Hz']
                                'Ez' means Ez, Hx, Hy mode. 'Hz' means Hz, Ex, Ey mode.

        Returns
        -------

        """

        if polarization == 'Ez' or 'Hz':
            self.polarization = polarization
        else:
            warn("Polarization must be either 'Ez' or 'Hz'.", UserWarning)

    def set_dt(self, dt):
        """

        Parameters
        ----------
        dt      :   float
                    time step in simulation. If simulation diverges, change to a smaller number.

        Returns
        -------

        """

        self.dt = dt

    def set_Nt(self, Nt):
        """
        set total run time

        Parameters
        ----------
        Nt      :   int
                    Total run time. After it stops, you can click "Play/Pause" again to run another Nt time steps.
        Returns
        -------

        """

        self.Nt = Nt

    def add_source_temporal(self, kind, omega=1., tau=1., phase=0., t_before=None, t_span=0.):
        """
        set up the temporal profile for sources.

        For any source, its temporal profile can be any of the following
        `cw` (continuous wave),
            Need to specify `omega`, `phase` (optional), `tau` (optional),  `t_before` (optional)
            There will be a ramping phase in the beginning of cw wave to avoid simulation diverging.
            `tau` is the ramping time constant.
            `t_before` is the total time before reaching constant magnitude in cw.
        `pulse` (single Gaussian pulse),
            Need to specify `tau`, `t_before` (optional)
            `tau is the width of the pulse.
            `t_before` is the total time before the peak of the pulse in the simulation.
        `packet` (cw wave modulated by a Gaussian pulse)
            Need to specify `omega`, `phase` (optional), `tau`, `t_before`, `t_span`.
            `tau` is the width of the pulse profile.
            `t_before` is the total time before reaching the peak in amplitude.
            `t_span` is the amount of time the wave stay at its peak amplitude, i.e. the wave packet can be such that there is a cw section between the ramping-up and the ramping-down stages.

        Parameters
        ----------
        kind            :   str
                            ['pulse'|'cw'|'packet']
        omega           :   float
                            frequency
        tau             :   float
                            time-constant for Gaussian pulse, or the wave packet
        phase           :   float
                            phase shift of the wave
        t_before        :   float
                            total time before the peak of the Gaussian. default to be 6 times of tau
        t_span          :   float
                            when the amplitude reaches the peak of a wave packet, maintain t_span time of cw

        Returns
        -------
        t_profile       :   Gsn or Hm or HP
                            the temporal profile object

        """
        t_profile = None
        if kind == 'pulse':
            t_profile = Gsn(self.c, tau, t_bf=t_before)
        elif kind == 'cw':
            t_profile = Hm(self.c, omega, phi=phase, t_bf=t_before, t_r=tau)
        elif kind == 'packet':
            t_profile = HP(self.c, omega, phi=phase, t_spn=t_span, t_bf=t_before, t_r=tau)
        else:
            warn('Temporal profile not recognized.', UserWarning)

        self.tps.append(t_profile)
        return t_profile

    def use_ndc(self, if_use_ndc, omega):
        """
        Use numeric dispersion correction in the main solver. This corrects the phase velocity of waves in vacuum to be light speed.

        Notes
        -----
        This is not to be confused with the numeric dispersion correction in sources. If numeric dispersion correction is used in sources, then the user shouldn't enable this in the main solver.

        In general, increasing spatial resolution reduces numeric dispersion.

        Parameters
        ----------
        if_use_ndc          :   bool
        omega               :   float
                                frequency for ndc

        Returns
        -------

        """
        if if_use_ndc:
            self.if_ndc = True
            self.omg_ndc = omega
            if self.rt:
                self.rt.set_ndc(if_use_ndc, omega)
        else:
            self.if_ndc = False

    def add_point_source(self, x, y, amp, t_profile):
        """
        Add a point source in the space

        Parameters
        ----------
        x, y        :   float
                        position of the point source
        amp         :   float
                        amplitude of the source
        t_profile   :   Hm or HP or Gsn
                        source temporal profile

        -------

        """
        sc = PS2(x, y, amp, t_profile, self.st, plrz=self.polarization)

        self.scs.append(sc)

    def add_tfsf_source(self, xmin, xmax, ymin, ymax,
                        kx, ky, amp, t_profile,
                        epsi=1., mu=1.,
                        if_ndc=False, omega_ndc=None,
                        sides='all'):
        """
        Add a TFSF source in the space

        Parameters
        ----------
        xmin, xmax ymin ymax    :   float
                                    four corners of TFSF source region
        kx, ky                  :   float
                                    kx and ky determine incident direction.

                                    The magnitudes of kx and ky don't matter, only the direction matters.

                                    The incident wave is assumed to be travelling in free space so it's k is determined by its frequency.

        amp                     :   float
                                    amplitude of the incident wave
        t_profile               :   Hm or Gsn or HP
                                    source temporal profile
        epsi, mu                :   float
        if_ndc                  :   bool
                                    if use numeric dispersion correction. If enabled, the phase velocity of the source fields will be the exact numerical one, not the analytic light speed in the space.
        omega_ndc               :   float
                                    the phase velocity of the source field will be the numeric phase velocity at this frequency
        sides                   :   str
                                    {'all', combinations of 't', 'b', 'l', 'r'}

                                    Controls which sides of the TFSF source exist.

                                    Default to be 'all', i.e. TFSF source in a rectangular region.

                                    Can choose any combination of 't', 'b', 'l', 'r' for top, bottom, left and right.

                                    If choose any ONE of the sides, it becomes a uni-directional plane-wave source.

        Returns
        -------

        """

        sc = TSS2(xmin, xmax, ymin, ymax, kx, ky, amp, t_profile, self.st, self.c.c0, self.dt, plrz=self.polarization, if_ndc=if_ndc, omg_ndc=omega_ndc, epsi=epsi, mu=mu, whr=sides)

        self.scs.append(sc)

    def add_tfsf_mode_source(self, xmin, xmax, ymin, ymax,
                             kx, ky, amp, t_profile,
                             xi, f, epsi, mu, omega,
                             xi0=None, reverse_direction=False,
                             if_ndc=False,
                             sides='all'):
        """
        Add a TFSF source of a waveguide mode.

        Parameters
        ----------
        xmin, xmax, ymin, ymax  :   float
                                    four corners of TFSF source region
        kx, ky                  :   float
                                    (kx, ky) determine the wave propagation constant. Both the magnitude and the direction of (kx, ky) matters. This is different from `add_tfsf_source`.
        amp                     :   float
                                    amplitude of the incident wave
        t_profile               :   Hm or Gsn or HP
                                    source temporal profile
        xi                      :   array_like
                                    1d array, spatial points where f is defined. See notes on `f` below.
        f                       :   array_like
                                    1d array value of f, the waveguide mode.

                                    For Ez mode, `f` is Ez. For Hz mode, `f` is Hz.

                                    `xi` and `f` can be analytically calculated or numerically simulated.

                                    `xi` are the actual physical points where `f` is defined.
                                    For Ez mode, `xi` is where Ez is defined. For Hz mode, `xi` is where Hz is defined.
                                    In fact `xi` does not know about Yee cells in general (for example, analytically simulated).

                                    If `xi` and `f` were simulation results of `tdyno` with a waveguide in the x direction, then for `Ez` polarization `xi` are the Yee cell corners i.e. grid points in y, while for `Hz` polarization `xi` are the half grid points in y in each Yee cell.

        epsi                    :   array_like
                                    2d array. The permittivity profile along the transverse direction of the waveguide.

                                    `epsi[i]` defines the values at index i. It has 2 elements, `epsi[i][0]` is the component  along the waveguide direction, `epsi[i][1]` is transverse to the waveguide.

        mu                      :   array_like
                                    2d array. The permeability profile along the transverse direction of the waveguide.

                                    `mu[i]` defines the values at index i. It has 2 elements, `mu[i][0]` is the component along the waveguide direction, `mu[i][1]` is transverse to the waveguide.

        omega                   :   float
                                    intended frequency of the waveguide mode. This frequency will also be used for ndc, i.e. the phase velocity of the source field will be the numeric phase velocity at this frequency
        xi0                     :   float
                                    signed distance from reference line (xi=0) to origin
        reverse_direction       :   bool
                                    If `True`, reverse `beta` direction, while keeping profile `f` unchanged.

                                    Can instead manually set kx and ky to negative, in which case it is rotating both `beta` and `f` 180 degrees. Note, for asymmetric waveguide, modal profile `f` is asymmetric.
        if_ndc                  :   bool
                                    if use numeric dispersion correction. If enabled, the phase velocity of the source fields will be the exact numerical one, not the analytic light speed in the space.

                                    If True, do not use ndc in the main update equations.

        sides                   :   str
                                    {'all', combinations of 't', 'b', 'l', 'r'}

                                    Controls which sides of the TFSF source exist.

                                    Default to be 'all', i.e. TFSF source in a rectangular region.

                                    Can choose any combination of 't', 'b', 'l', 'r' for top, bottom, left and right.

                                    If choose any ONE of the sides, it becomes a uni-directional plane-wave source.

        Notes
        -----
        The source does not know the actual structure in the solving space. This is to say, if the structure is the intended waveguide then this mode will be generated in the TF/SF fashion.

        To generate the waveguide mode, the permittivity and permeability are needed and are supplied. Again, this is regardless of the actual structure in the solving space.

        """

        sc = TSSMD2(xmin, xmax, ymin, ymax, kx, ky, amp, t_profile, xi, f, epsi, mu, omega, self.st, self.c.c0, self.dt, xi0=xi0, reverse_direction=reverse_direction, plrz=self.polarization, if_ndc=if_ndc, omg_ndc=omega, whr=sides)

        self.scs.append(sc)

    def add_pml(self, poly_ord, ratio, amax_factor=1., kappa=1., npx=0, npy=0):
        """
        Add PML to the space

        Parameters
        ----------
        poly_ord        :   float or Tuple[float, float, float]
                            polynomial order. Tuple of 3 if want different polynomial order for x, y, and z direction.
        ratio           :   float or Tuple[float, float, float]
                            target attenuation ratio
        amax_factor     :   float
                            a_max factor. Adjusted to reduce low frequency reflectance. Coefficient a_max is computed automatically, but one can manually scale this factor.
        kappa           :   float or Tuple[float, float, float]
                            kappa, the kappa parameter stretches the real part of xyz in PML.
                            If the material itself has some absorption, this helps with attenuation.
                            However, this is at the cost of all partial derivatives in the update equations will have to multiplied with the kappa matrix (it is some value in PML but is 1 in the solving domain).
        npx, npy        :   int
                            number of pml cells in x and y (npx=10 means 10 pml cells on each end).


        Returns
        -------

        """

        self.pc = PCT(self.dt, poly_ord, ratio, a_max_fct=amax_factor, kpp=kappa, Nx=self.st.Nx, Ny=self.st.Ny, dx=self.st.dx, dy=self.st.dy, npx=npx, npy=npy, npz=0, epsi0=self.c.epsi0, mu0=self.c.mu0)

    def solving_mode(self, mode):
        """
        Choose the solving mode.
        If permeability is 1 throughout the space, choose 'HDE' for faster solving.

        Parameters
        ----------
        mode        :   str
                        'HDE' or 'BHDE'

        Returns
        -------

        """
        self.md = mode

    def add_point_monitors(self, coords, delay=0, weights=None, omega_min=0., omega_max=2., n_omega=100, norm_factor=1., flux=False, refer_spectrum=None):
        """
        Add monitors to monitor the amplitude of the field on multi points. Record the amplitudes on each, do weighted sum, and do Fourier Transform.
        The monitors only record Ez and Hz for the two corresponding polarizations, respectively.

        Parameters
        ----------
        delay                   :   int
                                    time delay. Won't start Fourier Transform until this time step.
        coords                  :   list[Tuple[float, float]]
                                    coordinates of the points to monitor. Each element is a tuple of (x, y)
        weights                 :   list[float]
                                    the amplitudes of the monitored points are multiplied by these weights, and then add up
        omega_min, omega_max    :   float
        n_omega                 :   int
                                    number of frequency points
        norm_factor             :   float
                                    normalization factor for spectrum
        flux                    :   bool
                                    if true, plot the flux spectral density. Otherwise plot the power spectral density
        refer_spectrum          :   ndarray[float]
                                    reference spectrum. If supplied, the shown spectrum will be divided by this. This is for convenience, for example, if want to see transmission spectrum.

        Returns
        -------

        """
        if flux is True:
            show = 'flux spectral density'
        else:
            show = 'energy spectral density'
        mnt = MntMltPntAmp(self.st, coords, dt=self.dt, td=delay, wts=weights, omin=omega_min, omax=omega_max, n_o=n_omega, nmf=norm_factor, show=show, ref_spctrm=refer_spectrum)
        self.mnts.append(mnt)

    def add_poynting_monitor(self,
                             xmin=None, xmax=None, ymin=None, ymax=None,
                             delay=None,
                             omega_min=None, omega_max=None, n_omega=None,
                             norm_factor=None, refer_spectrum=None,
                             sides='all'):
        """
        Add Poynting flux monitor as a square box. Tracks the total energy flux from inside the box to outside in real time and calculates the spectrum.

        Parameters
        ----------
        xmin : float
        xmax : float
        ymin : float
        ymax : float
        delay : int
            delay starting monitor by this many time steps
        omega_min : float
        omega_max : float
        n_omega : int
        norm_factor : float
        refer_spectrum : np.ndarray
        sides : str
            Can be "all", or any combinations (any order) of 'l', 'r', 't', 'b'.

        Returns
        -------

        """
        mnt = Mnt2DSqPoynU(st=self.st,
                           xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                           dt=self.dt, td=delay,
                           omin=omega_min, omax=omega_max, n_o=n_omega,
                           nmf=norm_factor, ref_spctrm=refer_spectrum,
                           whr=sides, plrz=self.polarization)
        self.mnts.append(mnt)

    def add_mode_monitor(self,
                         x=None, y=None,
                         xmin=None, xmax=None, ymin=None, ymax=None,
                         ):
        """
        A monitor to record modal profile at an interface in either x or y.

        Parameters
        ----------
        x : float
            x position of the interface if it's in y
        y : float
            y position of the interface if it's in x
        xmin, xmax, ymin, ymax : float
            min and max of the interface. Default to the edge of the solving space.

        Notes
        -----
        If `x` is supplied, `ymin` and `ymax` are needed. If `y is supplied, `xmin` and `xmax` are needed. If both `x` and `y` are supplied, `y` is ignored.
        """

        mnt = MntWgMdU(st=self.st,
                       x=x, y=y,
                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                       plrz=self.polarization)
        self.mnts.append(mnt)

    def run(self, skipping=5, vmin=-1., vmax=1., **kwargs):
        """
        Run the FDTD simulation.

        Parameters
        ----------
        skipping        :   int
                            only plot the fields once every `skipping` time steps, to save time
        vmin, vmax      :   float
                            the colormap range for the field strength in plotting
        kwargs          ：  dict
                            key word arguments for PlotField2DinAx

        Returns
        -------

        """

        self.rt = RT2(self.c.c0, self.c.epsi0, self.c.mu0,
                      self.st,
                      self.scs,
                      self.dt, self.Nt,
                      pc=self.pc,
                      plrz=self.polarization,
                      if_ndc=self.if_ndc, omg_ndc=self.omg_ndc,
                      md=self.md,
                      mnts=self.mnts,
                      vmin=vmin, vmax=vmax,
                      skp=skipping,
                      **kwargs)

        plt.show()
