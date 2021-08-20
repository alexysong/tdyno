"""
Created by Alex Y. Song, 2021.

"""

import numpy as np
from matplotlib import pyplot as plt
from warnings import warn
from typing import Callable, Iterator, Union, Optional, List
from .cnst import Cnst
from .s2t import S2T
from .pct import PCT
from .sc_tp_t import Hm, HP, Gsn
from .sc2t import PS2, TSS2
from .rt2 import RT2
from .mnt_t import MntPntAmp
from .mnt_t import MntMltPntAmp


class TDyno:
    def __init__(self):
        """
        Finite-difference time-domain (FDTD) package with dynamic modulations in the refractive index and/or in gain/loss.

        """
        self.c = Cnst()
        self.st = None
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

        self.st = S2T(xmin, xmax, dx, ymin, ymax, dy, epsi, mu, x=x, y=y)
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
        ‘Lorentz modulated' means Lorentz model with modulated resonance strengh. It needs the following material parameters:
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
        use numeric dispersion correction

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
            for sc in self.scs:
                if hasattr(sc, 'set_ndc') and callable(getattr(sc, 'set_ndc')):
                    sc.set_ndc(if_use_ndc, omega)
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

    def add_tfsf_source(self, xmin, xmax, ymin, ymax, kx, ky, amp, t_profile, epsi=1., mu=1., sides='all'):
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
        sides                   :   str
                                    ['all'|'top'|'bottom'|'left'|'right']
                                    Controls which sides of the TFSF source exist.
                                    Default to be 'all', i.e. TFSF source in a rectangular region.
                                    If choose any of the sides, it becomes a uni-directional plane-wave source.

        Returns
        -------

        """

        if sides == 'left':
            where = 'l'
        elif sides == 'right':
            where = 'r'
        elif sides == 'top':
            where = 't'
        elif sides == 'bottom':
            where = 'b'
        elif sides == 'all':
            where = 'all'
        else:
            where = 'all'
            warn('The side of TFSF source not recognized. Default to "all".', UserWarning)

        sc = TSS2(xmin, xmax, ymin, ymax, kx, ky, amp, t_profile, self.st, self.c.c0, self.dt, plrz=self.polarization, if_ndc=self.if_ndc, omg_ndc=self.omg_ndc, epsi=epsi, mu=mu, whr=where)

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

    def add_point_monitors(self, delay=0., coords=None, weights=None, omega_min=0., omega_max=2., n_omega=100, norm_factor=1., power=True, flux=False, refer_spectrum=None):
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
        power                   :   bool
                                    if true, plot the power spectral density.
        flux                    :   bool
                                    if true, plot the flux. Overrides `power`.
        refer_spectrum          :   ndarray[float]
                                    reference spectrum. If supplied, the shown spectrum will be divided by this. This is for convenience, for example, if want to see transmission spectrum.

        Returns
        -------

        """
        mnt = MntMltPntAmp(self.st, dt=self.dt, td=delay, coords=coords, wts=weights, omin=omega_min, omax=omega_max, n_o=n_omega, nmf=norm_factor, psd=power, flx=flux, ref_spctrm=refer_spectrum)
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
