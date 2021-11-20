# -*- coding: utf-8 -*-


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from warnings import warn

from tdyno.s2t import S2T
from tdyno.mnt.sig_spct import SigSpct
from tdyno.mnt.mnt_ui import MntUI1S1S


class MntPntAmp:
    def __init__(self, st, dt=1., td=0, x=0., y=0., omin=0., omax=None, n_o=1000, nmf=1., psd=True, flx=False, ref_spctrm=None):
        """
        monitor the amplitude of field on a point. Record the amplitude, and do Fourier Transform.

        Parameters
        ----------
        st          :   structure
        dt          :   float
                        time resolution
        td          :   int
                        time delay. Won't start Fourier Transform until this time step.
        x, y        :   float
                        the coordinates of the monitor
        omin, omax  :   float
                        omega min and max for the spectrum
        n_o         :   int
                        number of frequency points
        nmf         :   float
                        normalization factor for spectrum
        psd         :   bool
                        if true, plot the power spectral density.
        flx         :   bool
                        if true, plot the flux. Overrides psd.
        ref_spctrm  :   ndarray[float]
                        reference spectrum. If supplied, the shown spectrum will be divided by this.
        """
        warn("This class is deprecated. Use `MntMltPntAmp` instead.", DeprecationWarning)
        self.st = st
        self.dt = dt
        self.td = td
        self.nmf = nmf
        self.psd = psd
        self.flx = flx
        self.ref_spctrm = ref_spctrm

        if x >= st.xmax:
            x = st.xmax - st.xres  #todo if x=st.xmax, still cause weird error
        if x < st.xmin:
            x = st.xmin
        if y >= st.ymax:
            y = st.ymax - st.xres
        if y < st.ymin:
            y = st.ymin
        x_n = int(round((x - st.xmin) / st.xres))
        y_n = int(round((y - st.ymin) / st.yres))
        self.idx_mnt = (st.xx_n == x_n) * (st.yy_n == y_n)

        # current time step
        self.nt = 0

        # list of time steps
        self.t_n = np.array([])

        # Initialize recorded field at the location of the monitor
        self.f_mnt = np.array([])

        # frequency space
        self.omin = omin
        if omax is None:
            self.omax = np.pi / self.dt
        else:
            self.omax = omax
        self.omg = np.linspace(self.omin, self.omax, n_o)

        # calculate the Fourier transform kernel
        self.knl = np.exp(-1j * self.omg * self.dt)

        # Initialize the Fourier transform of the field
        self.F_mnt = np.zeros(n_o) * 1j
        self.F_mnt_sq = np.zeros(n_o)
        self.F_mnt_flx = np.zeros(n_o)
        self.F_shn = np.zeros(n_o)

        # Start plotter
        mpl.rcParams['mathtext.fontset'] = 'cm'
        fontsize_label = 20
        fontsize_axis = 16
        self.fig = plt.figure(figsize=(14, 5))
        ax_f = self.fig.add_subplot(1, 2, 1)
        ax_f.set_xlabel('Time', fontsize=fontsize_label)
        ax_f.set_ylabel('Field Intensity', fontsize=fontsize_label)
        ax_f.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_f = ax_f
        self.ln_f = ax_f.plot(self.t_n*self.dt, self.f_mnt)[0]

        ax_F = self.fig.add_subplot(1, 2, 2)
        ax_F.set_xlabel('$\omega$ ($2\pi$)', fontsize=fontsize_label)
        if self.flx:
            ax_F.set_ylabel('Flux', fontsize=fontsize_label)
        else:
            ax_F.set_ylabel('Power Spectral Density', fontsize=fontsize_label)
        ax_F.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_F = ax_F
        self.ln_F = ax_F.plot(self.omg/2./np.pi, np.square(np.abs(self.F_mnt)))[0]
        ax_F.set_xlim([self.omg.min()/2./np.pi, self.omg.max()/2./np.pi])

        self.pch = {'shp': 'rct',
                    'xy': (x, y),
                    'width': st.dx,
                    'height': st.dy
                    }
        self.pchs = [self.pch]

    def rnf(self, fz, fx, fy):
        """
        receive new field

        Parameters
        ----------
        fz          :   ndarray

        Returns
        -------

        """
        self.t_n = np.append(self.t_n, self.nt)

        f_mnt = fz[self.idx_mnt]
        self.f_mnt = np.append(self.f_mnt, f_mnt)

        if self.nt > self.td:
            self.F_mnt += (self.knl ** self.nt) * f_mnt * self.dt
            self.F_mnt_sq = np.square(np.abs(self.F_mnt)) / self.nmf
            if self.flx:
                self.F_mnt_flx = self.F_mnt_sq / self.omg
                self.F_shn = self.F_mnt_flx
            else:
                self.F_shn = self.F_mnt_sq
            if self.ref_spctrm is not None:
                self.F_shn /= self.ref_spctrm

        self.nt += 1

    def up(self):
        """
        update plots

        Returns
        -------

        """
        self.ln_f.set_data(self.t_n*self.dt, self.f_mnt)
        xmin = self.t_n.min() * self.dt
        xmax = self.t_n.max() * self.dt
        ymin = self.f_mnt.min()
        ymax = self.f_mnt.max()
        self.ax_f.set_xlim([xmin, xmax])
        self.ax_f.set_ylim([ymin, ymax])

        self.ax_f.draw_artist(self.ax_f.patch)
        self.ax_f.draw_artist(self.ln_f)
        self.fig.canvas.flush_events()

        self.ln_F.set_ydata(self.F_shn)
        self.ax_F.set_ylim([self.F_shn.min(), self.F_shn.max()])

        self.ax_F.draw_artist(self.ax_F.patch)
        self.ax_F.draw_artist(self.ln_F)
        self.fig.canvas.flush_events()

        self.fig.canvas.update()

    def cf(self):
        """
        clear fields.

        Returns
        -------

        """
        self.nt = 0
        self.t_n = np.array([])
        self.f_mnt = np.array([])
        self.F_mnt = np.zeros(self.F_mnt.size) * 1j


class MntMltPntAmp:
    def __init__(self, st, coords, dt=None, td=None, wts=None, omin=None, omax=None, n_o=None, nmf=None, show=None, ref_spctrm=None):
        """
        Monitor the amplitude of field on multi points. Record the amplitudes on each, do weighted sum, and do Fourier Transform.

        Parameters
        ----------
        st          :   S2T
        dt          :   float
                        time resolution
        td          :   int
                        time delay. Won't start Fourier Transform until this time step.
        coords      :   list[tuple[float, float]]
                        coordinates of the points to monitor. Each element is a tuple of (x, y)
        wts         :   list
                        weights, the amplitudes of the monitored points are multiplied by these weights, and then add up
        omin, omax  :   float
                        omega min and max for the spectrum
        n_o         :   int
                        number of frequency points
        nmf         :   float
                        normalization factor for spectrum
        show        :   str
                        {'energy spectral density', 'power spectral density', 'flux spectral density', 'flux rate spectral density'}

                        showing what kind of Fourier Transform of the field.

                        "flux" means photon flux, i.e. dividing intensity by frequency.

        if_esd      :   bool
                        if true, plot the energy spectral density.
        if_psd      :   bool
                        if true, plot the energy spectral density.
        if_flx      :   bool
                        if true, plot the flux. Overrides if_esd.
        ref_spctrm  :   ndarray[float]
                        reference spectrum. If supplied, the shown spectrum will be divided by this.
        """
        if wts is None:
            wts = [1.]
        self.wts = wts

        if (show != 'energy spectral density') and (show != 'power spectral density') and (show != 'flux spectral density') and (show != 'flux rate spectral density'):
            warn('the choice of the spectrum to show not understood. Default to energy spectral density', UserWarning)
            show = 'energy spectral density'
        self.show = show

        self.coords = []
        self.idx_mnts = []
        self.f_mnts = []
        if coords is None:
            coords = [(0., 0.)]
        if not hasattr(coords[0], '__len__'):
            coords = [coords]
        for (x, y) in coords:
            if x >= st.xmax:
                x = st.xmax - st.xres
            if x < st.xmin:
                x = st.xmin
            if y >= st.ymax:
                y = st.ymax - st.xres
            if y < st.ymin:
                y = st.ymin
            self.coords.append((x, y))

            x_n = int(round((x - st.xmin) / st.xres))
            y_n = int(round((y - st.ymin) / st.yres))
            idx_mnt = (st.xx_n == x_n) * (st.yy_n == y_n)
            self.idx_mnts.append(idx_mnt)

        # Initialize weighted sum of recorded field
        self.s_ws = SigSpct(dt, td, omin, omax, n_o, nmf, ref_spctrm)

        self.F = None  # spectrum shown

        self.mui = MntUI1S1S(signal_y_label="Field strength", spectrum_y_label=self.show, Sx=self.s_ws.omg)  # monitor UI

        self.pchs = []
        for (x, y) in self.coords:
            pch = {'shp': 'rct',
                    'xy': (x, y),
                    'width': st.dx,
                    'height': st.dy
                    }
            self.pchs.append(pch)

    def rnf(self, fz, fx, fy):
        """
        Receive new field

        Parameters
        ----------
        f           :   ndarray

        Returns
        -------

        """
        f_n_ws = 0.
        for idx, w in zip(self.idx_mnts, self.wts):
            f_n_ws += w * fz[idx]
        self.s_ws.rnf(f_n_ws)

    def up(self):
        """
        Update plots

        Returns
        -------

        """

        if self.show == 'power spectral density':
            F = self.s_ws.psd
        elif self.show == 'flux spectral density':
            F = self.s_ws.flx
        elif self.show == 'flux rate spectral density':
            F = self.s_ws.flxr
        else:
            F = self.s_ws.esd
        self.F = F

        self.mui.up(sx=self.s_ws.ts, sy=self.s_ws.f, Sy=self.F.ravel())

    def cf(self):
        """
        Clear fields.
        """
        self.s_ws.reset()
