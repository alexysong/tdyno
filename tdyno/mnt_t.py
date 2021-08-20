#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song, Dec. 2017

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox
from matplotlib.gridspec import GridSpec

from .s2t import S2T


class MntPntAmp:
    def __init__(self, st, dt=1., td=0, x=0., y=0., omin=0., omax=None, n_o=1000, nmf=1., psd=True, flx=False, ref_spctrm=None):
        """
        monitor the amplitude of field on a point. Record the amplitude, and do Fourier Transform.

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

    def rnf(self, f):
        """
        receive new field

        Parameters
        ----------
        f           :   ndarray

        Returns
        -------

        """
        self.t_n = np.append(self.t_n, self.nt)

        f_mnt = f[self.idx_mnt]
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
    def __init__(self, st, dt=1., td=0, coords=None, wts=None, omin=0., omax=None, n_o=1000, nmf=1., psd=True, flx=False, ref_spctrm=None):
        """
        Monitor the amplitude of field on multi points. Record the amplitudes on each, do weighted sum, and do Fourier Transform.

        st          :   S2T
        dt          :   float
                        time resolution
        td          :   int
                        time delay. Won't start Fourier Transform until this time step.
        coords      :   list[Tuple]
                        coordinates of the points to monitor. Each element is a tuple of (x, y)
        wts         :   list
                        weights, the amplitudes of the monitored points are multiplied by these weights, and then add up
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
        self.st = st
        self.dt = dt
        self.td = td
        self.nmf = nmf
        self.psd = psd
        self.flx = flx
        self.ref_spctrm = ref_spctrm
        if wts is None:
            wts = [1.]
        self.wts = wts

        self.coords = []
        self.idx_mnts = []
        self.f_mnts = []
        if coords is None:
            coords = [(0., 0.)]
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

            # Initialize recorded field at each location of the monitors
            self.f_mnts.append(np.array([]))

        # Initialize weighted sum of recorded field
        self.f_mnt_ws = np.array([])

        # current time step
        self.nt = 0

        # list of time steps
        self.t_n = np.array([])

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
        self.F_sh = np.zeros(n_o)

        # Start plotter
        mpl.rcParams['mathtext.fontset'] = 'cm'
        fontsize_label = 20
        fontsize_axis = 16
        gs = GridSpec(nrows=2, ncols=2, height_ratios=[1, 15])

        self.fig = plt.figure(figsize=(14, 5))
        ax_f = self.fig.add_subplot(gs[1, 0])
        ax_f.set_xlabel('Time', fontsize=fontsize_label)
        ax_f.set_ylabel('Field Intensity', fontsize=fontsize_label)
        ax_f.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_f = ax_f
        self.ln_f = ax_f.plot(self.t_n*self.dt, self.f_mnt_ws)[0]

        ax_F = self.fig.add_subplot(gs[1, 1])
        ax_F.set_xlabel('$\omega$', fontsize=fontsize_label)
        if self.flx:
            ax_F.set_ylabel('Flux', fontsize=fontsize_label)
        else:
            ax_F.set_ylabel('Power Spectral Density', fontsize=fontsize_label)
        ax_F.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_F = ax_F
        self.ln_F = ax_F.plot(self.omg, np.square(np.abs(self.F_mnt)))[0]
        ax_F.set_xlim([self.omg.min(), self.omg.max()])

        self.pchs = []
        for (x, y) in self.coords:
            pch = {'shp': 'rct',
                    'xy': (x, y),
                    'width': st.dx,
                    'height': st.dy
                    }
            self.pchs.append(pch)

        gs.tight_layout(self.fig)

        self.rec_n = 0
        self.spct_n = 0

        # save recording button
        self.ax_sv_rec = plt.axes([0.05, 0.93, 0.1, 0.05])
        self.b_sv_rec = Button(self.ax_sv_rec, r'Save recording')
        self.b_sv_rec.on_clicked(self.sv_rec)

        # save recording title
        self.rec_ttl = 'rec'
        self.ax_rec_ttl = plt.axes([0.21, 0.93, 0.27, 0.05])
        self.tb_rec_ttl = TextBox(self.ax_rec_ttl, 'file name', initial=self.rec_ttl)
        self.tb_rec_ttl.on_text_change(self.set_rec_ttl)

        # save spectrum button
        self.ax_sv_spct = plt.axes([0.55, 0.93, 0.1, 0.05])
        self.b_sv_spct = Button(self.ax_sv_spct, r'Save spectrum')
        self.b_sv_spct.on_clicked(self.sv_spct)

        # save recording title
        self.spct_ttl = 'spectrum'
        self.ax_spct_ttl = plt.axes([0.71, 0.93, 0.27, 0.05])
        self.tb_spct_ttl = TextBox(self.ax_spct_ttl, 'file name', initial=self.spct_ttl)
        self.tb_spct_ttl.on_text_change(self.set_spct_ttl)

    def rnf(self, f):
        """
        Receive new field

        Parameters
        ----------
        f           :   ndarray

        Returns
        -------

        """
        self.t_n = np.append(self.t_n, self.nt)

        f_n_ws = 0.
        for m, (idx, f_mnt, wt) in enumerate(zip(self.idx_mnts, self.f_mnts, self.wts)):
            f_n = f[idx]
            f_n_ws += wt * f_n
            self.f_mnts[m] = np.append(f_mnt, f_n)
        self.f_mnt_ws = np.append(self.f_mnt_ws, f_n_ws)

        if self.nt > self.td:
            self.F_mnt += (self.knl ** self.nt) * f_n_ws * self.dt
            self.F_mnt_sq = np.square(np.abs(self.F_mnt)) / self.nmf
            if self.flx:
                self.F_mnt_flx = self.F_mnt_sq / self.omg
                self.F_sh = self.F_mnt_flx
            else:
                self.F_sh = self.F_mnt_sq
            if self.ref_spctrm is not None:
                self.F_sh /= self.ref_spctrm

        self.nt += 1

    def up(self):
        """
        Update plots

        Returns
        -------

        """
        self.ln_f.set_data(self.t_n*self.dt, self.f_mnt_ws)
        xmin = self.t_n.min() * self.dt
        xmax = self.t_n.max() * self.dt
        ymin = self.f_mnt_ws.min()
        ymax = self.f_mnt_ws.max()
        self.ax_f.set_xlim([xmin, xmax])
        self.ax_f.set_ylim([ymin, ymax])

        self.ax_f.draw_artist(self.ax_f.patch)
        self.ax_f.draw_artist(self.ln_f)
        self.fig.canvas.flush_events()

        self.ln_F.set_ydata(self.F_sh)
        self.ax_F.set_ylim([self.F_sh.min(), self.F_sh.max()])

        self.ax_F.draw_artist(self.ax_F.patch)
        self.ax_F.draw_artist(self.ln_F)
        self.fig.canvas.flush_events()

        self.fig.canvas.update()

    def cf(self):
        """
        Clear fields.
        Returns
        -------

        """
        self.nt = 0
        self.t_n = np.array([])
        self.f_mnts = [np.array([])] * len(self.coords)
        self.f_mnt_ws = np.array([])
        self.F_mnt = np.zeros(self.F_mnt.size) * 1j

    def sv_rec(self, event):
        """
        save recorded monitor values

        Returns
        -------

        """

        data = np.concatenate([[self.t_n], [self.f_mnt_ws]], axis=0).T
        ttl = self.rec_ttl + '_{:03d}.txt'.format(self.rec_n)
        np.savetxt(ttl, data)
        self.rec_n += 1

    def set_rec_ttl(self, tx):
        """
        set file title for recorded monitor values

        Parameters
        ----------
        tx      :   str

        Returns
        -------

        """
        self.rec_ttl = tx
        self.rec_n = 0

    def sv_spct(self, event):
        """
        save monitor spectrum

        Returns
        -------

        """

        data = np.concatenate([[self.omg], [self.F_sh]], axis=0).T
        ttl = self.spct_ttl + '_{:03d}.txt'.format(self.spct_n)
        np.savetxt(ttl, data)
        self.spct_n += 1

    def set_spct_ttl(self, tx):
        """
        set file title for monitor spectrum

        Parameters
        ----------
        tx      :   str

        Returns
        -------

        """
        self.spct_ttl = tx
        self.spct_n = 0

if __name__ == '__main__':

    def onclick(event):

        for ii in np.linspace(0., 2 * np.pi, 100):
            f = np.sin(st.xx) * np.sin(ii)
            mnt.rnf(f)
            mnt.up()
            plt.pause(0.01)


    # Initialize the structure to be an empty space
    dx = 0.1
    xmin = 0.
    xmax = 70.
    dy = 0.1
    ymin = -4.
    ymax = 4.
    epsi_bg = 1.
    mu_bg = 1.
    st = S2T(xmin, xmax, dx, ymin, ymax, dy, epsi_bg, mu_bg)

    dt = 0.01
    x_mnt = 20.
    y_mnt = 0.2
    mnt = MntPntAmp(st, x=x_mnt, y=y_mnt, dt=dt, omin=0.8, omax=1.5)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(10), np.arange(10))

    # if this, the animation will have a problem, the old lines will not be cleared until you resize the figure.
    # cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # if this, the animation works fine.
    cid = mnt.fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
