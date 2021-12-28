# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.gridspec import GridSpec
from warnings import warn


class MntMdUI:
    def __init__(self,
                 xi, epsi, mu,
                 x_label=None,
                 y_label=None,
                 f=None,
                 plrz=None
                 ):
        """
        Monitor UI for waveguide mode.

        Parameters
        ----------
        x_label, y_label : str
        xi : array_like
            1d array
        f : array_like
            1d array
        epsi, mu : array_like
            2d array of shape (len(xi), 2)
        plrz : str
            only needed for file name. if not supplied, it is ignored.
        """

        if f is None:
            f = np.zeros(len(xi))

        self.xi = xi
        self.f = f
        self.epsi = epsi
        self.mu = mu

        if plrz is None:
            plrz = ''
        self.plrz = plrz

        self.mm = 0

        # Start plotter
        mpl.rcParams['mathtext.fontset'] = 'cm'
        fontsize_label = 20
        fontsize_axis = 16
        gs = GridSpec(nrows=2, ncols=2, height_ratios=[1, 15])

        if x_label is None:
            x_label = r'$\xi$'
        if y_label is None:
            y_label = 'amplitude'

        _xi, _f, _epsi, _mu = [s if (s is not None) else [] for s in [xi, f, epsi, mu]]

        self.fig = plt.figure(figsize=(14, 5))
        ax_f = self.fig.add_subplot(gs[1, 0])
        ax_f.set_xlabel(x_label, fontsize=fontsize_label)
        ax_f.set_ylabel(y_label, fontsize=fontsize_label)
        ax_f.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_f = ax_f
        self.ln_f = ax_f.plot(_xi, _f)[0]
        if hasattr(_xi, '__len__'):
            ax_f.set_xlim([_xi[0], _xi[-1]])

        ax_em: axes.Axes = self.fig.add_subplot(gs[1, 1])
        ax_em.set_xlabel(x_label, fontsize=fontsize_label)
        ax_em.set_ylabel(r"$\varepsilon$, $\mu$", fontsize=fontsize_label)
        ax_em.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_em = ax_em
        self.lns_epmu = ax_em.plot(_xi, epsi, _xi, mu)
        ax_em.legend([r"$\varepsilon_x$", r"$\varepsilon_y$", r"$\mu_x$", r"$\mu_y$"], fontsize=fontsize_label)
        if hasattr(_xi, '__len__'):
            ax_em.set_xlim(left=_xi[0], right=_xi[-1])

        gs.tight_layout(self.fig)

        self.rec_n = 0

        # save button
        self.ax_sv_rec = plt.axes([0.05, 0.93, 0.1, 0.05])
        self.b_sv_rec = Button(self.ax_sv_rec, r'Save')
        self.b_sv_rec.on_clicked(self.sv_rec)

        # save title
        self.rec_ttl = 'mode'
        self.ax_rec_ttl = plt.axes([0.21, 0.93, 0.27, 0.05])
        self.tb_rec_ttl = TextBox(self.ax_rec_ttl, 'file name', initial=self.rec_ttl)
        self.tb_rec_ttl.on_text_change(self.set_rec_ttl)

    def up(self,
           f=None,
           ):
        """
        Update plots

        Parameters
        ----------
        f : array_like
        """

        if f is not None:
            self.f = f
            self.ln_f.set_ydata(self.f)
            mm = max(max(np.abs(self.f)), self.mm)
            self.mm = mm
            self.ax_f.set_ylim([-mm, mm])
            self.ax_f.draw_artist(self.ax_f.patch)
            self.ax_f.draw_artist(self.ln_f)

        self.fig.canvas.flush_events()
        self.fig.canvas.update()

    def sv_rec(self, event):
        """
        save recorded monitor values

        Returns
        -------

        """

        ttl = [self.rec_ttl + '_' + self.plrz + '_' + c + '_{:03d}.txt'.format(self.rec_n) for c in ["xi", "f", "epsi", "mu"]]

        for t, d in zip(ttl, [self.xi, self.f, self.epsi, self.mu]):
            np.savetxt(t, d)

        print("Data saved. See files: ")
        for t in ttl:
            print(t)

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
