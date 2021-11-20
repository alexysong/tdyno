import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.gridspec import GridSpec
from warnings import warn


class MntUI1S1S:
    def __init__(self,
                 signal_x_label=None,
                 signal_y_label=None,
                 spectrum_x_label=None,
                 spectrum_y_label=None,
                 sx=None, sy=None,
                 Sx=None, Sy=None
                 ):
        """
        Monitor UI, one signal, one spectrum.

        Parameters
        ----------
        signal_y_label : str
        spectrum_y_label : str
        sx : array_like
        sy : array_like
            sx and sy are the x and y axis of the "signal"
        Sx : array_like
        Sy : array_like
            Sx and Sy are the x and y axis of the "spectrum"
        """

        self.sx = sx
        self.sy = sy
        self.Sx = Sx
        self.Sy = Sy

        # Start plotter
        mpl.rcParams['mathtext.fontset'] = 'cm'
        fontsize_label = 20
        fontsize_axis = 16
        gs = GridSpec(nrows=2, ncols=2, height_ratios=[1, 15])

        if signal_x_label is None:
            signal_x_label = 'Time'
        if signal_y_label is None:
            signal_y_label = 'Signal'
        if spectrum_x_label is None:
            spectrum_x_label = "$\omega$"
        if spectrum_y_label is None:
            spectrum_y_label = "Energy spectral density"
        self.fig = plt.figure(figsize=(14, 5))
        ax_f = self.fig.add_subplot(gs[1, 0])
        ax_f.set_xlabel(signal_x_label, fontsize=fontsize_label)
        ax_f.set_ylabel(signal_y_label, fontsize=fontsize_label)
        ax_f.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_f = ax_f
        x, y = [s if (s is not None) else [] for s in [sx, sy]]
        self.ln_f = ax_f.plot(x, y)[0]

        ax_F = self.fig.add_subplot(gs[1, 1])
        ax_F.set_xlabel(spectrum_x_label, fontsize=fontsize_label)
        ax_F.set_ylabel(spectrum_y_label, fontsize=fontsize_label)
        ax_F.tick_params(axis='both', which='major', labelsize=fontsize_axis)
        self.ax_F = ax_F
        x, y = [s if (s is not None) else [] for s in [Sx, Sy]]
        if Sx is None:
            x = []
        else:
            x = Sx
            if Sy is None:
                y = np.zeros(len(Sx))
        self.ln_F = ax_F.plot(x, y)[0]
        if hasattr(Sx, '__len__'):
            ax_F.set_xlim([Sx[0], Sx[-1]])

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

    def up(self,
           sx=None, sy=None,
           Sx=None, Sy=None):
        """
        Update plots

        Parameters
        ----------
        sx : array_like
        sy : array_like
        Sx : array_like
        Sy : array_like

        Returns
        -------

        """
        self.sx, self.sy, self.Sx, self.Sy = [s if (s is not None) else ss for s, ss in zip([sx, sy, Sx, Sy], [self.sx, self.sy, self.Sx, self.Sy])]

        if (sx is not None) or (sy is not None):
            self.ln_f.set_data(self.sx, self.sy)
            self.ax_f.set_xlim([self.sx[0], self.sx[-1]])
            self.ax_f.set_ylim([min(self.sy), max(self.sy)])
            self.ax_f.draw_artist(self.ax_f.patch)
            self.ax_f.draw_artist(self.ln_f)

        if Sx is not None:  # most likely this does not change
            self.ln_F.set_xdata(Sx)
            self.ax_F.set_xlim([Sx[0], Sx[-1]])
            self.ax_F.draw_artist(self.ax_F.patch)
            self.ax_F.draw_artist(self.ln_F)
        if Sy is not None:
            self.ln_F.set_ydata(Sy)
            self.ax_F.set_ylim([min(Sy), max(Sy)])
            self.ax_F.draw_artist(self.ax_F.patch)
            self.ax_F.draw_artist(self.ln_F)

        self.fig.canvas.flush_events()
        self.fig.canvas.update()

    def sv_rec(self, event):
        """
        save recorded monitor values

        Returns
        -------

        """

        data = np.concatenate([[self.sx], [self.sy]], axis=0).T
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

        data = np.concatenate([[self.Sx], [self.Sy]], axis=0).T
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