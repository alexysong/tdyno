# -*- coding: utf-8 -*-


import numpy as np


class SigSpct:
    def __init__(self, dt, td=None, omin=None, omax=None, n_o=None, nmf=None, ref_spctrm=None):
        """"
        Given a time-series signal, calculate its Fourier spectrum (FT) as signal comes in.

        signal value is fed in one after another. The FT of the signal is updated in real time. Both the FT, the power spectral density (psd) and the flux are calculated

        Parameters
        ----------
        dt : float
            time step length
        td : float
            time steps delay before starting to calculate the spectrum
        omin : float
            lower bound of omega
        omax : float
            upper bound of omega
        n_o : int
            number of frequency points
        nmf : float
            normalization factor for spectrum
        ref_spctrm : np.ndarray
            reference energy spectral density spectrum to be divided by.
        """
        if dt is None:
            dt = 1.
        self.dt = dt
        if td is None:
            td = 0.
        self.td = td
        self.nmf = nmf
        self.ref_spctrm = ref_spctrm
        self.omin = omin
        if omin is None:
            omin = np.pi / self.dt / 1000
        self.omin = omin
        if omax is None:
            omax = np.pi / self.dt
        self.omax = omax
        if n_o is None:
            n_o = 201
        self.n_o = n_o
        self.omg = np.linspace(self.omin, self.omax, n_o)
        if nmf is None:
            nmf = 1.
        self.nmf = nmf
        if ref_spctrm is None:
            ref_spctrm = 1.
        self.ref_spctrm = ref_spctrm

        self.t_n = []  # list of integer time steps, last element is current
        self.ts = []  # list of real time steps, last element is current
        self._nt = 0   # next time step

        self.f = []  # real time signal

        self.knl = np.exp(-1j * self.omg * self.dt)  # kernel

        # Initialize the Fourier transform of the field
        self.reset()
        self.F = np.zeros(n_o) * 1j  # FT
        self.esd = np.zeros(n_o)  # energy spectral density
        self.psd = np.zeros(n_o)  # power spectral density
        self.flx = np.zeros(n_o)  # integral number flux spectral density
        self.flxr = np.zeros(n_o)  # number flux rate spectral density

    def rnf(self, f):
        """
        Receive new signal

        Parameters
        ----------
        f : float

        """
        self.f.append(f)
        t = self._nt * self.dt

        if self._nt > self.td:
            self.F += (self.knl ** self._nt) * f * self.dt
            self.esd = np.square(np.abs(self.F)) / self.nmf / self.ref_spctrm
            self.psd = self.esd / (t - self.dt * self.td)
            self.flx = self.esd / self.omg
            self.flxr = self.flx / (t - self.dt * self.td)

        self.t_n.append(self._nt)
        self.ts.append(self._nt * self.dt)
        self._nt += 1

    def reset(self):
        """
        reset/initialize signals and spectra

        """
        self.ts = []
        self.t_n = []
        self._nt = 0
        self.f = []

        n_o = self.n_o
        self.F = np.zeros(n_o) * 1j  # FT
        self.esd = np.zeros(n_o)  # energy spectral density
        self.psd = np.zeros(n_o)  # power spectral density
        self.flx = np.zeros(n_o)  # integral number flux spectral density
        self.flxr = np.zeros(n_o)  # number flux rate spectral density

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    def onclick(event):

        for f in sig:
            s.rnf(f)
            plt.pause(0.01)
            ln_f.set_data(s.ts, s.f)
            data = s.flxr
            ln_F.set_ydata(data)
            ax.set_xlim([s.ts[0], s.ts[-1]])
            ax2.set_ylim([data.min(), data.max()])
            fig.canvas.flush_events()
            fig.canvas.update()

    dt = 0.02
    omega = 2 * np.pi
    sig = np.sin(omega * np.arange(0., 20, dt))

    s = SigSpct(dt=dt, omin=omega/2, omax=omega*2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ln_f = ax.plot(s.t_n, s.f)[0]
    ax.set_ylim([-1, 1])
    ln_F = ax2.plot(s.omg, s.psd)[0]
    # ax.plot(np.arange(10), np.arange(10))

    # if this, the animation works fine.
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
