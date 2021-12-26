# -*- coding: utf-8 -*-


import numpy as np


class SigSpct:
    def __init__(self, dt, td=None, omin=None, omax=None, n_o=None, nmf=None, ref_spctrm=None, n=None):
        """"
        Given a set of independent time-series signals, calculate their respective Fourier spectrum (FT) as the signals comes in.

        Signal value is fed in at each time step. The FT of the signal is updated as they come in. The FT, the energy (esd) and power spectral density (psd), the number flux spectrum (flx) and flux rate spectrum (flxr) are calculated.

        Parameters
        ----------
        dt : float
            time step length
        td : int
            time steps delay before starting to calculate the spectrum
        omin : float
            lower bound of omega
        omax : float
            upper bound of omega
        n_o : int
            number of frequency points
        nmf : float or array_like
            normalization factor for spectrum. If it is array, it means we have `n` signals in parallel, the length must be equal to `n`.
        ref_spctrm : np.ndarray
            reference energy spectral density spectrum to be divided by. If it's 1d array, it means we are monitoring one signal. if it is (n, n_o) shape, it means we are monitoring several signals in parallel.
        n : int
            number of signals in parallel
        """
        if dt is None:
            dt = 1
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
        if n is None:
            n = 1
        self.n = n

        self.t_n = []  # list of integer time steps, last element is current
        self.ts = []  # list of real time points, last element is current
        self._nt = 0   # next time step

        self.f = []  # real time signal

        self.knl = np.exp(-1j * self.omg * self.dt)  # kernel

        # Initialize the Fourier transform of the field
        self.reset()
        self.F = np.zeros((n, n_o)) * 1j  # FT
        self.esd = np.zeros((n, n_o))  # energy spectral density
        self.psd = np.zeros((n, n_o))  # power spectral density
        self.flx = np.zeros((n, n_o))  # integral number flux spectral density
        self.flxr = np.zeros((n, n_o))  # number flux rate spectral density

    def rnf(self, f):
        """
        Receive new signal

        Parameters
        ----------
        f : float or array_like
            if f is array, it must be 1d array, which means it is several signals in parallel.

        """
        if not hasattr(f, '__len__'):
            f = np.array([f])
        f = np.array(f)
        if f.ndim > 1:
            raise Exception('The signal fed to the spectrum analyzer can only be 1d array.')
        self.f.append(f)
        t = self._nt * self.dt

        if self._nt > self.td:
            self.F += (self.knl ** self._nt)[None, :] * f[:, None] * self.dt
            self.esd = np.square(np.abs(self.F)) / self.nmf / self.ref_spctrm
            self.psd = self.esd / (t - self.dt * self.td)
            self.flx = self.esd / self.omg
            self.flxr = self.flx / (t - self.dt * self.td)

        self.t_n.append(self._nt)
        self.ts.append(t)
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
        n = self.n
        self.F = np.zeros((n, n_o)) * 1j
        self.esd = np.zeros((n, n_o))
        self.psd = np.zeros((n, n_o))
        self.flx = np.zeros((n, n_o))
        self.flxr = np.zeros((n, n_o))
