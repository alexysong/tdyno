# -*- coding: utf-8 -*-

import numpy as np
from warnings import warn
import time

from tdyno.s2t import S2T
from tdyno.mnt.sig_spct import SigSpct
from tdyno.mnt.mnt_ui import MntUI1S1S


class MntArrPoyn:
    def __init__(self, dt, td=None, omin=None, omax=None, n_o=None, nmf=None, ref_spctrm=None, n=None):
        """
        Poynting flux monitor at an array of points. Records both the real time and the frequency space Poynting vectors.

        Parameters
        ----------
        dt : float
        td : int
        omin : float
        omax : float
        n_o : int
        nmf : float
        ref_spctrm : np.ndarray
        """
        if n is None:
            n = 1
        self.n = n
        self._ss = [SigSpct(dt, td, omin, omax, n_o, nmf, ref_spctrm, n=n) for i in range(6)]  # length is 6, corresponding to 3 E and 3 H components
        self._s = []
        self._S = None

    @property
    def s(self):
        """
        Real time Poynting vectors.

        Returns
        -------
        s : list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        """
        return self._s

    @property
    def S(self):
        """
        Poynting flux in frequency space on the point.

        Returns
        -------
        S : tuple[np.ndarray, np.ndarray, np.ndarray]
            each is a complex 1d array
        """
        return self._S

    @property
    def omg(self):
        """
        frequencies

        Returns
        -------
        omg : np.ndarray, 1d array
        """
        return self._ss[0].omg

    @property
    def t(self):
        """
        List of real time points.

        Returns
        -------
        t : list[float]
        """
        t = None
        for i in range(6):
            if self._ss[i].ts:
                t = self._ss[i].ts
                break
        return t

    def rnf(self, ex=None, ey=None, ez=None, hx=None, hy=None, hz=None):
        for i, f in enumerate([ex, ey, ez, hx, hy, hz]):
            if f is not None:
                self._ss[i].rnf(f)

        _ex, _ey, _ez, _hx, _hy, _hz = [0. if (f is None) else f for f in [ex, ey, ez, hx, hy, hz]]

        sx = (_ey * _hz - _ez * _hy)
        sy = (_ez * _hx - _ex * _hz)
        sz = (_ex * _hy - _ey * _hx)
        self._s.append((sx, sy, sz))

        Ex, Ey, Ez, Hx, Hy, Hz = [self._ss[i].F if (f is not None) else 0. for i, f in enumerate([ex, ey, ez, hx, hy, hz])]
        Exc, Eyc, Ezc, Hxc, Hyc, Hzc = [f.conjugate() for f in [Ex, Ey, Ez, Hx, Hy, Hz]]
        sx = 1./2 * (Ey * Hzc - Ez * Hyc).real
        sy = 1./2 * (Ez * Hxc - Ex * Hzc).real
        sz = 1./2 * (Ex * Hyc - Ey * Hxc).real
        self._S = (sx, sy, sz)

    def reset(self):
        self._s = []
        self._S = None
        for s in self._ss:
            s.reset()


class Mnt2DSqPoyn:
    # todo: take modal profile, get Poynting in specific mode
    # todo: get Poynting flux rate in moving time-window

    def __init__(self,
                 st,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 dt=None, td=None,
                 omin=None, omax=None, n_o=None,
                 nmf=None, ref_spctrm=None,
                 whr='all', plrz=None):
        """
        Poynting flux through a square in 2D, from inside to outside.

        This monitor records the total real-time flux through the boundary of the box. It also calculates the Poynting flux spectrum.

        You can feed new field data into the monitor through `rnf()`. If polarization is "Ez", the arguments to `rnf()` is understood as Ez, Hx, Hy. For "Hz" polarization, it is understood as Hz, Ex, Ey.

        Parameters
        ----------
        st: S2T
        xmin : float
        xmax : float
        ymin : float
        ymax : float
        dt : float
        td : float
        omin : float
        omax : float
        n_o : int
        nmf : float
        show : str
        ref_spctrm : np.ndarray
        whr : str
            Controls which sides of the TFSF source exist.
            
            Default to be 'all'. Can be 't', 'b', 'l', or 'r' (top, bottom, left, or right).
        """

        xmin, xmax, ymin, ymax = [smm if mm is None else mm for (mm, smm) in zip([xmin, xmax, ymin, ymax], [st.xmin, st.xmax, st.ymin, st.ymax])]

        # same as TFSF indexing
        xx_n, yy_n = [st.xx_n, st.yy_n]
        self.xx_n, self.yy_n = (xx_n, yy_n)
        xmin_n, xmax_n, ymin_n, ymax_n = [int(np.floor(cor / dcor)) for cor, dcor in zip([xmin + st.dx / 1e4 - st.xmin, xmax + st.dx / 1e4 - st.xmin, ymin + st.dx / 1e4 - st.ymin, ymax + st.dx / 1e4 - st.ymin], [st.dx, st.dx, st.dy, st.dy])]
        if xmin_n < xx_n.min():
            warn('Source left boundary outside of the solving space. It has been reset to the left edge of the solving space.')
            xmin_n = xx_n.min()
        if xmax_n > xx_n.max():
            warn('Source right boundary outside of the solving space. It has been reset to the right edge of the solving space.')
            xmax_n = xx_n.max()
        if ymin_n < yy_n.min():
            warn('Source bottom boundary outside of the solving space. It has been reset to the bottom edge of the solving space.')
            ymin_n = yy_n.min()
        if ymax_n > yy_n.max():
            warn('Source top boundary outside of the solving space. It has been reset to the top edge of the solving space.')
            ymax_n = yy_n.max()
        self.xmin_n, self.xmax_n, self.ymin_n, self.ymax_n = [xmin_n, xmax_n, ymin_n, ymax_n]
        # bulk indexing
        self.i_te_TM = (yy_n == ymax_n) * (xx_n <= xmax_n) * (xx_n > xmin_n)
        self.i_be_TM = (yy_n == ymin_n) * (xx_n <= xmax_n) * (xx_n > xmin_n)
        self.i_le_TM = (xx_n == xmin_n) * (yy_n <= ymax_n) * (yy_n > ymin_n)
        self.i_re_TM = (xx_n == xmax_n) * (yy_n <= ymax_n) * (yy_n > ymin_n)
        #
        self.i_te_TE = (yy_n == ymax_n) * (xx_n < xmax_n) * (xx_n >= xmin_n)
        self.i_be_TE = (yy_n == ymin_n) * (xx_n < xmax_n) * (xx_n >= xmin_n)
        self.i_le_TE = (xx_n == xmin_n) * (yy_n < ymax_n) * (yy_n >= ymin_n)
        self.i_re_TE = (xx_n == xmax_n) * (yy_n < ymax_n) * (yy_n >= ymin_n)
        #
        self.i_e_TM = self.i_re_TM + self.i_le_TM + self.i_te_TM + self.i_be_TM
        self.i_e_TE = self.i_re_TE + self.i_le_TE + self.i_te_TE + self.i_be_TE
        # monitor edge indexing
        if plrz is None:
            plrz = 'Ez'
        self.plrz = plrz
        self.i = []
        self.n = []  # norm direction
        self.d = []  # dx or dy
        if self.plrz == 'Hz':
            if whr == 'all':
                self.i = [self.i_te_TE, self.i_be_TE, self.i_le_TE, self.i_re_TE]
                self.n = [(0, 1), (0, -1), (-1, 0), (0, 1)]
                self.d = [st.dx, st.dx, st.dy, st.dy]
            else:
                for w in whr:
                    if w == 't':
                        self.i.append(self.i_te_TE)
                        self.n.append((0, 1))
                        self.d.append(st.dx)
                    elif w == 'b':
                        self.i.append(self.i_be_TE)
                        self.n.append((0, -1))
                        self.d.append(st.dx)
                    elif w == 'l':
                        self.i.append(self.i_le_TE)
                        self.n.append((-1, 0))
                        self.d.append(st.dy)
                    elif w == 'r':
                        self.i.append(self.i_re_TE)
                        self.n.append((0, 1))
                        self.d.append(st.dy)
        elif self.plrz == 'Ez':
            if whr == 'all':
                self.i = [self.i_te_TM, self.i_be_TM, self.i_le_TM, self.i_re_TM]
                self.n = [(0, 1), (0, -1), (-1, 0), (0, 1)]
            else:
                for w in whr:
                    if w == 't':
                        self.i.append(self.i_te_TM)
                        self.n.append((0, 1))
                        self.d.append(st.dx)
                    elif w == 'b':
                        self.i.append(self.i_be_TM)
                        self.n.append((0, -1))
                        self.d.append(st.dx)
                    elif w == 'l':
                        self.i.append(self.i_le_TM)
                        self.n.append((-1, 0))
                        self.d.append(st.dy)
                    elif w == 'r':
                        self.i.append(self.i_re_TM)
                        self.n.append((0, 1))
                        self.d.append(st.dy)
        else:
            warn('Polarization not recognized. No monitor value recorded.')
            self.i = np.zeros([st.Ny, st.Nx], dtype=bool)

        self.mpps = [MntArrPoyn(dt, td, omin, omax, n_o, nmf, ref_spctrm, n=np.where(j)[0].size) for j in self.i]  # list, length equal to number of edges included, each element corresponds to one edge
        self.s = []
        self.S = None

        # patch
        self.pchs = []
        if whr == 'all':
            xmi = xmin
            ymi = ymin
            width = xmax - xmin
            height = ymax - ymin
            pch = {'shp': 'rct',
                   'xy': (xmi, ymi),
                   'width': width,
                   'height': height
                   }
            self.pchs.append(pch)
        else:
            for w in whr:
                if w == 'l':
                    xmi = xmin
                    ymi = ymin
                    width = 0.
                    height = ymax - ymin
                elif w == 'b':
                    xmi = xmin
                    ymi = ymin
                    width = xmax - xmin
                    height = 0.
                elif w == 'r':
                    xmi = xmax
                    ymi = ymin
                    width = 0.
                    height = ymax - ymin
                elif w == 't':
                    xmi = xmin
                    ymi = ymax
                    width = xmax - xmin
                    height = 0.
                else:
                    warn('Monitor location unrecognized.', UserWarning)
                    xmi = ymi = width = height = None

                pch = {'shp': 'rct',
                       'xy': (xmi, ymi),
                       'width': width,
                       'height': height
                       }

                self.pchs.append(pch)

    @property
    def omg(self):
        return self.mpps[0].omg

    @property
    def t(self):
        return self.mpps[0].t

    def rnf(self, fz, fx, fy):
        """
        Receive new fields

        Parameters
        ----------
        fx : np.ndarray
        fy : np.ndarray
        fz : np.ndarray
            all of these inputs have shape (Ny, Nx)
        """

        # pick out the field values at the monitor edges
        _fx, _fy, _fz, = [[f[j] for j in self.i] for f in [fx, fy, fz]]  # each is list of 1D arrays, each array corresponds to one edge

        s = 0.
        S = 0.
        for m, x, y, z, n in zip(self.mpps, _fx, _fy, _fz, self.n):
            if self.plrz == "Ez":
                m.rnf(hx=x, hy=y, ez=z)
            elif self.plrz == 'Hz':
                m.rnf(ex=x, ey=y, hz=z)
            else:
                warn("Polarization not recognized. Default to 'Hz'.")
                m.rnf(ex=x, ey=y, hz=z)
            for _s, _S, _n, _d in zip(m.s[-1][:2], m.S[:2], self.n, self.d):
                if _n != 0:
                    s += np.sum(_s) * _d
                    S += np.sum(_S, axis=0) * _d

        self.s.append(s)
        self.S = S

    def cf(self):
        self.s = []
        self.S = None
        for m in self.mpps:
            m.reset()


class Mnt2DSqPoynU:
    def __init__(self, *args, **kwargs):
        """ Wrapper class, `Mnt2DSqPoyn` with `MntUI1S1S`. """
        self.mnt: Mnt2DSqPoyn = Mnt2DSqPoyn(*args, **kwargs)
        self.mui = MntUI1S1S(signal_y_label="Poynting flux (real time)", spectrum_y_label="Poynting flux spectrum", Sx=self.mnt.omg)

    @property
    def pchs(self):
        return self.mnt.pchs

    def rnf(self, *args, **kwargs):
        self.mnt.rnf(*args, **kwargs)

    def up(self):
        self.mui.up(sx=self.mnt.t,
                    sy=self.mnt.s,
                    Sy=self.mnt.S)

    def cf(self):
        self.mnt.cf()
