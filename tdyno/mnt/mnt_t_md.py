# -*- coding: utf-8 -*-

import numpy as np
from warnings import warn
# import time

from tdyno.s2t import S2T
from tdyno.mnt.mnt_t_md_ui import MntMdUI


class MntWgMdU:
    def __init__(self, *args, **kwargs):
        """
        wrapper class, `MntWgMd` with `MntMdUI`
        """
        self.mnt: MntWgMd = MntWgMd(*args, **kwargs)
        self.mui: MntMdUI = MntMdUI(xi=self.mnt.xi, epsi=self.mnt.epsi, mu=self.mnt.mu, f=self.mnt.f, plrz=self.mnt.plrz)

    @property
    def pchs(self):
        return self.mnt.pchs

    def rnf(self, *args, **kwargs):
        self.mnt.rnf(*args, **kwargs)

    def up(self):
        self.mui.up(f=self.mnt.f)

    def cf(self):
        self.mnt.cf()


class MntWgMd:

    def __init__(self,
                 st,
                 x=None, y=None,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 plrz=None):
        """
        Monitor waveguide mode.

        At any time, record the instantaneous Ez or Hz values, the coordinate transverse to the waveguide, permittivity and permeability values at these coordinates.

        Can feed new field data into the monitor through `rnf()`. If polarization is "Ez", the arguments to `rnf()` is understood as Ez, Hx, Hy. For "Hz" polarization, it is understood as Hz, Ex, Ey.

        Only works for waveguides with simple dielectric materials.

        Parameters
        ----------
        st: S2T
        x : float
        y : float
        xmin : float
        xmax : float
        ymin : float
        ymax : float
        plrz : str

        Notes
        -----
        If `x` is specified, then need `ymin` and `ymax`. It set up an interface in y direction at `x`.

        If `y` is specified, then need `xmin` and `xmax`. It set up an interface in x direction at `y`.

        If both `x` and `y` are given, `y` is ignored.
        """

        self._st = st
        if (plrz != 'Ez') and (plrz != 'Hz'):
            warn("polarization not recognized, default to 'Hz' (TE).", UserWarning)
            plrz = 'Hz'
        self.plrz = plrz

        xmin, xmax, ymin, ymax = [smm if mm is None else mm
                                  for (mm, smm) in zip([xmin, xmax, ymin, ymax],
                                                       [st.xmin, st.xmax, st.ymin, st.ymax])]

        xx_n, yy_n = [st.xx_n, st.yy_n]
        self._xx_n, self._yy_n = (xx_n, yy_n)
        xmin_n, xmax_n, ymin_n, ymax_n = [int(np.floor(cor / dcor)) for cor, dcor in
                                          zip([xmin + st.dx / 1e4 - st.xmin, xmax + st.dx / 1e4 - st.xmin, ymin + st.dx / 1e4 - st.ymin, ymax + st.dx / 1e4 - st.ymin],
                                              [st.dx, st.dx, st.dy, st.dy])]

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
        self._xmin_n, self._xmax_n, self._ymin_n, self._ymax_n = [xmin_n, xmax_n, ymin_n, ymax_n]

        # indexing of edge in the bulk
        if x is not None:
            x_n = int(np.floor(((x + 1e-10 - st.xmin) / st.dx)))
            self._i_e = (xx_n == x_n) * (yy_n < ymax_n) * (yy_n >= ymin_n)

            xmi = x
            ymi = ymin
            width = 0.
            height = ymax - ymin

            if y is not None:
                warn("Both `x` and `y` are specified. y is ignored, x is retained.", UserWarning)
        else:
            if y is None:
                warn("Neither `x` or `y` is specified. Assume `y = 0`.")
                y = 0.

            y_n = int(np.floor(((y + 1e-10 - st.ymin) / st.dy)))
            self._i_e = (yy_n == y_n) * (xx_n < xmax_n) * (xx_n >= xmin_n)

            xmi = xmin
            ymi = y
            width = xmax - xmin
            height = 0.

        pch = {'shp': 'rct',
               'xy': (xmi, ymi),
               'width': width,
               'height': height
               }
        self.pchs = []
        self.pchs.append(pch)

        self.xi = self._st.yy[self._i_e]
        if self.plrz == "Hz":
            self.xi += self._st.dy / 2.

        embg = [em[[0, 1], [0, 1]] for em in [self._st.epsi_bg, self._st.mu_bg]]
        em = [a[None, None, :] * np.ones((self._st.Ny, self._st.Nx))[:, :, None] for a in embg]
        for bxsp in self._st.bxs_sp:
            for a, b in zip(em, [bxsp.mtr.epsi, bxsp.mtr.mu]):
                a[bxsp.gmt.iib, :] = b[[0, 1], [0, 1]]
        self.epsi, self.mu = [a[self._i_e, :] for a in em]

        self._fz = None
        self.f = None

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

        self._fz = fz
        # self._fx = fx
        # self._fy = fy

        self.f = fz[self._i_e]

    def cf(self):
        self._fz = None
        self.f = None
