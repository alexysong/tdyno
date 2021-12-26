# -*- coding: utf-8 -*-

# todo: anisotropic

import numpy as np
from scipy import optimize as opt
import time
from warnings import warn


class PS2:

    def __init__(self, x_s, y_s, amp, g, st, plrz='Hz'):

        """
        Generate a point source at a given location with a given temporal profile.

        Parameters
        ----------
        x_s,y_s :   float
                    x and y location of source
        amp     :   float
                    amplitude of the point source
        g       :   Union[Hm, HP, Gsn]
                    Temporal profile
        st      :   S2T
                    structure
        plrz    :   str
                    polarization

        """

        self.amp = amp
        self.g = g
        self.polarization = plrz

        if x_s > st.xmax:
            x_s = st.xmax
        if x_s < st.xmin:
            x_s = st.min
        if y_s > st.ymax:
            y_s = st.ymax
        if y_s < st.ymin:
            y_s = st.ymin
        x_s_n = int((x_s - st.xmin) / st.dx)
        y_s_n = int((y_s - st.ymin) / st.dy)

        self.iB_s = (st.xx_n == x_s_n) * (st.yy_n == y_s_n)
        self.iD_s = (st.xx_n == x_s_n) * (st.yy_n == y_s_n)

        self.Bx_s_a, self.By_s_a, self.Bz_s_a = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]
        self.Dx_s_a, self.Dy_s_a, self.Dz_s_a = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]

        self.Bx_s, self.By_s, self.Bz_s = [f for f in [self.Bx_s_a, self.By_s_a, self.Bz_s_a]]
        self.Dx_s, self.Dy_s, self.Dz_s = [f for f in [self.Dx_s_a, self.Dy_s_a, self.Dz_s_a]]

    def us(self, t):

        """
        update source

        Parameters
        ----------
        t   :   float
                time, not in time step, but in physical time.

        """

        if self.polarization == 'Hz':
            self.Bz_s[self.iB_s] = self.amp * self.g.f(t)
        elif self.polarization == 'Ez':
            self.Dz_s[self.iD_s] = self.amp * self.g.f(t)


class TSS2:

    def __init__(self,
                 xmin_ts, xmax_ts, ymin_ts, ymax_ts,
                 kx, ky, amp, g,
                 st, c0, dt, plrz='Ez', if_ndc=False, omg_ndc=None,
                 epsi=1., mu=1.,
                 whr='all'):

        """
        Generate TF/SF plane wave source with given wave vector with supplied temporal profile in the square defined by {xmin, xmax, ymin, ymax}_ts.

        Parameters
        ----------
        xmin_ts, xmax_ts, ymin_ts, ymax_ts  :   float
                                                four corners of TFSF source box
        kx, ky          :   float
        amp             :   float
                            amplitude of the incident wave
        st              :   S2T
        g               :   Union[Hm, HP, Gsn]
                            source temporal profile
        plrz            :   str
                            polarization
        if_ndc          :   bool
                            if numeric dispersion compensation
        omg_ndc         :   float
                            frequency for ndc
        epsi, mu        :   float
                            background relative permittivity and permeability, if the src is in a material rather than in vacuum
        whr             :   str
                            {'all', combinations of 't', 'b', 'l', 'r'}

                            Controls which sides of the TFSF source exist.

                            Default to be 'all', i.e. TFSF source in a rectangular region.

                            Can choose any combination of 't', 'b', 'l', 'r' for top, bottom, left and right.

                            If choose any ONE of the sides, it becomes a uni-directional plane-wave source.
        """

        self._epsi = epsi
        self._mu = mu
        self._n = np.sqrt(epsi * mu)
        self._c0 = c0
        self._dt = dt
        self._amp = amp
        self._st = st
        self._g = g
        self._plrz = plrz
        self._whr = whr

        if plrz == 'Ez':
            ymin_ts -= st.dy
            ymax_ts -= st.dy
            xmin_ts -= st.dx
            xmax_ts -= st.dx

        self.kux, self.kuy = 1. / np.sqrt(kx**2 + ky**2) * np.array([kx, ky])

        xx_n, yy_n = [st.xx_n, st.yy_n]
        self.xx_n, self.yy_n = (xx_n, yy_n)
        xmin_ts_n, xmax_ts_n, ymin_ts_n, ymax_ts_n = [int(np.floor(cor / dcor)) for cor, dcor in zip([xmin_ts + st.dx / 1e4 - st.xmin, xmax_ts + st.dx / 1e4 - st.xmin, ymin_ts + st.dx / 1e4 - st.ymin, ymax_ts + st.dx / 1e4 - st.ymin], [st.dx, st.dx, st.dy, st.dy])]
        if xmin_ts_n < xx_n.min():
            warn('Source left boundary outside of the solving space. It has been reset to the left edge of the solving space.')
            xmin_ts_n = xx_n.min()
        if xmax_ts_n > xx_n.max():
            warn('Source right boundary outside of the solving space. It has been reset to the right edge of the solving space.')
            xmax_ts_n = xx_n.max()
        if ymin_ts_n < yy_n.min():
            warn('Source bottom boundary outside of the solving space. It has been reset to the bottom edge of the solving space.')
            ymin_ts_n = yy_n.min()
        if ymax_ts_n > yy_n.max():
            warn('Source top boundary outside of the solving space. It has been reset to the top edge of the solving space.')
            ymax_ts_n = yy_n.max()
        self.xmin_ts_n, self.xmax_ts_n, self.ymin_ts_n, self.ymax_ts_n = [xmin_ts_n, xmax_ts_n, ymin_ts_n, ymax_ts_n]

        # indexing of the edge in the bulk
        # all 2d arrays
        self.i_te_TM = (yy_n == ymax_ts_n) * (xx_n <= xmax_ts_n) * (xx_n > xmin_ts_n)
        self.i_be_TM = (yy_n == ymin_ts_n) * (xx_n <= xmax_ts_n) * (xx_n > xmin_ts_n)
        self.i_le_TM = (xx_n == xmin_ts_n) * (yy_n <= ymax_ts_n) * (yy_n > ymin_ts_n)
        self.i_re_TM = (xx_n == xmax_ts_n) * (yy_n <= ymax_ts_n) * (yy_n > ymin_ts_n)
        #
        self.i_te_TE = (yy_n == ymax_ts_n) * (xx_n < xmax_ts_n) * (xx_n >= xmin_ts_n)
        self.i_be_TE = (yy_n == ymin_ts_n) * (xx_n < xmax_ts_n) * (xx_n >= xmin_ts_n)
        self.i_le_TE = (xx_n == xmin_ts_n) * (yy_n < ymax_ts_n) * (yy_n >= ymin_ts_n)
        self.i_re_TE = (xx_n == xmax_ts_n) * (yy_n < ymax_ts_n) * (yy_n >= ymin_ts_n)
        #
        self.i_e_TM = self.i_re_TM + self.i_le_TM + self.i_te_TM + self.i_be_TM
        self.i_e_TE = self.i_re_TE + self.i_le_TE + self.i_te_TE + self.i_be_TE
        #
        self.i_co_tr_TM = self.i_te_TM * self.i_re_TM
        self.i_co_bl_TE = self.i_le_TE * self.i_be_TE

        # edge indexing
        self.tblr = [False] * 4
        self.iB_s = np.zeros([st.Ny, st.Nx], dtype=bool)
        # self.iD_s = np.zeros([_st.Ny, _st.Nx], dtype=bool)
        if whr == "all":
            whr = "tblr"
        if self._plrz == 'Hz':
            for w in whr:
                for i, _w, idx in zip(range(4), "tblr", [self.i_te_TE, self.i_be_TE, self.i_le_TE, self.i_re_TE]):
                    if w == _w:
                        self.tblr[i] = True
                        self.iB_s += idx
        elif self._plrz == 'Ez':
            for w in whr:
                for i, _w, idx in zip(range(4), "tblr", [self.i_te_TM, self.i_be_TM, self.i_le_TM, self.i_re_TM]):
                    if w == _w:
                        self.tblr[i] = True
                        self.iB_s += idx
        else:
            raise Exception('Polarization not recognized. No TF/SF source was set.')
        self.iD_s = self.iB_s

        #
        self.Dx_s, self.Dy_s, self.Dz_s = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]
        self.Bx_s, self.By_s, self.Bz_s = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]

        # TSFS patch
        self.pchs = []
        if whr == 'all':
            xmi = xmin_ts
            ymi = ymin_ts
            width = xmax_ts - xmin_ts
            height = ymax_ts - ymin_ts
            pch = {'shp': 'rct',
                   'xy': (xmi, ymi),
                   'width': width,
                   'height': height
                   }
            self.pchs.append(pch)
        else:
            for w in whr:
                if w == 'l':
                    xmi = xmin_ts
                    ymi = ymin_ts
                    width = 0.
                    height = ymax_ts - ymin_ts
                elif w == 'b':
                    xmi = xmin_ts
                    ymi = ymin_ts
                    width = xmax_ts - xmin_ts
                    height = 0.
                elif w == 'r':
                    xmi = xmax_ts
                    ymi = ymin_ts
                    width = 0.
                    height = ymax_ts - ymin_ts
                elif w == 't':
                    xmi = xmin_ts
                    ymi = ymax_ts
                    width = xmax_ts - xmin_ts
                    height = 0.
                else:
                    warn('TF/SF source location unrecognized.', UserWarning)
                    xmi = ymi = width = height = None

                pch = {'shp': 'rct',
                       'xy': (xmi, ymi),
                       'width': width,
                       'height': height
                       }

                self.pchs.append(pch)

        self.set_ndc(if_ndc, omg_ndc)

        # self._calc_td()  # called in `self.set_ndc`

    def set_ndc(self, if_ndc, omg_ndc):
        """
        set numerical dispersion correction

        Parameters
        ----------
        if_ndc
        omg_ndc

        Returns
        -------

        """
        # numeric dispersion
        if if_ndc:
            if omg_ndc:
                omg = omg_ndc
            elif hasattr(self._g, 'omg'):
                omg = self._g.omg
            else:
                omg = self._c0 / .2
            n = np.sqrt(self._epsi * self._mu)

            def func(x):
                return np.square(np.sin(x * self.kux * self._st.dx / 2.) / self._st.dx) + np.square(np.sin(x * self.kuy * self._st.dy / 2.) / self._st.dy) - np.square(np.sin(omg * self._dt / 2.) / self._c0 * n / self._dt)
            k_numeric = opt.brentq(func, (omg / self._c0 * n * 0.5), (omg / self._c0 * n * 2.))
            ndc = omg / self._c0 * n / k_numeric
            # print(ndc)
        else:
            ndc = 1.
        self.ndc = ndc

        self._calc_td()

    def _calc_td(self):
        """
        calculate relative time delays

        Returns
        -------

        """
        ndc = self.ndc

        # calculate relative distances
        xx_r = self._st.dx * (self.xx_n - self.xmin_ts_n)
        yy_r = self._st.dy * (self.yy_n - self.ymin_ts_n)
        d_ts = xx_r * self.kux + yy_r * self.kuy
        d_ts -= d_ts[self.i_e_TM].min()

        # calculate time delay
        self.td = d_ts / self._c0 * self._n / ndc
        #
        self.tde_Hx = self._st.dy / 2. * self.kuy / self._c0 * self._n / ndc
        self.tde_Hy = self._st.dx / 2. * self.kux / self._c0 * self._n / ndc
        self.tde_Ex = self._st.dx / 2. * self.kux / self._c0 * self._n / ndc
        self.tde_Ey = self._st.dy / 2. * self.kuy / self._c0 * self._n / ndc
        self.tde_Hz = (self._st.dx / 2. * self.kux + self._st.dy / 2. * self.kuy) / self._c0 * self._n / ndc
        #
        self.td_TE_Hz = self.td + self.tde_Hz - self._dt / 2.
        self.td_TE_Ex = self.td + self.tde_Ex
        self.td_TE_Ey = self.td + self.tde_Ey
        self.td_TM_Ez = self.td
        self.td_TM_Hx = self.td + self.tde_Hx - self._dt / 2.
        self.td_TM_Hy = self.td + self.tde_Hy - self._dt / 2.
        #
        self.td_TE_Hz_t, self.td_TE_Hz_b, self.td_TE_Hz_l, self.td_TE_Hz_r = [self.td_TE_Hz[idx] for idx in [self.i_te_TE, self.i_be_TE, self.i_le_TE, self.i_re_TE]]
        self.td_TE_Ex_t, self.td_TE_Ex_b = [self.td_TE_Ex[idx] for idx in [self.i_te_TE, self.i_be_TE]]
        self.td_TE_Ey_l, self.td_TE_Ey_r = [self.td_TE_Ey[idx] for idx in [self.i_le_TE, self.i_re_TE]]
        self.td_TM_Ez_t, self.td_TM_Ez_b, self.td_TM_Ez_l, self.td_TM_Ez_r = [self.td_TM_Ez[idx] for idx in [self.i_te_TM, self.i_be_TM, self.i_le_TM, self.i_re_TM]]
        self.td_TM_Hx_t, self.td_TM_Hx_b = [self.td_TM_Hx[idx] for idx in [self.i_te_TM, self.i_be_TM]]
        self.td_TM_Hy_l, self.td_TM_Hy_r = [self.td_TM_Hy[idx] for idx in [self.i_le_TM, self.i_re_TM]]

        # self.Dx_ic, self.Dy_ic, self.Dz_ic = [np.zeros([_st.Ny, _st.Nx]) for ii in range(3)]
        # self.Bx_ic, self.By_ic, self.Bz_ic = [np.zeros([_st.Ny, _st.Nx]) for ii in range(3)]

    def us(self, t):

        """
        update source fields

        Parameters
        ----------
        t   :   float
                current time

        Returns
        -------

        """

        if self._plrz == 'Hz':
            # ======  method 3  ======
            if self.tblr[0]:
                Hz_ic = self._amp * self._g.f(t - self.td_TE_Hz_t)
                Ex_ic = self._amp * np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TE_Ex_t) * self.kuy
                self.Bz_s[self.i_te_TE] = - self._dt / self._st.dy * Ex_ic
                self.Dx_s[self.i_te_TE] = self._dt / self._st.dy * Hz_ic
            if self.tblr[1]:
                Hz_ic = self._amp * self._g.f(t - self.td_TE_Hz_b)
                Ex_ic = self._amp * np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TE_Ex_b) * self.kuy
                Ex_ic_bl_co = Ex_ic[0]
                self.Bz_s[self.i_be_TE] = self._dt / self._st.dy * Ex_ic
                self.Dx_s[self.i_be_TE] = - self._dt / self._st.dy * Hz_ic
            if self.tblr[2]:
                Hz_ic = self._amp * self._g.f(t - self.td_TE_Hz_l)
                Ey_ic = self._amp * np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TE_Ey_l) * self.kux
                self.Bz_s[self.i_le_TE] = self._dt / self._st.dx * Ey_ic  # - sign?
                self.Dy_s[self.i_le_TE] = self._dt / self._st.dx * Hz_ic
                if self.tblr[1]:
                    self.Bz_s[self.i_co_bl_TE] += self._dt / self._st.dy * Ex_ic_bl_co
            if self.tblr[3]:
                Hz_ic = self._amp * self._g.f(t - self.td_TE_Hz_r)
                Ey_ic = self._amp * np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TE_Ey_r) * self.kux
                self.Bz_s[self.i_re_TE] = - self._dt / self._st.dx * Ey_ic  # - sign?
                self.Dy_s[self.i_re_TE] = - self._dt / self._st.dx * Hz_ic

            # ======  method 4  ======  todo: why doesn't this work?
            # # try directly calculate B and D fields
            # Hz_ic = [1.2 * self._amp * self._g.f(t - td) for td in [self.td_TE_Hz_t, self.td_TE_Hz_b, self.td_TE_Hz_l, self.td_TE_Hz_r]]
            # Ex_ic = [self._amp * np.sqrt(self._mu / self._epsi) * self._g.f(t - td) * self.kuy for td in [self.td_TE_Ex_t, self.td_TE_Ex_b]]
            # Ey_ic = [self._amp * np.sqrt(self._mu / self._epsi) * self._g.f(t - td) * self.kux for td in [self.td_TE_Ey_l, self.td_TE_Ey_r]]
            # self.Bz_s[self.i_te_TE] = - self._mu * Hz_ic[0]
            # self.Bz_s[self.i_be_TE] = - self._mu * Hz_ic[1]
            # self.Bz_s[self.i_le_TE] = - self._mu * Hz_ic[2]
            # self.Bz_s[self.i_re_TE] = - self._mu * Hz_ic[3]
            # #
            # self.Dx_s[self.i_te_TE] = - self._epsi * Ex_ic[0]
            # self.Dx_s[self.i_be_TE] = - self._epsi * Ex_ic[1]
            # #
            # self.Dy_s[self.i_le_TE] = - self._epsi * Ey_ic[0]
            # self.Dy_s[self.i_re_TE] = - self._epsi * Ey_ic[1]

        elif self._plrz == 'Ez':
            # ======  method 3  ======
            # t1 = time.perf_counter()
            if self.tblr[0]:
                Ez_ic = self._amp * self._g.f(t - self.td_TM_Ez_t)
                Hx_ic = self._amp / np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TM_Hx_t) * self.kuy
                Hx_ic_tr_co = Hx_ic[-1]
                self.Dz_s[self.i_te_TM] = self._dt / self._st.dy * Hx_ic
                self.Bx_s[self.i_te_TM] = self._dt / self._st.dy * Ez_ic
            if self.tblr[1]:
                Ez_ic = self._amp * self._g.f(t - self.td_TM_Ez_b)
                Hx_ic = self._amp / np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TM_Hx_b) * self.kuy
                self.Dz_s[self.i_be_TM] = -self._dt / self._st.dy * Hx_ic
                self.Bx_s[self.i_be_TM] = -self._dt / self._st.dy * Ez_ic
            if self.tblr[2]:
                Ez_ic = self._amp * self._g.f(t - self.td_TM_Ez_l)
                Hy_ic = self._amp / np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TM_Hy_l) * self.kux
                self.Dz_s[self.i_le_TM] = -self._dt / self._st.dx * Hy_ic  # - sign?
                self.By_s[self.i_le_TM] = self._dt / self._st.dx * Ez_ic
            if self.tblr[3]:
                Ez_ic = self._amp * self._g.f(t - self.td_TM_Ez_r)
                Hy_ic = self._amp / np.sqrt(self._mu / self._epsi) * self._g.f(t - self.td_TM_Hy_r) * self.kux
                self.Dz_s[self.i_re_TM] = self._dt / self._st.dx * Hy_ic  # - sign?
                self.By_s[self.i_re_TM] = -self._dt / self._st.dx * Ez_ic
                if self.tblr[0]:
                    self.Dz_s[self.i_co_tr_TM] += self._dt / self._st.dy * Hx_ic_tr_co
            # t2 = time.perf_counter()
            # print("{:e}".format(t2-t1))
