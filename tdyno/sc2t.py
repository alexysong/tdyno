#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song, Jul 2017

"""
# todo: anisotropic

import numpy as np
from scipy import optimize as opt
# import time
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

        # idx_source_time = np.arange(0., t_total, dt)
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
                            Controls which sides of the TFSF source exist.
                            Default to be 'all'. Can be 't', 'b', 'l', or 'r' (top, bottom, left, or right) for a uni-directional plane wave source.
        """

        self.epsi = epsi
        self.mu = mu
        self.n = np.sqrt(epsi*mu)
        # c0 = c0 / np.sqrt(epsi*mu)
        self.c0 = c0
        self.dt = dt
        self.amp = amp
        self.st = st
        self.g = g
        self.plrz = plrz
        self.whr = whr

        if plrz == 'Ez':
            if whr == 'l' or whr == 'r':
                ymin_ts = ymin_ts - st.dy
                ymax_ts = ymax_ts - st.dy
            elif whr == 't' or whr == 'b':
                xmin_ts = xmin_ts - st.dx
                xmax_ts = xmax_ts - st.dx

        self.k_unit = 1. / np.sqrt(kx**2 + ky**2) * np.array([kx, ky])

        xx_n, yy_n = [st.xx_n, st.yy_n]
        self.xx_n, self.yy_n = (xx_n, yy_n)
        xmin_ts_n, xmax_ts_n, ymin_ts_n, ymax_ts_n = [int(np.floor(cor / dcor)) for cor, dcor in zip([xmin_ts + st.dx / 1e4 - st.xmin, xmax_ts + st.dx / 1e4 - st.xmin, ymin_ts + st.dx / 1e4 - st.ymin, ymax_ts + st.dx / 1e4 - st.ymin], [st.dx, st.dx, st.dy, st.dy])]
        self.xmin_ts_n, self.xmax_ts_n, self.ymin_ts_n, self.ymax_ts_n = [xmin_ts_n, xmax_ts_n, ymin_ts_n, ymax_ts_n]

        # bulk indexing
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

        # self.i_TM_rlv_Dz = self.i_e_TM
        # self.i_TM_rlv_Bx = self.i_te_TM + self.i_be_TM
        # self.i_TM_rlv_By = self.i_le_TM + self.i_re_TM
        # self.i_TE_rlv_Bz = self.i_e_TE
        # self.i_TE_rlv_Dx = self.i_te_TE + self.i_be_TE
        # self.i_TE_rlv_Dy = self.i_le_TE + self.i_re_TE

        self.set_ndc(if_ndc, omg_ndc)

        self._calc_td()

        # edge indexing
        if self.plrz == 'Hz':
            if whr == 'all':
                self.iB_s = self.i_e_TE
                self.iD_s = self.i_e_TE
            elif whr == 't':
                self.iB_s = self.i_te_TE
                self.iD_s = self.i_te_TE
            elif whr == 'b':
                self.iB_s = self.i_be_TE
                self.iD_s = self.i_be_TE
            elif whr == 'l':
                self.iB_s = self.i_le_TE
                self.iD_s = self.i_le_TE
            elif whr == 'r':
                self.iB_s = self.i_re_TE
                self.iD_s = self.i_re_TE
        elif self.plrz == 'Ez':
            if whr == 'all':
                self.iB_s = self.i_e_TM
                self.iD_s = self.i_e_TM
            elif whr == 't':
                self.iB_s = self.i_te_TM
                self.iD_s = self.i_te_TM
            elif whr == 'b':
                self.iB_s = self.i_be_TM
                self.iD_s = self.i_be_TM
            elif whr == 'l':
                self.iB_s = self.i_le_TM
                self.iD_s = self.i_le_TM
            elif whr == 'r':
                self.iB_s = self.i_re_TM
                self.iD_s = self.i_re_TM
        else:
            self.iB_s = np.zeros([st.Ny, st.Nx], dtype=bool)
            self.iD_s = np.zeros([st.Ny, st.Nx], dtype=bool)

        # calculate source fields
        self.Dx_s_a, self.Dy_s_a, self.Dz_s_a = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]
        self.Bx_s_a, self.By_s_a, self.Bz_s_a = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]
        #
        self.Dx_s, self.Dy_s, self.Dz_s = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]
        self.Bx_s, self.By_s, self.Bz_s = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]

        # TSFS patch
        self.pchs = []
        if whr == 'all':
            xmin = xmin_ts
            ymin = ymin_ts
            width = xmax_ts - xmin_ts
            height = ymax_ts - ymin_ts
        elif whr == 'l':
            xmin = xmin_ts
            ymin = ymin_ts
            width = 0.
            height = ymax_ts - ymin_ts
        elif whr == 'b':
            xmin = xmin_ts
            ymin = ymin_ts
            width = xmax_ts - xmin_ts
            height = 0.
        elif whr == 'r':
            xmin = xmax_ts
            ymin = ymin_ts
            width = 0.
            height = ymax_ts - ymin_ts
        elif whr == 't':
            xmin = xmin_ts
            ymin = ymax_ts
            width = xmax_ts - xmin_ts
            height = 0.
        else:
            warn('TFSF location unrecognized.', UserWarning)
            xmin = ymin = width = height = None

        pch = {'shp': 'rct',
               'xy': (xmin, ymin),
               'width': width,
               'height': height
               }

        self.pchs.append(pch)

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
            elif hasattr(g, 'omg'):
                omg = g.omg
            else:
                omg = self.c0 / .2
            n = np.sqrt(self.epsi * self.mu)

            def func(x):
                return np.square(np.sin(x * self.k_unit[0] * self.st.dx / 2.) / self.st.dx) + np.square(np.sin(x * self.k_unit[1] * self.st.dy / 2.) / self.st.dy) - np.square(np.sin(omg * self.dt / 2.) / self.c0 * n / self.dt)
            k_numeric = opt.brentq(func, (omg / self.c0 * self.n * 0.5), (omg / self.c0 * self.n * 2.))
            ndc = omg / self.c0 * self.n / k_numeric
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
        xx_r = self.st.dx * (self.xx_n - (self.xmin_ts_n-1))
        yy_r = self.st.dy * (self.yy_n - (self.ymin_ts_n-1))
        d_ts = xx_r * self.k_unit[0] + yy_r * self.k_unit[1]
        d_ts = d_ts - d_ts[self.i_e_TM].min()

        # calculate time delay
        self.td = d_ts / self.c0 * self.n / ndc
        #
        self.tde_Hx = self.st.dy / 2. * self.k_unit[1] / self.c0 * self.n / ndc
        self.tde_Hy = self.st.dx / 2. * self.k_unit[0] / self.c0 * self.n / ndc
        self.tde_Ex = self.st.dx / 2. * self.k_unit[0] / self.c0 * self.n / ndc
        self.tde_Ey = self.st.dy / 2. * self.k_unit[1] / self.c0 * self.n / ndc
        self.tde_Hz = (self.st.dx / 2. * self.k_unit[0] + self.st.dy / 2. * self.k_unit[1]) / self.c0 * self.n / ndc
        #
        self.td_TE_Hz = self.td + self.tde_Hz - self.dt / 2.
        self.td_TE_Ex = self.td + self.tde_Ex
        self.td_TE_Ey = self.td + self.tde_Ey
        self.td_TM_Ez = self.td
        self.td_TM_Hx = self.td + self.tde_Hx - self.dt / 2.
        self.td_TM_Hy = self.td + self.tde_Hy - self.dt / 2.
        #
        self.td_TE_Hz_t, self.td_TE_Hz_b, self.td_TE_Hz_l, self.td_TE_Hz_r = [self.td_TE_Hz[idx] for idx in [self.i_te_TE, self.i_be_TE, self.i_le_TE, self.i_re_TE]]
        self.td_TE_Ex_t, self.td_TE_Ex_b = [self.td_TE_Ex[idx] for idx in [self.i_te_TE, self.i_be_TE]]
        self.td_TE_Ey_l, self.td_TE_Ey_r = [self.td_TE_Ey[idx] for idx in [self.i_le_TE, self.i_re_TE]]
        self.td_TM_Ez_t, self.td_TM_Ez_b, self.td_TM_Dz_l, self.td_TM_Dz_r = [self.td_TM_Ez[idx] for idx in [self.i_te_TM, self.i_be_TM, self.i_le_TM, self.i_re_TM]]
        self.td_TM_Hx_t, self.td_TM_Hx_b = [self.td_TM_Hx[idx] for idx in [self.i_te_TM, self.i_be_TM]]
        self.td_TM_Hy_l, self.td_TM_Hy_r = [self.td_TM_Hy[idx] for idx in [self.i_le_TM, self.i_re_TM]]

        # self.Dx_ic, self.Dy_ic, self.Dz_ic = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]
        # self.Bx_ic, self.By_ic, self.Bz_ic = [np.zeros([st.Ny, st.Nx]) for ii in range(3)]

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

        if self.plrz == 'Hz':

            # ========  old method ========
            # self.Hz_ic[self.i_TE_rlv_Bz] = self.amp * self.g.f(t - self.td_TE_Hz[self.i_TE_rlv_Bz])
            # self.Ex_ic[self.i_TE_rlv_Dx] = self.amp * np.sqrt(self.st.epsi_bg[0, 0] / self.st.mu_bg[2, 2]) * self.g.f(t - self.td_TE_Ex[self.i_TE_rlv_Dx]) * self.k_unit[1]
            # self.Ey_ic[self.i_TE_rlv_Dy] = self.amp * np.sqrt(self.st.epsi_bg[1, 1] / self.st.mu_bg[2, 2]) * self.g.f(t - self.td_TE_Ey[self.i_TE_rlv_Dy]) * self.k_unit[0]
            # #
            # self.Bz_s_a[self.i_te_TE] = -self.c0 * self.dt * self.Ex_ic[self.i_te_TE]
            # self.Bz_s_a[self.i_be_TE] = self.c0 * self.dt * self.Ex_ic[self.i_be_TE]
            # self.Bz_s_a[self.i_le_TE] = self.c0 * self.dt * self.Ey_ic[self.i_le_TE]  # - sign?
            # self.Bz_s_a[self.i_re_TE] = -self.c0 * self.dt * self.Ey_ic[self.i_re_TE]  # + sign?
            # #
            # self.Dx_s_a[self.i_te_TE] = self.c0 * self.dt * self.Hz_ic[self.i_te_TE]
            # self.Dx_s_a[self.i_be_TE] = -self.c0 * self.dt * self.Hz_ic[self.i_be_TE]
            # #
            # self.Dy_s_a[self.i_le_TE] = self.c0 * self.dt * self.Hz_ic[self.i_le_TE]
            # self.Dy_s_a[self.i_re_TE] = -self.c0 * self.dt * self.Hz_ic[self.i_re_TE]

            # ======== New Method  ==========
            Hz_ic = [self.amp * self.g.f(t - td) for td in [self.td_TE_Hz_t, self.td_TE_Hz_b, self.td_TE_Hz_l, self.td_TE_Hz_r]]
            Ex_ic = [self.amp * np.sqrt(self.mu / self.epsi) * self.g.f(t - td) * self.k_unit[1] for td in [self.td_TE_Ex_t, self.td_TE_Ex_b]]
            Ey_ic = [self.amp * np.sqrt(self.mu / self.epsi) * self.g.f(t - td) * self.k_unit[0] for td in [self.td_TE_Ey_l, self.td_TE_Ey_r]]

            self.Bz_s_a[self.i_te_TE] = -self.dt / self.st.dy * Ex_ic[0]
            self.Bz_s_a[self.i_be_TE] = self.dt / self.st.dy * Ex_ic[1]
            self.Bz_s_a[self.i_le_TE] = self.dt / self.st.dx * Ey_ic[0]  # - sign?
            self.Bz_s_a[self.i_re_TE] = -self.dt / self.st.dx * Ey_ic[1]  # + sign?
            #
            self.Dx_s_a[self.i_te_TE] = self.dt / self.st.dy * Hz_ic[0]
            self.Dx_s_a[self.i_be_TE] = -self.dt / self.st.dy * Hz_ic[1]
            #
            self.Dy_s_a[self.i_le_TE] = self.dt / self.st.dx * Hz_ic[2]
            self.Dy_s_a[self.i_re_TE] = -self.dt / self.st.dx * Hz_ic[3]
            # ================================

            self.Bz_s = self.Bz_s_a
            self.Dx_s = self.Dx_s_a
            self.Dy_s = self.Dy_s_a

            # t1 = time.clock()
            # t2 = time.clock()
            # print(t2-t1)

        elif self.plrz == 'Ez':

            # =======   Old Method   =========
            # self.Ez_ic[self.i_TM_rlv_Dz] = self.amp * self.g.f(t - self.td_TM_Ez[self.i_TM_rlv_Dz])
            # self.Hx_ic[self.i_TM_rlv_Bx] = self.amp * np.sqrt(self.st.mu_bg[0, 0] / self.st.epsi_bg[2, 2]) * self.g.f(t - self.td_TM_Hx[self.i_TM_rlv_Bx]) * self.k_unit[1]
            # self.Hy_ic[self.i_TM_rlv_By] = self.amp * np.sqrt(self.st.mu_bg[1, 1] / self.st.epsi_bg[2, 2]) * self.g.f(t - self.td_TM_Hy[self.i_TM_rlv_By]) * self.k_unit[0]
            #
            # self.Dz_s_a[self.i_te_TM] = self.c0 * self.dt * self.Hx_ic[self.i_te_TM]
            # self.Dz_s_a[self.i_be_TM] = -self.c0 * self.dt * self.Hx_ic[self.i_be_TM]
            # self.Dz_s_a[self.i_le_TM] = -self.c0 * self.dt * self.Hy_ic[self.i_le_TM]
            # self.Dz_s_a[self.i_re_TM] = self.c0 * self.dt * self.Hy_ic[self.i_re_TM]
            # #
            # self.Bx_s_a[self.i_te_TM] = self.c0 * self.dt * self.Ez_ic[self.i_te_TM]
            # self.Bx_s_a[self.i_be_TM] = -self.c0 * self.dt * self.Ez_ic[self.i_be_TM]
            # #
            # self.By_s_a[self.i_le_TM] = self.c0 * self.dt * self.Ez_ic[self.i_le_TM]
            # self.By_s_a[self.i_re_TM] = -self.c0 * self.dt * self.Ez_ic[self.i_re_TM]

            # ========   New Method   ========
            Ez_ic = [self.amp * self.g.f(t - td) for td in [self.td_TM_Ez_t, self.td_TM_Ez_b, self.td_TM_Dz_l, self.td_TM_Dz_r]]
            Hx_ic = [self.amp / np.sqrt(self.mu / self.epsi) * self.g.f(t - td) * self.k_unit[1] for td in [self.td_TM_Hx_t, self.td_TM_Hx_b]]
            Hy_ic = [self.amp / np.sqrt(self.mu / self.epsi) * self.g.f(t - td) * self.k_unit[0] for td in [self.td_TM_Hy_l, self.td_TM_Hy_r]]

            self.Dz_s_a[self.i_te_TM] = self.dt / self.st.dy * Hx_ic[0]
            self.Dz_s_a[self.i_be_TM] = -self.dt / self.st.dy * Hx_ic[1]
            self.Dz_s_a[self.i_le_TM] = -self.dt / self.st.dx * Hy_ic[0]  # - sign?
            self.Dz_s_a[self.i_re_TM] = self.dt / self.st.dx * Hy_ic[1]  # + sign?
            #
            self.Bx_s_a[self.i_te_TM] = self.dt / self.st.dy * Ez_ic[0]
            self.Bx_s_a[self.i_be_TM] = -self.dt / self.st.dy * Ez_ic[1]
            #
            self.By_s_a[self.i_le_TM] = self.dt / self.st.dx * Ez_ic[2]
            self.By_s_a[self.i_re_TM] = -self.dt / self.st.dx * Ez_ic[3]
            # ==============================

            self.Dz_s = self.Dz_s_a
            self.Bx_s = self.Bx_s_a
            self.By_s = self.By_s_a
