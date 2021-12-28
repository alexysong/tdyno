# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
# import time
from warnings import warn
from tdyno.s2t import S2T


class TSSMD2:

    def __init__(self,
                 xmin_ts, xmax_ts, ymin_ts, ymax_ts,
                 kx, ky, amp, g,
                 xi, f, epsi, mu, omg,
                 st, c0, dt,
                 xi0=None, reverse_direction=False,
                 plrz='Ez', if_ndc=False, omg_ndc=None,
                 whr='all',
                 use_epsi_mu_in_structure=False):
        """
        Generate TF/SF plane wave source with given wave vector with supplied temporal profile in the square defined by {xmin, xmax, ymin, ymax}_ts.

        Parameters
        ----------
        xmin_ts, xmax_ts, ymin_ts, ymax_ts :    float
                                                four corners of TFSF source box
        kx, ky          :   float
                            (kx, ky) determine the wave propagation constant. Both the magnitude and the direction of (kx, ky) matters. This is different from `add_tfsf_source`.
        amp             :   float
                            amplitude of the incident wave
        st              :   S2T
        g               :   Hm or Gsn or HP
                            source temporal profile
        xi              :   array_like
                            1d array, spatial points where f is defined, coordinate transverse to the waveguide. See notes on `f` below.
        f               :   array_like
                            1d array value of f, the waveguide mode.

                            For Ez mode, `f` is Ez. For Hz mode, `f` is Hz.

                            `xi` and `f` can be analytically calculated or numerically simulated.

                            `xi` are the actual physical points where `f` is defined.
                            For Ez mode, `xi` is where Ez is defined. For Hz mode, `xi` is where Hz is defined.
                            In fact `xi` does not know about Yee cells in general (for example, analytically simulated).

                            If `xi` and `f` were simulation results of `tdyno` with a waveguide in the x direction, then for `Ez` polarization `xi` are the Yee cell corners i.e. grid points in y, while for `Hz` polarization `xi` are the half grid points in y in each Yee cell.

        epsi            :   array_like
                            2d array. The permittivity profile along the transverse direction of the waveguide.

                            `epsi[i]` defines the values at index i. It has 2 elements, `epsi[i][0]` is the component  along the waveguide direction, `epsi[i][1]` is transverse to the waveguide.

                            See notes in `mu` below.

        mu              :   array_like
                            2d array. The permeability profile along the transverse direction of the waveguide.

                            `mu[i]` defines the values at index i. It has 2 elements, `mu[i][0]` is the component along the waveguide direction, `mu[i][1]` is transverse to the waveguide.

                            For Ez (TM) mode, epsi[i] and mu[i] defines the value in the interval xi[i] to xi[i+1].

                            For Hz (TE) mode, epsi[i] and mu[i] defines the value in the interval (xi[i-1] + xi[i])/2 to (xi[i] + xi[i+1])/2.

                            Such definition is in accordance with Yee-cell definition, i.e. Ez is defined at the corner, Hz is defined at the center.
                            Such definition is in accordance with `tdyno` grid system, i.e. with this `epsi` and `mu`, `tdyno` will calculate the mode `f` at the physical `xi`.

        use_epsi_mu_in_structure : bool
                            If True, use the the permittivity and permeability of the structure being solved.

                            Typically used when `f` is generated with a different resolution.

                            todo: implement this

        omg             :   float
                            intended frequency
        xi0             :   float
                            signed distance from reference line (xi=0) to origin
        reverse_direction : bool
                            If `True`, reverse `beta` direction, while keeping profile `f` unchanged.

                            Can instead manually set kx and ky to negative, in which case it is rotating both `beta` and `f` 180 degrees. Note, for asymmetric waveguide, modal profile `f` is asymmetric.

        plrz            :   str
                            polarization
        if_ndc          :   bool
                            if numeric dispersion compensation

                            If True, do not use ndc in the main update equations.

        omg_ndc         :   float
                            frequency for ndc
        whr             :   str
                            {'all', combinations of 't', 'b', 'l', 'r'}

                            Controls which sides of the TFSF source exist.

                            Default to be 'all', i.e. TFSF source in a rectangular region.

                            Can choose any combination of 't', 'b', 'l', 'r' for top, bottom, left and right.

                            If choose any ONE of the sides, it becomes a uni-directional plane-wave source.
        """

        y = np.array(xi)
        f = np.array(f)
        epsi = np.array(epsi)
        mu = np.array(mu)

        self._xi = y
        # self._xi_h = y_h
        self.f = f
        self.epsi = epsi
        self.mu = mu

        self.reverse_direction = reverse_direction

        dy = y[1] - y[0]
        xp = y[-1] + dy
        xm = y[0] - dy
        yp = np.append(y[1:], xp)
        ym = np.insert(y[:-1], 0, xm)
        yhp = (y + yp) / 2.
        yhm = (y + ym) / 2.
        if plrz == 'Ez' or plrz == 'TM':
            y_h = np.array([y, yhp]).T.ravel()
        else:
            if (plrz != 'Hz') and (plrz != 'TE'):
                warn("polarization not recognized. assume 'Hz'", UserWarning)
            y_h = np.array([yhm, y]).T.ravel()

        self.ff = interp1d(y, f, fill_value="extrapolate")

        epsi_x = epsi[:, 0]
        epsi_y = epsi[:, 1]
        mu_x = mu[:, 0]
        mu_y = mu[:, 1]

        epsi_h = np.array([epsi_y, epsi_y]).T.ravel()
        self.epsif = interp1d(y_h, epsi_h, fill_value="extrapolate")
        mu_h = np.array([mu_y, mu_y]).T.ravel()
        self.muf = interp1d(y_h, mu_h, fill_value="extrapolate")

        fp, fm = self.ff([xp, xm])
        pyff = (np.append(f[1:], fp) - f) / (yp - y)
        pybf = (f - np.insert(f[:-1], 0, fm)) / (y - ym)
        pyffmu = pyff / mu_x
        self.pfmf = interp1d(yhp, pyffmu, fill_value="extrapolate")
        pybfep = pybf / epsi_x
        self.pfef = interp1d(yhm, pybfep, fill_value="extrapolate")

        self.omg = omg
        if xi0 is None:
            self.xi0 = 0.
        else:
            self.xi0 = xi0

        self._c0 = c0
        self._dt = dt
        self._amp = amp
        self._st = st
        self._g = g
        self._plrz = plrz
        self._whr = whr

        self.kx = kx
        self.ky = ky
        self.k = np.sqrt(kx**2 + ky**2)
        k_unit = 1. / np.sqrt(kx**2 + ky**2) * np.array([kx, ky])
        self.kux = k_unit[0]
        self.kuy = k_unit[1]

        self.v = self.omg / self.k * self._c0

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

        # indexing from the bulk
        self.i_te_TM = (yy_n == ymax_ts_n) * (xx_n <= xmax_ts_n) * (xx_n > xmin_ts_n)
        self.i_be_TM = (yy_n == ymin_ts_n) * (xx_n <= xmax_ts_n) * (xx_n > xmin_ts_n)
        self.i_le_TM = (xx_n == xmin_ts_n) * (yy_n <= ymax_ts_n) * (yy_n > ymin_ts_n)
        self.i_re_TM = (xx_n == xmax_ts_n) * (yy_n <= ymax_ts_n) * (yy_n > ymin_ts_n)
        # todo: for modal source should always include min exclue max (that's how modal profile should be recorded)
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

        # self.i_TM_rlv_Dz = self.i_e_TM
        # self.i_TM_rlv_Bx = self.i_te_TM + self.i_be_TM
        # self.i_TM_rlv_By = self.i_le_TM + self.i_re_TM
        # self.i_TE_rlv_Bz = self.i_e_TE
        # self.i_TE_rlv_Dx = self.i_te_TE + self.i_be_TE
        # self.i_TE_rlv_Dy = self.i_le_TE + self.i_re_TE

        self.set_ndc(if_ndc, omg_ndc)

        self._calc_xi()

        self._calc_td()

        # edge indexing
        self.tblr = [False] * 4
        self.iB_s = np.zeros([st.Ny, st.Nx], dtype=bool)
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
            warn('Polarization not recognized. No TFSF source was set.')
            self.i = [np.zeros([st.Ny, st.Nx], dtype=bool)]
        self.iD_s = self.iB_s

        # prepare source fields
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
            # n = np.sqrt(self.epsi * self.mu)
            n = self.k / self.omg  # effective refractive index

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

        kux, kuy, kx, ky = [a if not self.reverse_direction else -a for a in [self.kux, self.kuy, self.kx, self.ky]]

        # calculate relative distances
        xx_r = self._st.dx * (self.xx_n - self.xmin_ts_n)
        yy_r = self._st.dy * (self.yy_n - self.ymin_ts_n)
        d_ts = xx_r * kux + yy_r * kuy
        d_ts -= d_ts[self.i_e_TM].min()

        # calculate time delay
        self.td = d_ts / self.v / ndc
        #
        self.tde_Hx = self._st.dy / 2. * ky / self.omg / self._c0 / ndc
        self.tde_Hy = self._st.dx / 2. * kx / self.omg / self._c0 / ndc
        self.tde_Ex = self._st.dx / 2. * kx / self.omg / self._c0 / ndc
        self.tde_Ey = self._st.dy / 2. * ky / self.omg / self._c0 / ndc
        self.tde_Hz = (self._st.dx / 2. * kx + self._st.dy / 2. * ky) / self.omg / self._c0 / ndc
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

    def _calc_xi(self):
        """
        calculate coordinate xi

        Returns
        -------

        """

        # calculate xi to grid
        xx_r = self._st.xx
        yy_r = self._st.yy
        xi1 = yy_r * self.kux - xx_r * self.kuy
        self.xi = xi1 + self.xi0

        # calculate extra distances
        xi_Hx = self._st.dy / 2. * self.kux
        xi_Hy = - self._st.dx / 2. * self.kuy
        xi_Ex = - self._st.dx / 2. * self.kuy
        xi_Ey = self._st.dy / 2. * self.kux
        xi_Hz = - self._st.dx / 2. * self.kuy + self._st.dy / 2. * self.kux
        #
        self.xi_TE_Hz = self.xi + xi_Hz
        self.xi_TE_Ex = self.xi + xi_Ex
        self.xi_TE_Ey = self.xi + xi_Ey
        self.xi_TM_Ez = self.xi
        self.xi_TM_Hx = self.xi + xi_Hx
        self.xi_TM_Hy = self.xi + xi_Hy
        #
        self.xi_TE_Hz_t, self.xi_TE_Hz_b, self.xi_TE_Hz_l, self.xi_TE_Hz_r = [self.xi_TE_Hz[idx] for idx in [self.i_te_TE, self.i_be_TE, self.i_le_TE, self.i_re_TE]]
        self.xi_TE_Ex_t, self.xi_TE_Ex_b = [self.xi_TE_Ex[idx] for idx in [self.i_te_TE, self.i_be_TE]]
        self.xi_TE_Ey_l, self.xi_TE_Ey_r = [self.xi_TE_Ey[idx] for idx in [self.i_le_TE, self.i_re_TE]]
        self.xi_TM_Ez_t, self.xi_TM_Ez_b, self.xi_TM_Ez_l, self.xi_TM_Ez_r = [self.xi_TM_Ez[idx] for idx in [self.i_te_TM, self.i_be_TM, self.i_le_TM, self.i_re_TM]]
        self.xi_TM_Hx_t, self.xi_TM_Hx_b = [self.xi_TM_Hx[idx] for idx in [self.i_te_TM, self.i_be_TM]]
        self.xi_TM_Hy_l, self.xi_TM_Hy_r = [self.xi_TM_Hy[idx] for idx in [self.i_le_TM, self.i_re_TM]]

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

        kux, kuy, kx, ky = [a if not self.reverse_direction else -a for a in [self.kux, self.kuy, self.kx, self.ky]]

        if self._plrz == 'Hz':

            # todo: should there be pi/2 phase delay in Ex?

            if self.tblr[0]:
                f = self.ff(self.xi_TE_Hz_t)
                td = self.td_TE_Hz_t
                Hz_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TE_Ex_t
                fe = self.ff(self.xi_TE_Ex_t) / self.epsif(self.xi_TE_Ex_t)
                pfe = self.pfef(self.xi_TE_Ex_t)
                Ex_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kux
                                                - self._g.f(t - td) * self.k * fe * kuy)
                #
                self.Bz_s[self.i_te_TE] = - self._dt / self._st.dy * Ex_ic
                self.Dx_s[self.i_te_TE] = self._dt / self._st.dy * Hz_ic
            if self.tblr[1]:
                f = self.ff(self.xi_TE_Hz_b)
                td = self.td_TE_Hz_b
                Hz_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TE_Ex_b
                fe = self.ff(self.xi_TE_Ex_b) / self.epsif(self.xi_TE_Ex_b)
                pfe = self.pfef(self.xi_TE_Ex_b)
                Ex_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kux
                                                - self._g.f(t - td) * self.k * fe * kuy)
                Ex_ic_bl_co = Ex_ic[0]
                #
                self.Bz_s[self.i_be_TE] = self._dt / self._st.dy * Ex_ic
                self.Dx_s[self.i_be_TE] = - self._dt / self._st.dy * Hz_ic
            if self.tblr[2]:
                f = self.ff(self.xi_TE_Hz_l)
                td = self.td_TE_Hz_l
                Hz_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TE_Ey_l
                fe = self.ff(self.xi_TE_Ey_l) / self.epsif(self.xi_TE_Ey_l)
                pfe = self.pfef(self.xi_TE_Ey_l)
                Ey_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kuy
                                                + self._g.f(t - td) * self.k * fe * kux)
                #
                self.Bz_s[self.i_le_TE] = self._dt / self._st.dx * Ey_ic
                self.Dy_s[self.i_le_TE] = self._dt / self._st.dx * Hz_ic
                if self.tblr[1]:
                    self.Bz_s[self.i_co_bl_TE] += self._dt / self._st.dy * Ex_ic_bl_co
            if self.tblr[3]:
                f = self.ff(self.xi_TE_Hz_r)
                td = self.td_TE_Hz_r
                Hz_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TE_Ey_r
                fe = self.ff(self.xi_TE_Ey_r) / self.epsif(self.xi_TE_Ey_r)
                pfe = self.pfef(self.xi_TE_Ey_r)
                Ey_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kuy
                                                + self._g.f(t - td) * self.k * fe * kux)
                #
                self.Bz_s[self.i_re_TE] = - self._dt / self._st.dx * Ey_ic
                self.Dy_s[self.i_re_TE] = - self._dt / self._st.dx * Hz_ic

        elif self._plrz == 'Ez':

            if self.tblr[0]:
                f = self.ff(self.xi_TM_Ez_t)
                td = self.td_TM_Ez_t
                Ez_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TM_Hx_t
                fe = self.ff(self.xi_TM_Hx_t) / self.muf(self.xi_TM_Hx_t)
                pfe = self.pfmf(self.xi_TM_Hx_t)
                Hx_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kux
                                                - self._g.f(t - td) * self.k * fe * kuy)
                Hx_ic_tr_co = Hx_ic[-1]
                #
                self.Dz_s[self.i_te_TM] = self._dt / self._st.dy * Hx_ic
                self.Bx_s[self.i_te_TM] = self._dt / self._st.dy * Ez_ic
            if self.tblr[1]:
                f = self.ff(self.xi_TM_Ez_b)
                td = self.td_TM_Ez_b
                Ez_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TM_Hx_b
                fe = self.ff(self.xi_TM_Hx_b) / self.muf(self.xi_TM_Hx_b)
                pfe = self.pfmf(self.xi_TM_Hx_b)
                Hx_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kux
                                                - self._g.f(t - td) * self.k * fe * kuy)
                #
                self.Dz_s[self.i_be_TM] = - self._dt / self._st.dy * Hx_ic
                self.Bx_s[self.i_be_TM] = - self._dt / self._st.dy * Ez_ic
            if self.tblr[2]:
                f = self.ff(self.xi_TM_Ez_l)
                td = self.td_TM_Ez_l
                Ez_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TM_Hy_l
                fe = self.ff(self.xi_TM_Hy_l) / self.muf(self.xi_TM_Hy_l)
                pfe = self.pfmf(self.xi_TM_Hy_l)
                Hy_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kuy
                                                + self._g.f(t - td) * self.k * fe * kux)
                #
                self.Dz_s[self.i_le_TM] = - self._dt / self._st.dx * Hy_ic
                self.By_s[self.i_le_TM] = self._dt / self._st.dx * Ez_ic
            if self.tblr[3]:
                f = self.ff(self.xi_TM_Ez_r)
                td = self.td_TM_Ez_r
                Ez_ic = self._amp * f * self._g.f(t - td)
                #
                td = self.td_TM_Hy_r
                fe = self.ff(self.xi_TM_Hy_r) / self.muf(self.xi_TM_Hy_r)
                pfe = self.pfmf(self.xi_TM_Hy_r)
                Hy_ic = self._amp / self.omg * (self._g.f(t - td - np.pi / 2 / self.omg) * pfe * kuy
                                                + self._g.f(t - td) * self.k * fe * kux)
                #
                self.Dz_s[self.i_re_TM] = self._dt / self._st.dx * Hy_ic
                self.By_s[self.i_re_TM] = - self._dt / self._st.dx * Ez_ic
                if self.tblr[0]:
                    self.Dz_s[self.i_co_tr_TM] += self._dt / self._st.dy * Hx_ic_tr_co


