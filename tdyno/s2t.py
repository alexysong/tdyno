#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song, Sep 2017

Structure 2d in FDTD.
"""


import numpy as np
import numpy.linalg as la
from warnings import warn


class S2T:

    def __init__(self, xmin, xmax, dx, ymin, ymax, dy, epsi_bg, mu_bg, x=None, y=None):

        """
        Set up the structure to be solved in 2D FDTD.

        Parameters
        ----------
        xmin, xmax, dx, ymin, ymax, dy      :   float
        epsi_bg, mu_bg                      :   float
        x, y                                :   ndarray
                                                1d ndarray, If supplied, override xmin etc.
        """

        if type(epsi_bg) is float:
            self.epsi_bg = np.eye(3, 3) * epsi_bg
        elif np.array(epsi_bg).ndim == 1:
            self.epsi_bg = np.diag(epsi_bg)
        else:
            self.epsi_bg = epsi_bg
        if type(mu_bg) is float:
            self.mu_bg = np.eye(3, 3) * mu_bg
        elif np.array(mu_bg).ndim == 1:
            self.mu_bg = np.diag(mu_bg)
        else:
            self.mu_bg = mu_bg
        self.epsi_bg_inv = la.inv(self.epsi_bg)
        self.mu_bg_inv = la.inv(self.mu_bg)

        # parameters of the solving space
        if x is not None:
            dx = x[1] - x[0]
            xmin = x[0]
            xmax = x[-1] + dx
            Nx = x.size
            x_n = np.arange(Nx, dtype=int)
        else:
            Nx = int((xmax + dx / 1.e4 - xmin) / dx)
            x_n = np.arange(Nx, dtype=int)
            x = x_n * dx + xmin
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.xres = dx
        self.Nx = Nx
        self.x_n = x_n
        self.x = x
        #
        if y is not None:
            dy = y[1] - y[0]
            ymin = y[0]
            ymax = y[-1] + dy
            Ny = y.size
            y_n = np.arange(Nx, dtype=int)
        else:
            Ny = int((ymax + dx / 1.e4 - ymin) / dy)
            y_n = np.arange(Ny, dtype=int)
            y = y_n * dy + ymin
        self.ymin = ymin
        self.ymax = ymax
        self.dy = dy
        self.yres = dy
        self.Ny = Ny
        self.y_n = y_n
        self.y = y
        #
        self.xx_n, self.yy_n = np.meshgrid(self.x_n, self.y_n)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # list of box objects
        self.bxs = []
        self.bxs_sp = []
        self.bxs_Lz = []
        self.bxs_dmLz = []
        self.bxs_Dr = []
        self.bxs_dmri = []
        self.bxs_lg = []
        self.bxs_dmlg = []

    def a_b(self,
            shp='rct',
            xmin_b=None, xmax_b=None, ymin_b=None, ymax_b=None,
            xc_b=None, yc_b=None,
            r_b=None,
            a_1=None, a_2=None,
            w_b=None,
            knd='sp',
            epsi_b=None, mu_b=None,
            epsi_infty_b=None, mu_infty_b=None,
            omgs_rsn=None, dlts_epsi=None, Gms=None,
            sgm=None,
            m_a=None, m_o=None, m_q=None, m_p=0.):

        """
        Add boxes to the structure.

        Parameters
        ----------
        shp                             :   str
                                            ['rct'|'ccl'|'rng'|'wdg']
        xmin_b, xmax_b, ymin_b, ymax_b  :   float
                                            four corners of the box.
        xc_b, yc_b                      :   float
                                            x and y center of box
        r_b                             :   float
                                            radius of circular box, outer radius for ring and wedge box
        a_1, a_2                        :   float
                                            start and end angle of wedge, in degrees, between 0 to 360
        w_b                             :   float
                                            width of ring or wedge box
        knd                             :   str
                                            ['sp'|'Lz'|'dmLz'|'Dr'|'lg'|'dmri'|'dmlg']
        epsi_b, mu_b                    :   float or Tuple[float, float, float] or 3x3 ndarray
                                            The permittivity and permeability.
        epsi_infty_b, mu_infty_b        :   float or Tuple[float, float, float] or 3x3 ndarray
                                            The high-frequency permittivity and permeability.
                                            If not supplied, fall back to epsi_b and mu_b.
        omgs_rsn                        :   list[float]
                                            resonance frequencies
        dlts_epsi                       :   list[float]
                                            permittivity change at each resonance
        Gms                             :   list[float]
                                            Broadening of each resonance
        sgm                             :   float
                                            loss (positive number) or gain (negative) in the material
        m_a                             :   float or list[float]
                                            modulation amplitude
        m_o                             :   float or list[float]
                                            modulation frequency
        m_q                             :   Tuple[float, float] or list[Tuple[float, float]]
                                            (qx, qy), wave vector of modulation.
        m_p                             :   float or list[float]
                                            modulation phase
                                            By default, cos modulation.

        Returns
        -------

        """

        gmt = Gmt2(self.xx, self.yy,
                   shp=shp,
                   xmin_rec_b=xmin_b, xmax_rec_b=xmax_b, ymin_rec_b=ymin_b, ymax_rec_b=ymax_b,
                   xc_b=xc_b, yc_b=yc_b,
                   r_b=r_b,
                   a_1=a_1, a_2=a_2,
                   w_b=w_b)

        mtr = Mtr(knd=knd,
                  epsi=epsi_b, mu=mu_b, epsi_infty=epsi_infty_b, mu_infty=mu_infty_b,
                  omgs_rsn=omgs_rsn, dlts_epsi=dlts_epsi, Gms=Gms,
                  sgm=sgm,
                  m_a=m_a, m_o=m_o, m_q=m_q, m_p=m_p)

        bx = Bx2(gmt, mtr)

        if knd == 'sp':
            self.bxs_sp.append(bx)
        elif knd == 'Lz':
            self.bxs_Lz.append(bx)
        elif knd == 'dmLz':
            self.bxs_dmLz.append(bx)
        elif knd == 'Dr':
            self.bxs_Dr.append(bx)
        elif knd == 'dmri':
            self.bxs_dmri.append(bx)
        elif knd == 'lg':
            self.bxs_lg.append(bx)
        elif knd == 'dmlg':
            self.bxs_dmlg.append(bx)
        else:
            print('Material not recognized.')

        if bx is not None:
            self.bxs.append(bx)


class Bx2:
    def __init__(self, gmt, mtr):
        """
        A box in 2D.

        Parameters
        ----------
        gmt        :   geometry
        mtr        :   material
        """

        self.gmt = gmt
        self.mtr = mtr


class MtrBsc(object):
    def __init__(self, epsi=1., mu=1., **kwargs):
        """
        Basic material.

        Parameters
        ----------
        epsi, mu    :   float or Tuple[float, float, float] or 3x3 ndarray
        kwargs      :
                        other kwargs not related to materials
        """

        if type(epsi) is float:
            epsi = np.eye(3, 3) * epsi
        elif np.array(epsi).ndim == 1:
            epsi = np.diag(epsi)
        if type(mu) is float:
            mu = np.eye(3, 3) * mu
        elif np.array(mu).ndim == 1:
            mu = np.diag(mu)
        self.epsi = epsi
        self.mu = mu


class Mtr(MtrBsc):
    def __init__(self,
                 knd='sp',
                 epsi_infty=1., mu_infty=1., omgs_rsn=None, dlts_epsi=None, Gms=None,
                 sgm=None,
                 m_a=None, m_o=None, m_q=None, m_p=None,
                 **kw):
        """
        Material type and parameters.

        Parameters
        ----------
        knd                             :   str
                                            'sp' (simple),
                                            'Lz' (Lorentz),
                                            'dmLz' (dynamic modulation of delta epsi),
                                            'Dr' (Drude),
                                            'lg' (loss and gain),
                                            'dmri' (dynamic modulation of real part of index),
                                            'dmlg' (dynamic modulation of loss and gain).
        epsi_infty_b, mu_infty_b        :   float or Tuple[float, float, float] or 3x3 ndarray
                                            The high-frequency permittivity and permeability for use in Lorentz and Drude.
                                            If not supplied, fall back to epsi_b and mu_b.
        omgs_rsn                        :   list[float]
                                            resonance frequencies, for use in Lorentz or Drude
        dlts_epsi                       :   list[float]
                                            permittivity change at each resonance, for use in Lorentz
        Gms                             :   list[float]
                                            Broadening of each resonance, for use in Lorentz or Drude
        sgm                             :   float
                                            loss (positive number) or gain (negative) in the material
        m_a                             :   float or list[float]
                                            modulation amplitude
        m_o                             :   float or list[float]
                                            modulation frequency
        m_q                             :   Tuple[float, float] or list[Tuple[float, float]]
                                            (qx, qy), wave vector of modulation.
        m_p                             :   float or list[float]
                                            modulation phase
                                            By default, cos modulation.

        Keyword Arguments
        -----------------
        kw                      :   keyword arguments to pass to MtrBsc
        """

        if (knd == 'Lz') or (knd == 'dmLz') or (knd == 'Dr'):
            if epsi_infty is None:
                epsi_infty = kw.pop('epsi')
            if mu_infty is None:
                mu_infty = kw.pop('mu')
            super(Mtr, self).__init__(epsi=epsi_infty, mu=mu_infty)
            self.epsi_infty = self.epsi
            self.mu_infty = self.mu
        else:
            super(Mtr, self).__init__(**kw)

        self.knd = knd

        if omgs_rsn is None:
            omgs_rsn = []
        if dlts_epsi is None:
            dlts_epsi = []
        if Gms is None:
            Gms = []

        self.omgs_rsn = omgs_rsn
        self.dlts_epsi = dlts_epsi
        self.Gms = Gms

        self.sgm = sgm

        self.m_a = m_a
        self.m_o = m_o
        self.m_q = m_q
        self.m_p = m_p


class Gmt2:
    def __init__(self, xx, yy,
                 shp='rct',
                 xmin_rec_b=None, xmax_rec_b=None, ymin_rec_b=None, ymax_rec_b=None,
                 xc_b=None, yc_b=None,
                 r_b=None,
                 w_b=None,
                 a_1=None, a_2=None,
                 **kwargs
                 ):

        """
        Geometrical information of a box.

        Parameters
        ----------
        xx, yy          :   ndarray
        shp             :   str
                            'rct' (rectangular),
                            'ccl' (circular),
                            'rng' (ring),
                            'wdg' (wedge)
        xmin_rec_b
        xmax_rec_b
        ymin_rec_b
        ymax_rec_b      :   float
                            four corners of a rectangular box.
        xc_b, yc_b      :   float
                            x and y center of box
        r_b             :   float
                            radius of circular box, the outer radius for ring and wedge box
        a_1, a_2        :   float
                            the start and the end angle of an wedge, in degrees, between 0 to 360
        w_b             :   float
                            width of a ring or wedge box
        kwargs          :
                            other kwargs supplied but not related to geometry
        """

        # self.shp = shp
        if shp == 'rct':
            self.pch = {'shp': 'rct',
                        'xy': (xmin_rec_b, ymin_rec_b),
                        'width': xmax_rec_b - xmin_rec_b,
                        'height': ymax_rec_b - ymin_rec_b
                        }
            # indexing in box
            self.iib = (xx < xmax_rec_b) * (xx >= xmin_rec_b) * (yy < ymax_rec_b) * (yy >= ymin_rec_b)
            self.iibV = self.iib.ravel()

        elif shp == 'ccl':
            self.pch = {'shp': 'ccl',
                        'center': (xc_b, yc_b),
                        'radius': r_b,
                        }
            # indexing in box
            d = np.sqrt(((xx - xc_b) ** 2 + (yy - yc_b) ** 2))
            self.iib = d < r_b
            self.iibV = self.iib.ravel()

        elif shp == 'rng':
            self.pch = {'shp': 'rng',
                        'center': (xc_b, yc_b),
                        'radius': r_b,
                        'width': w_b,
                        }
            # indexing in box
            d = np.sqrt(((xx - xc_b) ** 2 + (yy - yc_b) ** 2))
            self.iib = (d < r_b) * (d >= r_b - w_b)
            self.iibV = self.iib.ravel()

        elif shp == 'wdg':
            self.pch = {'shp': 'wdg',
                        'center': (xc_b, yc_b),
                        'radius': r_b,
                        'angles': (a_1, a_2),
                        'width': w_b,
                        }
            # indexing in box
            d = np.sqrt(((xx - xc_b) ** 2 + (yy - yc_b) ** 2))
            if w_b is None:
                i_i_r = d < r_b
            else:
                i_i_r = (d < r_b) * (d >= r_b - w_b)
            z_ = (xx - xc_b) + (yy - yc_b) * 1j
            a_ = np.angle(z_, deg=True)
            a1 = (a_1 + 180) % 360 - 180
            # regularize angles
            if a_2 != 180.:
                a2 = (a_2 + 180) % 360 - 180
            else:
                a2 = 180.
            if a2 > a1:
                i_i_a = (a_ >= a1) * (a_ < a2)
            else:
                i_i_a = np.logical_not(((a_ >= a1) * (a_ < a2)))
            self.iib = i_i_r * i_i_a
            self.iibV = self.iib.ravel()
