# -*- coding: utf-8 -*-


import numpy as np
from warnings import warn
from scipy import sparse as sps


class PCT:

    def __init__(self, dt, p, rto, a_max_fct=1., kpp=1., Nx=1, Ny=1, Nz=1, dx=1., dy=1., dz=1., npx=0, npy=0, npz=0, epsi0=1., mu0=1.):

        """
        PML coefficients for FDTD.

        Parameters
        ----------
        dt              :   float
        p               :   float or Tuple[float, float, float]
                            polynomial order. List of 3 if want different polynomial order for x, y, and z direction.
        rto             :   float or Tuple[float, float, float]
                            target attenuation ratio
        a_max_fct       :   float
                            a_max factor. Coefficient a_max is computed based on sigma(dx), but one can manually scale this factor.
        kpp             :   float or Tuple[float, float, float]
                            kappa, the kappa parameter stretches the real part of xyz in PML.
        Nx, Ny, Nz      :   int
        dx, dy, dz      :   float
                            The solving space parameters.
        npx, npy, npz   :   int
                            number of pml cells in x, y and z (npx=10 means 10 pml cells on each end).
        epsi0, mu0      :   float
        """

        if type(p) is float:
            self.px, self.py, self.pz = [p, p, p]
        else:
            self.px, self.py, self.pz = p

        if type(rto) is float:
            self.rto_x, self.rto_y, self.rto_z = [rto, rto, rto]
        else:
            self.rto_x, self.rto_y, self.rto_z = rto

        if type(kpp) is float:
            kpp = [kpp, kpp, kpp]
        kpp_x, kpp_y, kpp_z = kpp
        kpp_inv = [1. / ka for ka in kpp]

        self.kix = np.ones([Ny, Nx, Nz], dtype=float)
        self.kiy = np.ones([Ny, Nx, Nz], dtype=float)
        self.kiz = np.ones([Ny, Nx, Nz], dtype=float)
        self.kix[:, :npx, :] = kpp_inv[0]
        self.kix[:, (Nx - npx):, :] = kpp_inv[0]
        self.kiy[:npy, :, :] = kpp_inv[1]
        self.kiy[(Ny - npy):, :, :] = kpp_inv[1]
        self.kiz[:, :, :npz] = kpp_inv[2]
        self.kiz[:, :, (Nz - npz):] = kpp_inv[2]
        self.kixV, self.kiyV, self.kizV = [s.reshape(Nx * Ny * Nz) for s in [self.kix, self.kiy, self.kiz]]

        # coefficients sgm, a, b and c
        sgm_xb_a, a_xb_a, b_xb_a, c_xb_a = self.c_pc_1_a('b', dt, p, rto, a_max_fct, kpp_x, dx, Nx, npx, epsi0, mu0)
        sgm_xf_a, a_xf_a, b_xf_a, c_xf_a = self.c_pc_1_a('f', dt, p, rto, a_max_fct, kpp_x, dx, Nx, npx, epsi0, mu0)
        sgm_yb_a, a_yb_a, b_yb_a, c_yb_a = self.c_pc_1_a('b', dt, p, rto, a_max_fct, kpp_y, dy, Ny, npy, epsi0, mu0)
        sgm_yf_a, a_yf_a, b_yf_a, c_yf_a = self.c_pc_1_a('f', dt, p, rto, a_max_fct, kpp_y, dy, Ny, npy, epsi0, mu0)
        sgm_zb_a, a_zb_a, b_zb_a, c_zb_a = self.c_pc_1_a('b', dt, p, rto, a_max_fct, kpp_z, dz, Nz, npz, epsi0, mu0)
        sgm_zf_a, a_zf_a, b_zf_a, c_zf_a = self.c_pc_1_a('f', dt, p, rto, a_max_fct, kpp_z, dz, Nz, npz, epsi0, mu0)

        # coefficients b and c, in x y and z direction
        self.bxb, self.byb, self.bzb = np.meshgrid(b_xb_a, b_yb_a, b_zb_a)
        self.bxf, self.byf, self.bzf = np.meshgrid(b_xf_a, b_yf_a, b_zf_a)
        self.cxb, self.cyb, self.czb = np.meshgrid(c_xb_a, c_yb_a, c_zb_a)
        self.cxf, self.cyf, self.czf = np.meshgrid(c_xf_a, c_yf_a, c_zf_a)

        # self.bxbM, self.bybM, self.bzbM, self.bxfM, self.byfM, self.bzfM, \
        # self.cxbM, self.cybM, self.czbM, self.cxfM, self.cyfM, self.czfM \
        #     = [sps.diags(s.reshape(Nx*Ny*Nz), 0, shape=(Nx*Ny*Nz, Nx*Ny*Nz))
        #        for s in [self.bxb, self.byb, self.bzb, self.bxf, self.byf, self.bzf,
        #                  self.cxb, self.cyb, self.czb, self.cxf, self.cyf, self.czf]]

        # flatten the coefficient matrices.
        self.bxbV, self.bybV, self.bzbV, self.bxfV, self.byfV, self.bzfV, \
        self.cxbV, self.cybV, self.czbV, self.cxfV, self.cyfV, self.czfV \
            = [s.reshape(Nx*Ny*Nz) for s in [self.bxb, self.byb, self.bzb, self.bxf, self.byf, self.bzf,
                                             self.cxb, self.cyb, self.czb, self.cxf, self.cyf, self.czf]]

        x_n = np.arange(Nx)
        y_n = np.arange(Ny)
        z_n = np.arange(Nz)
        xx_n, yy_n, zz_n = np.meshgrid(x_n, y_n, z_n)

        # indexing of PML regions
        ixiP = np.logical_or(xx_n < npx, xx_n >= (Nx - npx))
        iyiP = np.logical_or(yy_n < npy, yy_n >= (Ny - npy))
        iziP = np.logical_or(zz_n < npz, zz_n >= (Nz - npz))
        self.iiP = np.logical_or(ixiP, iyiP, iziP)
        self.iiPV = self.iiP.ravel()
        # self.iiPM = np.diagflat(self.iiP)

        # b and c coefficients
        self.bxbiP, self.bybiP, self.bzbiP, self.bxfiP, self.byfiP, self.bzfiP, \
        self.cxbiP, self.cybiP, self.czbiP, self.cxfiP, self.cyfiP, self.czfiP \
            = [coef[self.iiP] for coef in [self.bxb, self.byb, self.bzb, self.bxf, self.byf, self.bzf,
                                           self.cxb, self.cyb, self.czb, self.cxf, self.cyf, self.czf]]
        self.bxbViP, self.bybViP, self.bzbViP, self.bxfViP, self.byfViP, self.bzfViP, \
        self.cxbViP, self.cybViP, self.czbViP, self.cxfViP, self.cyfViP, self.czfViP \
            = [coef[self.iiPV] for coef in [self.bxbV, self.bybV, self.bzbV, self.bxfV, self.byfV, self.bzfV,
                                            self.cxbV, self.cybV, self.czbV, self.cxfV, self.cyfV, self.czfV]]

    def c_pc_1_a(self, fb, dt, p, rto, a_max_fct, kpp, dL, N, Np, epsi0, mu0):

        """
        Calculate PML coefficients as 1d array.

        Parameters
        ----------
        fb              :   str
                            'f' or 'b'
        dt              :   float
        p               :   int
        rto             :   float
                            ratio
        a_max_fct       :   float
                            a_max factor
        kpp             :   float
                            kappa
        dL              :   float
        N               :   int
        Np              :   int
                            number of PML cells
        epsi0
        mu0

        Returns
        -------
        sgm         :   float
        a           :   ndarray
        b           :   ndarray
        c           :   ndarray
        """

        a = np.zeros(N, dtype=float)
        sgm = np.zeros(N, dtype=float)
        c = np.zeros(N, dtype=float)

        if Np != 0:
            dp = Np * dL
            # vacuum impedance
            vac_imp = np.sqrt(mu0 / epsi0)
            sgm_max = - (p + 1) * np.log(rto) / 2. / vac_imp / dp
            a_max = a_max_fct * 1000 * sgm_max * (1. / 2 / N) ** p

            if fb == 'f':
                # sigma as a function of l
                sgml = sgm_max * ((np.arange(Np) + 0.5) / float(Np)) ** p
                sgm[(N - Np):] = sgml
                sgm[(Np - 1)::-1] = sgml

                # a as a function l
                al = a_max * (1 - (np.arange(Np) + 0.5) / float(Np)) ** p
                a[(N - Np):] = al
                a[(Np - 1)::-1] = al

            elif fb == 'b':
                sgml_r = sgm_max * (np.arange(Np) / float(Np)) ** p
                sgml_l = sgm_max * ((np.arange(Np) + 1.) / float(Np)) ** p
                sgm[(N - Np):] = sgml_r
                sgm[(Np - 1)::-1] = sgml_l

                al_r = a_max * (1 - np.arange(Np) / float(Np)) ** p
                al_l = a_max * (1 - (np.arange(Np) + 1.) / float(Np)) ** p
                a[(N - Np):] = al_r
                a[(Np - 1)::-1] = al_l
            else:
                warn('fb must be f or b!', UserWarning)

            b = np.exp(-1. / epsi0 * (sgm / kpp + a) * dt)

            # indexing of pml region.
            ip = (sgm != 0)
            c[ip] = sgm[ip] / (sgm[ip] * kpp + kpp ** 2 * a[ip]) * (b[ip] - 1)

        else:
            b = np.zeros(N, dtype=float)
            c = np.zeros(N, dtype=float)

        return sgm, a, b, c
