# -*- coding: utf-8 -*-


import scipy.sparse as sps


class DOM:

    def __init__(self, Nx=1, Ny=1, Nz=1, dx=1., dy=1., dz=1.):
        """
        Differential Operator Matrices.

        Parameters
        ----------
        Nx, Ny, Nz  :   integer
                        number of unit-cell sites in x, y, and z direction
        dx, dy, dz  :   float
                        unit-cell sizes

        """

        eye_Nx = sps.eye(Nx)
        eye_Ny = sps.eye(Ny)
        eye_Nz = sps.eye(Nz)

        if Nx > 1:
            pxfr = sps.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
            pxbr = sps.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
        else:
            pxfr = sps.diags([0.], 0, shape=(1, 1))
            pxbr = sps.diags([0.], 0, shape=(1, 1))
        self.pxf = sps.kron(eye_Nz, sps.kron(eye_Ny, pxfr)) / dx
        self.pxb = sps.kron(eye_Nz, sps.kron(eye_Ny, pxbr)) / dx

        if Ny > 1:
            self.pyf = sps.kron(eye_Nz,  (sps.kron(sps.diags([1, -1, 1], [-Ny + 1, 0, 1], shape=(Ny, Ny)), eye_Nx))  ) / dy
            self.pyb = sps.kron(eye_Nz,  (sps.kron(sps.diags([-1, 1, -1], [-1, 0, Ny - 1], shape=(Ny, Ny)), eye_Nx))  ) / dy
        else:
            self.pyf = sps.kron(eye_Nz,  (sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_Nx))  ) / dy
            self.pyb = sps.kron(eye_Nz,  (sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_Nx))  ) / dy

        eye_NyNx = sps.kron(eye_Ny, eye_Nx)
        if Nz > 1:
            self.pzf = sps.kron(sps.diags([1, -1, 1], [-Nz+1, 0, 1], shape=(Nz, Nz)), eye_NyNx) / dz
            self.pzb = sps.kron(sps.diags([-1, 1, -1], [-1, 0, Nz-1], shape=(Nz, Nz)), eye_NyNx) / dz
        else:
            self.pzf = sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_NyNx) / dz
            self.pzb = sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_NyNx) / dz


class DOMH2D:

    def __init__(self, Nx=1, Ny=1, Nz=1, dx=1., dy=1., dz=1.):
        """
        Differential Operator Matrices Hexagonal 2D.

        Parameters
        ----------
        Nx, Ny, Nz      :   int
        dx, dy, dz      :   float
        """

        # self.Nx = Nx
        # self.Ny = Ny
        # self.P = Nx * Ny

        eye_Nx = sps.eye(Nx)
        eye_Ny = sps.eye(Ny)

        # Nx and Ny should be > 1
        if Nx > 1:
            pxfr = sps.diags([-1, 1, 1], [0, 1, -Nx + 1], shape=(Nx, Nx))
            pxbr = sps.diags([1, -1, -1], [0, -1, Nx - 1], shape=(Nx, Nx))
            self.pxf = sps.kron(eye_Ny, pxfr) / dx
            self.pxb = sps.kron(eye_Ny, pxbr) / dx
        if Ny > 1:
            self.pyf = (-sps.kron(eye_Ny, eye_Nx)
                        + sps.kron(sps.diags([1], [1], shape=(Ny, Ny)), eye_Nx)
                        + sps.kron(sps.diags([1], [-Ny + 1], shape=(Ny, Ny)), sps.diags([1, 1], [Nx / 2, -Nx / 2], shape=(Nx, Nx)))) / dy
            self.pyb = (sps.kron(eye_Ny, eye_Nx)
                        + sps.kron(sps.diags([-1], [-1], shape=(Ny, Ny)), eye_Nx)
                        + sps.kron(sps.diags([1], [Ny - 1], shape=(Ny, Ny)), sps.diags([-1, -1], [Nx / 2, -Nx / 2], shape=(Nx, Nx)))) / dy

        eye_NyNx = sps.kron(eye_Ny, eye_Nx)
        if Nz > 1:
            self.pzf = sps.kron(sps.diags([1, -1, 1], [-Nz+1, 0, 1], shape=(Nz, Nz)), eye_NyNx) / dz
            self.pzb = sps.kron(sps.diags([-1, 1, -1], [-1, 0, Nz-1], shape=(Nz, Nz)), eye_NyNx) / dz
        else:
            self.pzf = sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_NyNx) / dz
            self.pzb = sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_NyNx) / dz
