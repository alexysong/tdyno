# -*- coding: utf-8 -*-


import time

import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, CheckButtons, TextBox
from matplotlib.animation import FFMpegWriter

from .dom import DOM
from .a_pchs import a_pchs
from .plot_field_2d_in_ax_3 import PlotField2DinAx


class RT2:

    def __init__(self,
                 c0, epsi0, mu0,
                 st,
                 scs,
                 dt, Nt,
                 pc=None,
                 plrz='Hz',
                 if_ndc=False, omg_ndc=None,
                 md='HDE',
                 mnts=None,
                 skp=10,
                 **kwargs
                 ):

        """
        Run FDTD 2D simulation.

        Parameters
        ----------
        c0                                  :   float
        epsi0                               :   float
        mu0                                 :   float
        st                                  :   S2T
                                                the structure to be solved
        scs                                 :   list[Union[PS2, TSS2]]
                                                sources
        dt                                  :   float
        Nt                                  :   int
                                                total number of time steps after each time you click Play/Pause button.
        pc                                  ：  PCT
                                                pml
        plrz                                :   str
                                                polarization, either 'Hz' or 'Ez'
        if_ndc                              :   bool
                                                whether to implement numeric dispersion compensation
        omg_ndc                             :   float
                                                the frequency for numeric dispersion compensation.
        md                                  :   str
                                                mode, either 'HDE' or 'BHDE'
        mnts                                :   list[Union[MntPntAmp, MntMltPntAmp]]
                                                list of monitors
        skp                                 :   int
                                                skipping in plotting.

        Keyword arguments
        -----------------
        kwargs                              :   dict
                                                key word arguments for PlotField2DinAx


        Attributes
        ----------

        """

        self.c0 = c0
        self.epsi0 = epsi0
        self.mu0 = mu0
        self.st = st
        self.scs = scs
        self.pc = pc
        self.plrz = plrz
        self.md = md
        self.dt = dt
        self.Nt = Nt
        self.mnts = mnts
        self.skp = skp

        if not self.pc:
            self.if_pc = False
        else:
            self.if_pc = True

        # For dynamic modulation
        if self.st.bxs_dmri:
            self.epsi_x_dm = self.st.epsi_bg[0, 0] * np.ones([self.st.Ny, self.st.Nx])
            self.epsi_y_dm = self.st.epsi_bg[1, 1] * np.ones([self.st.Ny, self.st.Nx])
            self.epsi_z_dm = self.st.epsi_bg[2, 2] * np.ones([self.st.Ny, self.st.Nx])

        # For loss/gain boxes in structure
        self.if_lg = False
        if self.st.bxs_lg or self.st.bxs_dmlg:
            self.if_lg = True

            # index of the region not in loss/gain boxes.
            self.i_not_lg = np.ones([self.st.Ny, self.st.Nx], dtype=bool)

            # todo: This does not seem necessary. Doesn't need to store sgm in self.
            self.sgm_x = np.zeros([self.st.Ny, self.st.Nx])
            self.sgm_y = np.zeros([self.st.Ny, self.st.Nx])
            self.sgm_z = np.zeros([self.st.Ny, self.st.Nx])
            for box in (self.st.bxs_lg+self.st.bxs_dmlg):
                self.i_not_lg[box.gmt.iib] = False
                self.sgm_x[box.gmt.iib] = box.mtr.sgm
                self.sgm_y[box.gmt.iib] = box.mtr.sgm
                self.sgm_z[box.gmt.iib] = box.mtr.sgm
            self.i_not_lg_V = self.i_not_lg.ravel()
            self.sgm_x_V = self.sgm_x.ravel()
            self.sgm_y_V = self.sgm_y.ravel()
            self.sgm_z_V = self.sgm_z.ravel()

        self.t = 0
        self.psd = True  # paused

        # Initialize B, H, D, E fields
        self.Bx, self.By, self.Bz, self.Hx, self.Hy, self.Hz, self.Dx, self.Dy, self.Dz, self.Ex, self.Ey, self.Ez = [np.zeros([self.st.Ny, self.st.Nx]) for ii in range(12)]
        self.Bx_V, self.By_V, self.Bz_V, self.Hx_V, self.Hy_V, self.Hz_V, self.Dx_V, self.Dy_V, self.Dz_V, self.Ex_V, self.Ey_V, self.Ez_V = [f.ravel() for f in [self.Bx, self.By, self.Bz, self.Hx, self.Hy, self.Hz, self.Dx, self.Dy, self.Dz, self.Ex, self.Ey, self.Ez]]

        # Initialize psi, JP and P fields
        if self.pc:
            self.psi_Bzx_V_iP, self.psi_Bzy_V_iP, self.psi_Dxy_V_iP, self.psi_Dyx_V_iP = [np.zeros(self.pc.iiPV[self.pc.iiPV].size) for ii in range(4)]
            self.psi_Bxy_V_iP, self.psi_Byx_V_iP, self.psi_Dzx_V_iP, self.psi_Dzy_V_iP = [np.zeros(self.pc.iiPV[self.pc.iiPV].size) for ii in range(4)]
        self.JPx_Lz, self.JPy_Lz, self.JPz_Lz = [[[np.zeros(box.gmt.iib[box.gmt.iib].shape, dtype=float) for g in box.mtr.Gms] for box in self.st.bxs_Lz] for ii in range(3)]
        self.Px_Lz, self.Py_Lz, self.Pz_Lz = [[[np.zeros(box.gmt.iib[box.gmt.iib].shape, dtype=float) for g in box.mtr.Gms] for box in self.st.bxs_Lz] for ii in range(3)]
        self.JPx_Dr, self.JPy_Dr, self.JPz_Dr = [[[np.zeros(box.gmt.iib[box.gmt.iib].shape, dtype=float) for g in box.mtr.Gms] for box in self.st.bxs_Dr] for ii in range(3)]
        self.Px_Dr, self.Py_Dr, self.Pz_Dr = [[[np.zeros(box.gmt.iib[box.gmt.iib].shape, dtype=float) for g in box.mtr.Gms] for box in self.st.bxs_Dr] for ii in range(3)]
        self.JPx_dmLz, self.JPy_dmLz, self.JPz_dmLz = [[[np.zeros(box.gmt.iib[box.gmt.iib].shape, dtype=float) for g in box.mtr.Gms] for box in self.st.bxs_dmLz] for ii in range(3)]
        self.Px_dmLz, self.Py_dmLz, self.Pz_dmLz = [[[np.zeros(box.gmt.iib[box.gmt.iib].shape, dtype=float) for g in box.mtr.Gms] for box in self.st.bxs_dmLz] for ii in range(3)]

        #  Differential operators
        self.d = DOM(Nx=self.st.Nx, Ny=self.st.Ny, dx=self.st.dx, dy=self.st.dy)

        # numeric dispersion compensation
        self.set_ndc(if_ndc, omg_ndc)

        # ==========   Start the plotting window, handle plotting and user interface   ==========

        self.fig = plt.figure()
        gs = GridSpec(1, 1, top=0.85, bottom=0.25, left=0.1, right=0.9)

        # play/pause button
        self.ax_pp = plt.axes([0.1, 0.04, 0.1, 0.05])
        self.b_pp = Button(self.ax_pp, r'Play/Pause')
        self.b_pp.on_clicked(self.pp)

        # step button
        self.ax_s = plt.axes([0.25, 0.04, 0.1, 0.05])
        self.b_s = Button(self.ax_s, r'Step')
        self.b_s.on_clicked(self.stp)

        # pcm render button
        self.ax_pr = plt.axes([0.4, 0.04, 0.1, 0.05])
        self.b_pr = Button(self.ax_pr, r'Render')
        self.b_pr.on_clicked(self.pcm_rndr)
        self.fig_pr_p = None
        self.ax_pr_p = None

        # reset button
        self.ax_r = plt.axes([0.55, 0.04, 0.1, 0.05])
        self.b_r = Button(self.ax_r, 'Reset')
        self.b_r.on_clicked(self.r_f)

        # press 'right' key  = step
        self.fig.canvas.mpl_connect('key_press_event', self.k_p)

        # main ax containing the field plot
        self.ax = self.fig.add_subplot(gs[0, 0])  # Axes
        # if use blitting, need to set_animated
        # self.ax.set_animated(True)
        if plrz == 'Hz':
            sf = self.Hz
            title = 'Hz'
        else:  # i.e. plrz is 'Ez'
            sf = self.Ez
            title = 'Ez'

        # patch collection list
        self.phcs = []

        # collect different type of boxes (different edge colors)
        for box in self.st.bxs_sp:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[0., 0., 0., 0.5])  # black
            self.phcs.append(phc)
        for box in self.st.bxs_Lz:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[0., 0.3, 0.8, 1.])  # blue
            self.phcs.append(phc)
        for box in self.st.bxs_dmLz:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[1., 0.6, 0.1, 1.])  # orange
            self.phcs.append(phc)
        for box in self.st.bxs_Dr:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[1., 0.8, 0.2, 1.])  # yellow
            self.phcs.append(phc)
        for box in self.st.bxs_dmri:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[1., 0.2, 1., 1.])  # purple
            self.phcs.append(phc)
        for box in self.st.bxs_lg:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[0.1, 1., 0.2, 1.])  # green
            self.phcs.append(phc)
        for box in self.st.bxs_dmlg:
            phc = a_pchs(self.ax, [box.gmt.pch], fc='None', alpha=None, lw=0.75, ec=[0.7, 0.0, 0.0, 1.])  # red
            self.phcs.append(phc)

        # store the source box
        for sc in self.scs:
            if hasattr(sc, 'pchs') and sc.pchs:
                phc = a_pchs(self.ax, sc.pchs, fc='none', lw=0.75, alpha=0.4, ec='purple')
                self.phcs.append(phc)

        # store the monitor as a box
        if mnts is not None and mnts:
            for mnt in mnts:
                phc = a_pchs(self.ax, mnt.pchs, fc='none', lw=0.75, alpha=0.4, ec='black')
                self.phcs.append(phc)

        if self.if_pc:
            shaded_region = self.pc.iiP[:, :, 0]
        else:
            shaded_region = None
        self.plotter = PlotField2DinAx(
            self.ax,
            xmin=self.st.xmin, xmax=self.st.xmax, xres=self.st.dx, ymin=self.st.ymin, ymax=self.st.ymax, yres=self.st.dy, dire='v',
            title=title, xlabel='$z$', ylabel='$x$',
            sf=sf, if_scalar_field_intensity=True, scalar_field_part='real', if_colorbar=True,
            shaded_region=shaded_region, if_update_shading=False,
            **kwargs)

        # Show time steps
        bbox_props = dict(boxstyle="round, pad=0.3", fc=[.85, .9, 1., .9], lw=0)
        self.an_t = self.ax.annotate('time step : {:d}'.format(self.t), fontsize=14, xy=(0.5, 0.5), xycoords="data",
                                     xytext=(0.8, 0.065), textcoords='figure fraction',
                                     horizontalalignment='center', verticalalignment='center',
                                     bbox=bbox_props
                                     )

        # video recording parameters
        self.if_vid_rec = False
        self.if_vid_rec_ps = False
        self.vid_fps = 30
        self.vid_dpi = 100
        self.vid_n = 0
        self.vid_ttl = 'v'
        self.vid_of = self.vid_ttl + '_{:d}.mp4'.format(self.vid_n)
        self.vid_metadata = dict(artist='Alex Y. Song')

        # video writer
        self.vid_wtr = FFMpegWriter(fps=self.vid_fps, metadata=self.vid_metadata)
        # self.vid_wtr.setup(fig=self.fig, outfile=self.vid_of)

        # video recording button
        self.ax_vid_rec = plt.axes([0.1, 0.93, 0.085, 0.05])
        self.b_vid_rec = CheckButtons(self.ax_vid_rec, [r'Record'], [False])
        self.b_vid_rec.on_clicked(self.vid_rec)

        # video recording pause
        self.ax_vid_rec_ps = plt.axes([0.195, 0.93, 0.08, 0.05])
        self.b_vid_rec_ps = CheckButtons(self.ax_vid_rec_ps, [r'Pause'], [False])
        self.b_vid_rec_ps.on_clicked(self.vid_rec_ps)

        # video fps
        self.ax_vid_fps = plt.axes([0.31, 0.93, 0.04, 0.05])
        self.tb_vid_fps = TextBox(self.ax_vid_fps, 'fps', initial='{:d}'.format(self.vid_fps))
        self.tb_vid_fps.on_submit(self.set_vid_fps)

        # video dpi
        self.ax_vid_dpi = plt.axes([0.39, 0.93, 0.05, 0.05])
        self.tb_vid_dpi = TextBox(self.ax_vid_dpi, 'dpi', initial='{:d}'.format(self.vid_dpi))
        self.tb_vid_dpi.on_submit(self.set_vid_dpi)

        # video title
        self.ax_vid_ttl = plt.axes([0.54, 0.93, 0.35, 0.05])
        self.tb_vid_ttl = TextBox(self.ax_vid_ttl, 'file name', initial=self.vid_ttl)
        self.tb_vid_ttl.on_text_change(self.set_vid_ttl)

        self.fig.set_size_inches(8., 5.01)

        # save the ax for blitting
        # self.ax_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # self.fig.draw_artist(self.ax)

    def set_ndc(self, if_ndc, omg_ndc):
        """
        set numeric dispersion correction

        Parameters
        ----------
        if_ndc      :   bool
        omg_ndc     :   float

        Returns
        -------

        """
        c0 = self.c0
        dt = self.dt

        if omg_ndc is None:
            omg_ndc = c0 / .2
        self.omg_ndc = omg_ndc
        if if_ndc:
            n = np.sqrt(self.st.epsi_bg[2, 2] * self.st.mu_bg[2, 2])  # todo: anisotropy case
            self.ndc_45deg = c0 * dt * np.sqrt(2.) / n / self.st.dx * np.sin(omg_ndc / 2. / np.sqrt(2.) * n * self.st.dx / c0) / np.sin(omg_ndc / 2. * dt)
            self.ndc_0deg = c0 * dt / n / self.st.dx * np.sin(omg_ndc / 2. * n * self.st.dx / c0) / np.sin(omg_ndc / 2. * dt)
            self.ndc = (self.ndc_0deg + self.ndc_45deg) / 2.
        else:
            self.ndc = 1.

    def vid_rec(self, event):
        """
        video recording.

        Parameters
        ----------
        event

        Returns
        -------

        """
        if self.if_vid_rec:
            self.vid_wtr.finish()
            self.vid_n += 1
        else:
            self.vid_of = self.vid_ttl + '_{:d}.mp4'.format(self.vid_n)
            self.vid_wtr.setup(fig=self.fig, outfile=self.vid_of, dpi=self.vid_dpi)
        self.if_vid_rec = self.b_vid_rec.get_status()[0]

    def vid_rec_ps(self, event):
        """
        If video recording is paused

        Parameters
        ----------
        event

        Returns
        -------

        """
        self.if_vid_rec_ps = self.b_vid_rec_ps.get_status()[0]

    def set_vid_fps(self, text):
        """
        set video fps

        Parameters
        ----------
        text

        Returns
        -------

        """
        self.vid_fps = eval(text)
        self.vid_wtr.fps = self.vid_fps

    def set_vid_dpi(self, text):
        """
        set the video dpi

        Parameters
        ----------
        text

        Returns
        -------

        """
        self.vid_dpi = eval(text)
        self.vid_wtr.dpi = self.vid_dpi

    def set_vid_ttl(self, text):
        """
        set recording video title

        Parameters
        ----------
        text

        Returns
        -------

        """
        self.vid_ttl = text
        self.vid_n = 0

    def pp(self, event):

        """
        play / pause.

        Parameters
        ----------
        event

        Returns
        -------

        """

        self.psd ^= True
        if not self.psd:
            self.rn()

    def stp(self, event):

        """
        Move forward one step in time.

        Parameters
        ----------
        event

        Returns
        -------

        """

        if self.psd:
            self.t += 1
            t1 = time.process_time()
            self.uf()
            t2 = time.process_time()
            if self.mnts is not None and self.mnts:
                if self.plrz == 'Hz':
                    for mnt in self.mnts:
                        mnt.rnf(self.Hz)
                else:
                    for mnt in self.mnts:
                        mnt.rnf(self.Ez)
            t3 = time.process_time()
            self.up()
            if self.mnts is not None and self.mnts:
                for mnt in self.mnts:
                    mnt.up()
            self.fig.canvas.flush_events()
            t4 = time.process_time()
            if self.if_vid_rec and not self.if_vid_rec_ps:
                self.vid_wtr.grab_frame()
            t5 = time.process_time()
            print('Computation time: {:.5f} | Monitor time: {:.5f} | Plotting time: {:.5f}'.format((t2 - t1), (t3 - t2), (t4 - t3)))

    def k_p(self, event):

        """
        key pressed.

        Parameters
        ----------
        event

        Returns
        -------

        """

        if event.key == 'right':
            self.stp(event)
        elif event.key == ' ':
            self.pp(None)

    def pcm_rndr(self, event):
        """
        pcolormesh render

        Parameters
        ----------
        event

        Returns
        -------

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        if self.plrz == 'Hz':
            self.plotter.plot_sf_intensity_temp_pcm(ax=ax, sf=self.Hz)
        elif self.plrz == 'Ez':
            self.plotter.plot_sf_intensity_temp_pcm(ax=ax, sf=self.Ez)
        else:
            print('What polarization is this?')

        # self.fig.canvas.update()
        plt.show(block=False)

        self.fig_pr_p = fig
        self.ax_pr_p = ax

    def r_f(self, event):

        """
        reset fields

        Parameters
        ----------
        event

        Returns
        -------

        """

        for f in [self.Bx, self.By, self.Bz, self. Hx, self.Hy, self.Hz, self.Dx, self.Dy, self.Dz, self.Ex, self.Ey, self.Ez]:
            f.fill(0.)

        for f in [self.psi_Bzx_V_iP, self.psi_Bzy_V_iP, self.psi_Dxy_V_iP, self.psi_Dyx_V_iP, self.psi_Dzx_V_iP, self.psi_Dzy_V_iP, self.psi_Bxy_V_iP, self.psi_Byx_V_iP]:
            f.fill(0.)

        for JP in [self.JPx_Lz, self.JPy_Lz, self.JPz_Lz]:
            for JP_box in JP:
                for jp in JP_box:
                    jp.fill(0.)
        for P in [self.Px_Lz, self.Py_Lz, self.Pz_Lz]:
            for P_box in P:
                for p in P_box:
                    p.fill(0.)
        for JP in [self.JPx_Dr, self.JPy_Dr, self.JPz_Dr]:
            for JP_box in JP:
                for jp in JP_box:
                    jp.fill(0.)
        for P in [self.Px_Dr, self.Py_Dr, self.Pz_Dr]:
            for P_box in P:
                for p in P_box:
                    p.fill(0.)
        for JP in [self.JPx_dmLz, self.JPy_dmLz, self.JPz_dmLz]:
            for JP_box in JP:
                for jp in JP_box:
                    jp.fill(0.)
        for P in [self.Px_dmLz, self.Py_dmLz, self.Pz_dmLz]:
            for P_box in P:
                for p in P_box:
                    p.fill(0.)
        self.t = 0
        self.up()
        if self.mnts is not None and self.mnts:
            for mnt in self.mnts:
                mnt.cf()

    def rn(self):

        """
        running simulation

        Returns
        -------

        """

        ttt = 0
        while ttt < self.Nt:
            # print(self.t * self.dt)
            if not self.psd:
                t1 = time.perf_counter()
                self.uf()
                t2 = time.perf_counter()
                if self.mnts is not None and self.mnts:
                    if self.plrz == 'Hz':
                        for mnt in self.mnts:
                            mnt.rnf(self.Hz, self.Ex, self.Ey)
                    else:
                        for mnt in self.mnts:
                            mnt.rnf(self.Ez, self.Hx, self.Hy)
                t3 = time.perf_counter()
                if self.t % self.skp == 0:
                    self.up()
                    if self.mnts is not None and self.mnts:
                        for mnt in self.mnts:
                            mnt.up()
                    # plt.pause(0.0001)
                    self.fig.canvas.flush_events()
                t4 = time.perf_counter()
                if self.t % self.skp == 0:
                    if self.if_vid_rec and not self.if_vid_rec_ps:
                        self.vid_wtr.grab_frame()
                t5 = time.perf_counter()
                print('Computation time: {:.5f} | Monitor time: {:.5f} | Plotting time: {:.5f}'.format((t2-t1), (t3-t2), (t4 - t3)))
                ttt += 1
                self.t += 1
            else:
                ttt = self.Nt
                print('paused!')

        self.psd = True

    def uf(self):

        """
        update fields.

        Returns
        -------

        """

        if self.if_pc:  # has PML
            iiP = self.pc.iiPV

            if self.plrz == 'Hz':

                # PML psi fields
                self.psi_Bzx_V_iP = self.pc.bxfViP * self.psi_Bzx_V_iP + self.pc.cxfViP * (self.d.pxf * self.Ey_V)[iiP]
                self.psi_Bzy_V_iP = self.pc.byfViP * self.psi_Bzy_V_iP + self.pc.cyfViP * (self.d.pyf * self.Ex_V)[iiP]

                # update B fields
                self.Bz_V += - self.c0 * self.dt * (self.pc.kixV * (self.d.pxf * self.Ey_V) - self.pc.kiyV * (self.d.pyf * self.Ex_V))
                self.Bz_V[iiP] += - self.c0 * self.dt * (self.psi_Bzx_V_iP - self.psi_Bzy_V_iP)

                # add source to B fields
                for sc in self.scs:
                    sc.us(self.dt * self.t)
                    self.Bz[sc.iB_s] += sc.Bz_s[sc.iB_s]

                # update H fields
                self.Hz = 1. / self.st.mu_bg[2, 2] / self.ndc * self.Bz
                if self.md is 'BHDE':
                    for bx in self.st.bxs:
                        self.Hz[bx.gmt.iib] = 1. / bx.mtr.mu[2, 2] / self.ndc * self.Bz[bx.gmt.iib]
                self.Hz_V = self.Hz.reshape(self.st.Ny * self.st.Nx)

                # Psi fields
                self.psi_Dxy_V_iP = self.pc.bybViP * self.psi_Dxy_V_iP + self.pc.cybViP * (self.d.pyb * self.Hz_V)[iiP]
                self.psi_Dyx_V_iP = self.pc.bxbViP * self.psi_Dyx_V_iP + self.pc.cxbViP * (self.d.pxb * self.Hz_V)[iiP]

                # update D fields
                if not self.if_lg:
                    self.Dx_V += self.c0 * self.dt * (self.pc.kiyV * (self.d.pyb * self.Hz_V))
                    self.Dy_V += - self.c0 * self.dt * (self.pc.kixV * (self.d.pxb * self.Hz_V))
                else:
                    self.Dx_V[self.i_not_lg_V] += self.c0 * self.dt * (self.pc.kiyV * (self.d.pyb * self.Hz_V))[self.i_not_lg_V]
                    self.Dy_V[self.i_not_lg_V] += - self.c0 * self.dt * (self.pc.kixV * (self.d.pxb * self.Hz_V))[self.i_not_lg_V]
                    # loss/gain
                    for bx in self.st.bxs_lg:
                        Dx_V_ib_pv = (1./self.dt - bx.mtr.sgm/2./bx.mtr.epsi[2, 2]) * self.Dx_V[bx.gmt.iibV]
                        self.Dx_V[bx.gmt.iibV] = (1./(1./self.dt + bx.mtr.sgm/2./bx.mtr.epsi[2, 2])) * (self.c0 * (self.pc.kiyV[bx.gmt.iibV] * (self.d.pyb * self.Hz_V)[bx.gmt.iibV]) + Dx_V_ib_pv)
                        Dy_V_ib_pv = (1./self.dt - bx.mtr.sgm/2./bx.mtr.epsi[2, 2]) * self.Dy_V[bx.gmt.iibV]
                        self.Dy_V[bx.gmt.iibV] = (1./(1./self.dt + bx.mtr.sgm/2./bx.mtr.epsi[2, 2])) * ((-self.c0) * (self.pc.kixV[bx.gmt.iibV] * (self.d.pxb * self.Hz_V)[bx.gmt.iibV]) + Dy_V_ib_pv)
                    # dynamic loss/gain modulations
                    for bx in self.st.bxs_dmlg:
                        self.sgm_x[bx.gmt.iib] = bx.mtr.sgm + bx.mtr.m_a * np.cos(-bx.mtr.m_o * (self.t - 1. / 2.) * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                        self.sgm_y[bx.gmt.iib] = bx.mtr.sgm + bx.mtr.m_a * np.cos(-bx.mtr.m_o * (self.t - 1. / 2.) * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                        Dx_V_ib_pv = (1. / self.dt - self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2]) * self.Dx_V[bx.gmt.iibV]
                        self.Dx_V[bx.gmt.iibV] = (1. / (1. / self.dt + self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2])) * (self.c0 * (self.pc.kiyV[bx.gmt.iibV] * (self.d.pyb * self.Hz_V)[bx.gmt.iibV]) + Dx_V_ib_pv)
                        Dy_V_ib_pv = (1. / self.dt - self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2]) * self.Dy_V[bx.gmt.iibV]
                        self.Dy_V[bx.gmt.iibV] = (1. / (1. / self.dt + self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2])) * ((-self.c0) * (self.pc.kixV[bx.gmt.iibV] * (self.d.pxb * self.Hz_V)[bx.gmt.iibV]) + Dy_V_ib_pv)

                # handle PML
                self.Dx_V[iiP] += self.c0 * self.dt * self.psi_Dxy_V_iP
                self.Dy_V[iiP] += - self.c0 * self.dt * self.psi_Dyx_V_iP

                # add source to D fields
                for sc in self.scs:
                    self.Dx[sc.iD_s] += sc.Dx_s[sc.iD_s]
                    self.Dy[sc.iD_s] += sc.Dy_s[sc.iD_s]

                # JP and P fields
                for JPx_b, JPy_b, Px_b, Py_b, bx in zip(self.JPx_Lz, self.JPy_Lz, self.Px_Lz, self.Py_Lz, self.st.bxs_Lz):
                    for JPx, JPy, Px, Py, omg, dlt_epsi, Gm in zip(JPx_b, JPy_b, Px_b, Py_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms):
                        JPx[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPx - omg**2 * Px + dlt_epsi * omg**2 * self.Ex[bx.gmt.iib])
                        JPy[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPy - omg**2 * Py + dlt_epsi * omg**2 * self.Ey[bx.gmt.iib])
                        Px += self.dt * JPx
                        Py += self.dt * JPy
                for JPx_b, JPy_b, Px_b, Py_b, bx in zip(self.JPx_Dr, self.JPy_Dr, self.Px_Dr, self.Py_Dr, self.st.bxs_Dr):
                    for JPx, JPy, Px, Py, omg, Gm in zip(JPx_b, JPy_b, Px_b, Py_b, bx.mtr.omgs_rsn, bx.mtr.Gms):
                        JPx[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPx + omg**2 * self.Ex[bx.gmt.iib])
                        JPy[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPy + omg**2 * self.Ey[bx.gmt.iib])
                        Px += self.dt * JPx
                        Py += self.dt * JPy
                for JPx_b, JPy_b, Px_b, Py_b, bx in zip(self.JPx_dmLz, self.JPy_dmLz, self.Px_dmLz, self.Py_dmLz, self.st.bxs_dmLz):
                    for JPx, JPy, Px, Py, omg, dlt_epsi, Gm, m_a, m_o, m_q, m_p in zip(JPx_b, JPy_b, Px_b, Py_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms, bx.mtr.m_a, bx.mtr.m_o, bx.mtr.m_q, bx.mtr.m_p):
                        dlt_epsi_a = dlt_epsi + m_a * np.cos(-m_o * (self.t - 1. / 2.) * self.dt - m_q[0] * self.st.xx[bx.gmt.iib] - m_q[1] * self.st.yy[bx.gmt.iib] + m_p)
                        JPx[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPx - omg**2 * Px + dlt_epsi_a * omg**2 * self.Ex[bx.gmt.iib])
                        JPy[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPy - omg**2 * Py + dlt_epsi_a * omg**2 * self.Ey[bx.gmt.iib])
                        Px += self.dt * JPx
                        Py += self.dt * JPy

                # update E fields
                self.Ex = 1. / self.st.epsi_bg[0, 0] / self.ndc * self.Dx
                self.Ey = 1. / self.st.epsi_bg[1, 1] / self.ndc * self.Dy
                # simple boxes
                for bx in self.st.bxs_sp:
                    self.Ex[bx.gmt.iib] = 1. / bx.mtr.epsi[0, 0] / self.ndc * (self.Dx[bx.gmt.iib])
                    self.Ey[bx.gmt.iib] = 1. / bx.mtr.epsi[1, 1] / self.ndc * (self.Dy[bx.gmt.iib])
                for bx, Px_b, Py_b in zip((self.st.bxs_Lz+self.st.bxs_Dr+self.st.bxs_dmLz), (self.Px_Lz+self.Px_Dr+self.Px_dmLz), (self.Py_Lz+self.Py_Dr+self.Py_dmLz)):
                    self.Ex[bx.gmt.iib] = 1. / bx.mtr.epsi[0, 0] / self.ndc * (self.Dx[bx.gmt.iib] - sum([p for p in Px_b]))
                    self.Ey[bx.gmt.iib] = 1. / bx.mtr.epsi[1, 1] / self.ndc * (self.Dy[bx.gmt.iib] - sum([p for p in Py_b]))
                for bx in (self.st.bxs_lg + self.st.bxs_dmlg):
                    self.Ex[bx.gmt.iib] = 1. / bx.mtr.epsi[0, 0] / self.ndc * (self.Dx[bx.gmt.iib])
                    self.Ey[bx.gmt.iib] = 1. / bx.mtr.epsi[1, 1] / self.ndc * (self.Dy[bx.gmt.iib])
                # dynamic modulations
                for bx in self.st.bxs_dmri:
                    self.epsi_x_dm[bx.gmt.iib] = bx.mtr.epsi[2, 2] + bx.mtr.m_a * np.cos(-bx.mtr.m_o * self.t * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                    # todo: why all negative sign?
                    self.epsi_y_dm[bx.gmt.iib] = bx.mtr.epsi[2, 2] + bx.mtr.m_a * np.cos(-bx.mtr.m_o * self.t * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                    self.Ey[bx.gmt.iib] = 1. / self.epsi_y_dm[bx.gmt.iib] / self.ndc * self.Dy[bx.gmt.iib]

                self.Ex_V = self.Ex.reshape(self.st.Ny * self.st.Nx)
                self.Ey_V = self.Ey.reshape(self.st.Ny * self.st.Nx)

            elif self.plrz == 'Ez':

                # Psi fields
                self.psi_Bxy_V_iP = self.pc.byfViP * self.psi_Bxy_V_iP + self.pc.cyfViP * (self.d.pyf * self.Ez_V)[iiP]
                self.psi_Byx_V_iP = self.pc.bxfViP * self.psi_Byx_V_iP + self.pc.cxfViP * (self.d.pxf * self.Ez_V)[iiP]

                # update B fields
                self.Bx_V += - self.c0 * self.dt * (self.pc.kiyV * (self.d.pyf * self.Ez_V))
                self.Bx_V[iiP] += - self.c0 * self.dt * self.psi_Bxy_V_iP
                self.By_V += self.c0 * self.dt * (self.pc.kixV * (self.d.pxf * self.Ez_V))
                self.By_V[iiP] += self.c0 * self.dt * self.psi_Byx_V_iP

                # add source to B fields
                for sc in self.scs:
                    sc.us(self.dt * self.t)
                    self.Bx[sc.iB_s] += sc.Bx_s[sc.iB_s]
                    self.By[sc.iB_s] += sc.By_s[sc.iB_s]

                # update H fields
                self.Hx = 1. / self.st.mu_bg[0, 0] / self.ndc * self.Bx
                self.Hy = 1. / self.st.mu_bg[1, 1] / self.ndc * self.By
                if self.md is 'BHDE':
                    for bx in self.st.bxs:
                        self.Hx[bx.gmt.iib] = 1. / bx.mtr.mu[0, 0] / self.ndc * self.Bx[bx.gmt.iib]
                        self.Hy[bx.gmt.iib] = 1. / bx.mtr.mu[0, 0] / self.ndc * self.By[bx.gmt.iib]
                self.Hx_V = self.Hx.reshape(self.st.Ny * self.st.Nx)
                self.Hy_V = self.Hy.reshape(self.st.Ny * self.st.Nx)

                # Psi fields
                self.psi_Dzx_V_iP = self.pc.bxbViP * self.psi_Dzx_V_iP + self.pc.cxbViP * (self.d.pxb * self.Hy_V)[iiP]
                self.psi_Dzy_V_iP = self.pc.bybViP * self.psi_Dzy_V_iP + self.pc.cybViP * (self.d.pyb * self.Hx_V)[iiP]

                # update D fields
                if not self.if_lg:
                    self.Dz_V += self.c0 * self.dt * (self.pc.kixV * (self.d.pxb * self.Hy_V) - self.pc.kiyV * (self.d.pyb * self.Hx_V))
                else:
                    self.Dz_V[self.i_not_lg_V] += self.c0 * self.dt * (self.pc.kixV * (self.d.pxb * self.Hy_V) - self.pc.kiyV * (self.d.pyb * self.Hx_V))[self.i_not_lg_V]
                    # loss/gain
                    for bx in self.st.bxs_lg:
                        Dz_V_ib_pv = (1./self.dt - bx.mtr.sgm/2./bx.mtr.epsi[2, 2]) * self.Dz_V[bx.gmt.iibV]
                        self.Dz_V[bx.gmt.iibV] = (1. / (1./self.dt + bx.mtr.sgm/2./bx.mtr.epsi[2, 2])) * (self.c0 * (self.pc.kixV[bx.gmt.iibV] * (self.d.pxb * self.Hy_V)[bx.gmt.iibV] - self.pc.kiyV[bx.gmt.iibV] * (self.d.pyb * self.Hx_V)[bx.gmt.iibV]) + Dz_V_ib_pv)
                    # dynamic gain/loss modulations
                    for bx in self.st.bxs_dmlg:
                        self.sgm_z[bx.gmt.iib] = bx.mtr.sgm + bx.mtr.m_a * np.cos(-bx.mtr.m_o * (self.t - 1. / 2.) * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                        Dz_V_ib_pv = (1. / self.dt - self.sgm_z_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2]) * self.Dz_V[bx.gmt.iibV]
                        self.Dz_V[bx.gmt.iibV] = (1. / (1. / self.dt + self.sgm_z_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2])) * (self.c0 * (self.pc.kixV[bx.gmt.iibV] * (self.d.pxb * self.Hy_V)[bx.gmt.iibV] - self.pc.kiyV[bx.gmt.iibV] * (self.d.pyb * self.Hx_V)[bx.gmt.iibV]) + Dz_V_ib_pv)

                # handle PML
                self.Dz_V[iiP] += self.c0 * self.dt * (self.psi_Dzx_V_iP - self.psi_Dzy_V_iP)

                # add source to D fields
                for sc in self.scs:
                    self.Dz[sc.iD_s] += sc.Dz_s[sc.iD_s]

                # JP and P fields
                for JPz_b, Pz_b, bx in zip(self.JPz_Lz, self.Pz_Lz, self.st.bxs_Lz):
                    for JPz, Pz, omg, dlt_epsi, Gm in zip(JPz_b, Pz_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms):
                        JPz[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPz - omg**2 * Pz + dlt_epsi * omg**2 * self.Ez[bx.gmt.iib])
                        Pz += self.dt * JPz
                for JPz_b, Pz_b, bx in zip(self.JPz_Dr, self.Pz_Dr, self.st.bxs_Dr):
                    for JPz, Pz, omg, Gm in zip(JPz_b, Pz_b, bx.mtr.omgs_rsn, bx.mtr.Gms):
                        JPz[:] = 1. / (1./self.dt + Gm/2.) * ((1./self.dt - Gm/2.) * JPz + omg**2 * self.Ez[bx.gmt.iib])
                        Pz += self.dt * JPz
                for JPz_b, Pz_b, bx in zip(self.JPz_dmLz, self.Pz_dmLz, self.st.bxs_dmLz):
                    for JPz, Pz, omg, dlt_epsi, Gm, m_a, m_o, m_q, m_p in zip(JPz_b, Pz_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms, bx.mtr.m_a, bx.mtr.m_o, bx.mtr.m_q, bx.mtr.m_p):
                        dlt_epsi_a = dlt_epsi + m_a * np.cos(-m_o * (self.t - 1. / 2.) * self.dt - m_q[0] * self.st.xx[bx.gmt.iib] - m_q[1] * self.st.yy[bx.gmt.iib] + m_p)
                        JPz[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPz - omg ** 2 * Pz + dlt_epsi_a * omg ** 2 * self.Ez[bx.gmt.iib])
                        Pz += self.dt * JPz

                # update E fields
                self.Ez = 1. / self.st.epsi_bg[2, 2] / self.ndc * self.Dz
                # simple boxes
                for bx in (self.st.bxs_sp):
                    self.Ez[bx.gmt.iib] = 1. / bx.mtr.epsi[2, 2] / self.ndc * (self.Dz[bx.gmt.iib])
                for bx, Pz_b, in zip((self.st.bxs_Lz+self.st.bxs_Dr+self.st.bxs_dmLz), (self.Pz_Lz + self.Pz_Dr + self.Pz_dmLz)):
                    self.Ez[bx.gmt.iib] = 1. / bx.mtr.epsi[2, 2] / self.ndc * (self.Dz[bx.gmt.iib] - sum([p for p in Pz_b]))
                for bx in (self.st.bxs_lg + self.st.bxs_dmlg):
                    self.Ez[bx.gmt.iib] = 1. / bx.mtr.epsi[2, 2] / self.ndc * (self.Dz[bx.gmt.iib])
                # dynamic modulations
                for bx in self.st.bxs_dmri:
                    self.epsi_z_dm[bx.gmt.iib] = bx.mtr.epsi[2, 2] + bx.mtr.m_a * np.cos(-bx.mtr.m_o * self.t * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                    self.Ez[bx.gmt.iib] = 1. / self.epsi_z_dm[bx.gmt.iib] / self.ndc * self.Dz[bx.gmt.iib]

                self.Ez_V = self.Ez.ravel()

        else:  # no PML

            if self.plrz == 'Hz':

                # update B fields
                self.Bz_V += - self.c0 * self.dt * ((self.d.pxf * self.Ey_V) - (self.d.pyf * self.Ex_V))

                # add source to B fields
                for sc in self.scs:
                    sc.us(self.dt * self.t)
                    self.Bz[sc.iB_s] += sc.Bz_s[sc.iB_s]

                # update H fields
                self.Hz = 1. / self.st.mu_bg[2, 2] / self.ndc * self.Bz
                if self.md is 'BHDE':
                    for bx in self.st.bxs:
                        self.Hz[bx.gmt.iib] = 1. / bx.mtr.mu[2, 2] / self.ndc * self.Bz[bx.gmt.iib]
                self.Hz_V = self.Hz.reshape(self.st.Ny * self.st.Nx)

                # update D fields
                if not self.if_lg:
                    self.Dx_V += self.c0 * self.dt * ((self.d.pyb * self.Hz_V))
                    self.Dy_V += - self.c0 * self.dt * ((self.d.pxb * self.Hz_V))
                else:
                    self.Dx_V[self.i_not_lg_V] += self.c0 * self.dt * ((self.d.pyb * self.Hz_V))[self.i_not_lg_V]
                    self.Dy_V[self.i_not_lg_V] += - self.c0 * self.dt * ((self.d.pxb * self.Hz_V))[self.i_not_lg_V]
                    # loss/gain
                    for bx in self.st.bxs_lg:
                        Dx_V_ib_pv = (1. / self.dt - bx.mtr.sgm / 2. / bx.mtr.epsi[2, 2]) * self.Dx_V[bx.gmt.iibV]
                        self.Dx_V[bx.gmt.iibV] = (1. / (1. / self.dt + bx.mtr.sgm / 2. / bx.mtr.epsi[2, 2])) * (self.c0 * ((self.d.pyb * self.Hz_V)[bx.gmt.iibV]) + Dx_V_ib_pv)
                        Dy_V_ib_pv = (1. / self.dt - bx.mtr.sgm / 2. / bx.mtr.epsi[2, 2]) * self.Dy_V[bx.gmt.iibV]
                        self.Dy_V[bx.gmt.iibV] = (1. / (1. / self.dt + bx.mtr.sgm / 2. / bx.mtr.epsi[2, 2])) * ((-self.c0) * ((self.d.pxb * self.Hz_V)[bx.gmt.iibV]) + Dy_V_ib_pv)
                    # dynamic loss/gain modulations
                    for bx in self.st.bxs_dmlg:
                        self.sgm_x[bx.gmt.iib] = bx.mtr.sgm + bx.mtr.m_a * np.cos(-bx.mtr.m_o * (self.t - 1. / 2.) * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                        self.sgm_y[bx.gmt.iib] = bx.mtr.sgm + bx.mtr.m_a * np.cos(-bx.mtr.m_o * (self.t - 1. / 2.) * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                        Dx_V_ib_pv = (1. / self.dt - self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2]) * self.Dx_V[bx.gmt.iibV]
                        self.Dx_V[bx.gmt.iibV] = (1. / (1. / self.dt + self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2])) * (self.c0 * ((self.d.pyb * self.Hz_V)[bx.gmt.iibV]) + Dx_V_ib_pv)
                        Dy_V_ib_pv = (1. / self.dt - self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2]) * self.Dy_V[bx.gmt.iibV]
                        self.Dy_V[bx.gmt.iibV] = (1. / (1. / self.dt + self.sgm_x_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2])) * ((-self.c0) * ((self.d.pxb * self.Hz_V)[bx.gmt.iibV]) + Dy_V_ib_pv)

                # add source to D fields
                for sc in self.scs:
                    self.Dx[sc.iD_s] += sc.Dx_s[sc.iD_s]
                    self.Dy[sc.iD_s] += sc.Dy_s[sc.iD_s]

                # JP and P fields
                for JPx_b, JPy_b, Px_b, Py_b, bx in zip(self.JPx_Lz, self.JPy_Lz, self.Px_Lz, self.Py_Lz, self.st.bxs_Lz):
                    for JPx, JPy, Px, Py, omg, dlt_epsi, Gm in zip(JPx_b, JPy_b, Px_b, Py_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms):
                        JPx[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPx - omg ** 2 * Px + dlt_epsi * omg ** 2 * self.Ex[bx.gmt.iib])
                        JPy[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPy - omg ** 2 * Py + dlt_epsi * omg ** 2 * self.Ey[bx.gmt.iib])
                        Px += self.dt * JPx
                        Py += self.dt * JPy
                for JPx_b, JPy_b, Px_b, Py_b, bx in zip(self.JPx_Dr, self.JPy_Dr, self.Px_Dr, self.Py_Dr, self.st.bxs_Dr):
                    for JPx, JPy, Px, Py, omg, Gm in zip(JPx_b, JPy_b, Px_b, Py_b, bx.mtr.omgs_rsn, bx.mtr.Gms):
                        JPx[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPx + omg ** 2 * self.Ex[bx.gmt.iib])
                        JPy[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPy + omg ** 2 * self.Ey[bx.gmt.iib])
                        Px += self.dt * JPx
                        Py += self.dt * JPy
                for JPx_b, JPy_b, Px_b, Py_b, bx in zip(self.JPx_dmLz, self.JPy_dmLz, self.Px_dmLz, self.Py_dmLz, self.st.bxs_dmLz):
                    for JPx, JPy, Px, Py, omg, dlt_epsi, Gm, m_a, m_o, m_q, m_p in zip(JPx_b, JPy_b, Px_b, Py_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms, bx.mtr.m_a, bx.mtr.m_o, bx.mtr.m_q, bx.mtr.m_p):
                        dlt_epsi_a = dlt_epsi + m_a * np.cos(-m_o * (self.t - 1. / 2.) * self.dt - m_q[0] * self.st.xx[bx.gmt.iib] - m_q[1] * self.st.yy[bx.gmt.iib] + m_p)
                        JPx[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPx - omg ** 2 * Px + dlt_epsi_a * omg ** 2 * self.Ex[bx.gmt.iib])
                        JPy[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPy - omg ** 2 * Py + dlt_epsi_a * omg ** 2 * self.Ey[bx.gmt.iib])
                        Px += self.dt * JPx
                        Py += self.dt * JPy

                # update E fields
                self.Ex = 1. / self.st.epsi_bg[0, 0] / self.ndc * self.Dx
                self.Ey = 1. / self.st.epsi_bg[1, 1] / self.ndc * self.Dy
                # simple boxes
                for bx in self.st.bxs_sp:
                    self.Ex[bx.gmt.iib] = 1. / bx.mtr.epsi[0, 0] / self.ndc * (self.Dx[bx.gmt.iib])
                    self.Ey[bx.gmt.iib] = 1. / bx.mtr.epsi[1, 1] / self.ndc * (self.Dy[bx.gmt.iib])
                for bx, Px_b, Py_b in zip((self.st.bxs_Lz + self.st.bxs_Dr + self.st.bxs_dmLz), (self.Px_Lz + self.Px_Dr + self.Px_dmLz), (self.Py_Lz + self.Py_Dr + self.Py_dmLz)):
                    self.Ex[bx.gmt.iib] = 1. / bx.mtr.epsi[0, 0] / self.ndc * (self.Dx[bx.gmt.iib] - sum([p for p in Px_b]))
                    self.Ey[bx.gmt.iib] = 1. / bx.mtr.epsi[1, 1] / self.ndc * (self.Dy[bx.gmt.iib] - sum([p for p in Py_b]))
                for bx in (self.st.bxs_lg + self.st.bxs_dmlg):
                    self.Ex[bx.gmt.iib] = 1. / bx.mtr.epsi[0, 0] / self.ndc * (self.Dx[bx.gmt.iib])
                    self.Ey[bx.gmt.iib] = 1. / bx.mtr.epsi[1, 1] / self.ndc * (self.Dy[bx.gmt.iib])
                # dynamic modulations
                for bx in self.st.bxs_dmri:
                    self.epsi_x_dm[bx.gmt.iib] = bx.mtr.epsi[2, 2] + bx.mtr.m_a * np.cos(-bx.mtr.m_o * self.t * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                    # todo: why all negative sign?
                    self.epsi_y_dm[bx.gmt.iib] = bx.mtr.epsi[2, 2] + bx.mtr.m_a * np.cos(-bx.mtr.m_o * self.t * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                    self.Ey[bx.gmt.iib] = 1. / self.epsi_y_dm[bx.gmt.iib] / self.ndc * self.Dy[bx.gmt.iib]

                self.Ex_V = self.Ex.reshape(self.st.Ny * self.st.Nx)
                self.Ey_V = self.Ey.reshape(self.st.Ny * self.st.Nx)

            elif self.plrz == 'Ez':

                # update B fields
                self.Bx_V += - self.c0 * self.dt * ((self.d.pyf * self.Ez_V))
                self.By_V += self.c0 * self.dt * ((self.d.pxf * self.Ez_V))

                # add source to B fields
                for sc in self.scs:
                    sc.us(self.dt * self.t)
                    self.Bx[sc.iB_s] += sc.Bx_s[sc.iB_s]
                    self.By[sc.iB_s] += sc.By_s[sc.iB_s]

                # update H fields
                self.Hx = 1. / self.st.mu_bg[0, 0] / self.ndc * self.Bx
                self.Hy = 1. / self.st.mu_bg[1, 1] / self.ndc * self.By
                if self.md is 'BHDE':
                    for bx in self.st.bxs:
                        self.Hx[bx.gmt.iib] = 1. / bx.mtr.mu[0, 0] / self.ndc * self.Bx[bx.gmt.iib]
                        self.Hy[bx.gmt.iib] = 1. / bx.mtr.mu[0, 0] / self.ndc * self.By[bx.gmt.iib]
                self.Hx_V = self.Hx.reshape(self.st.Ny * self.st.Nx)
                self.Hy_V = self.Hy.reshape(self.st.Ny * self.st.Nx)

                # update D fields
                if not self.if_lg:
                    self.Dz_V += self.c0 * self.dt * ((self.d.pxb * self.Hy_V) - (self.d.pyb * self.Hx_V))
                else:
                    self.Dz_V[self.i_not_lg_V] += self.c0 * self.dt * ((self.d.pxb * self.Hy_V) - (self.d.pyb * self.Hx_V))[self.i_not_lg_V]
                    # loss/gain
                    for bx in self.st.bxs_lg:
                        Dz_V_ib_pv = (1. / self.dt - bx.mtr.sgm / 2. / bx.mtr.epsi[2, 2]) * self.Dz_V[bx.gmt.iibV]
                        self.Dz_V[bx.gmt.iibV] = (1. / (1. / self.dt + bx.mtr.sgm / 2. / bx.mtr.epsi[2, 2])) * (
                                    self.c0 * ((self.d.pxb * self.Hy_V)[bx.gmt.iibV] - (self.d.pyb * self.Hx_V)[bx.gmt.iibV]) + Dz_V_ib_pv)
                    # dynamic gain/loss modulations
                    for bx in self.st.bxs_dmlg:
                        self.sgm_z[bx.gmt.iib] = bx.mtr.sgm + bx.mtr.m_a * np.cos(-bx.mtr.m_o * (self.t - 1. / 2.) * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                        Dz_V_ib_pv = (1. / self.dt - self.sgm_z_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2]) * self.Dz_V[bx.gmt.iibV]
                        self.Dz_V[bx.gmt.iibV] = (1. / (1. / self.dt + self.sgm_z_V[bx.gmt.iibV] / 2. / bx.mtr.epsi[2, 2])) * (
                                    self.c0 * ((self.d.pxb * self.Hy_V)[bx.gmt.iibV] - (self.d.pyb * self.Hx_V)[bx.gmt.iibV]) + Dz_V_ib_pv)

                # add source to D fields
                for sc in self.scs:
                    self.Dz[sc.iD_s] += sc.Dz_s[sc.iD_s]

                # JP and P fields
                for JPz_b, Pz_b, bx in zip(self.JPz_Lz, self.Pz_Lz, self.st.bxs_Lz):
                    for JPz, Pz, omg, dlt_epsi, Gm in zip(JPz_b, Pz_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms):
                        JPz[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPz - omg ** 2 * Pz + dlt_epsi * omg ** 2 * self.Ez[bx.gmt.iib])
                        Pz += self.dt * JPz
                for JPz_b, Pz_b, bx in zip(self.JPz_Dr, self.Pz_Dr, self.st.bxs_Dr):
                    for JPz, Pz, omg, Gm in zip(JPz_b, Pz_b, bx.mtr.omgs_rsn, bx.mtr.Gms):
                        JPz[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPz + omg ** 2 * self.Ez[bx.gmt.iib])
                        Pz += self.dt * JPz
                for JPz_b, Pz_b, bx in zip(self.JPz_dmLz, self.Pz_dmLz, self.st.bxs_dmLz):
                    for JPz, Pz, omg, dlt_epsi, Gm, m_a, m_o, m_q, m_p in zip(JPz_b, Pz_b, bx.mtr.omgs_rsn, bx.mtr.dlts_epsi, bx.mtr.Gms, bx.mtr.m_a, bx.mtr.m_o, bx.mtr.m_q, bx.mtr.m_p):
                        dlt_epsi_a = dlt_epsi + m_a * np.cos(-m_o * (self.t - 1. / 2.) * self.dt - m_q[0] * self.st.xx[bx.gmt.iib] - m_q[1] * self.st.yy[bx.gmt.iib] + m_p)
                        JPz[:] = 1. / (1. / self.dt + Gm / 2.) * ((1. / self.dt - Gm / 2.) * JPz - omg ** 2 * Pz + dlt_epsi_a * omg ** 2 * self.Ez[bx.gmt.iib])
                        Pz += self.dt * JPz

                # update E fields
                self.Ez = 1. / self.st.epsi_bg[2, 2] / self.ndc * self.Dz
                # simple boxes
                for bx in (self.st.bxs_sp):
                    self.Ez[bx.gmt.iib] = 1. / bx.mtr.epsi[2, 2] / self.ndc * (self.Dz[bx.gmt.iib])
                for bx, Pz_b, in zip((self.st.bxs_Lz + self.st.bxs_Dr + self.st.bxs_dmLz), (self.Pz_Lz + self.Pz_Dr + self.Pz_dmLz)):
                    self.Ez[bx.gmt.iib] = 1. / bx.mtr.epsi[2, 2] / self.ndc * (self.Dz[bx.gmt.iib] - sum([p for p in Pz_b]))
                for bx in (self.st.bxs_lg + self.st.bxs_dmlg):
                    self.Ez[bx.gmt.iib] = 1. / bx.mtr.epsi[2, 2] / self.ndc * (self.Dz[bx.gmt.iib])
                # dynamic modulations
                for bx in self.st.bxs_dmri:
                    self.epsi_z_dm[bx.gmt.iib] = bx.mtr.epsi[2, 2] + bx.mtr.m_a * np.cos(-bx.mtr.m_o * self.t * self.dt - bx.mtr.m_q[0] * self.st.xx[bx.gmt.iib] - bx.mtr.m_q[1] * self.st.yy[bx.gmt.iib] + bx.mtr.m_p)
                    self.Ez[bx.gmt.iib] = 1. / self.epsi_z_dm[bx.gmt.iib] / self.ndc * self.Dz[bx.gmt.iib]

                self.Ez_V = self.Ez.ravel()

    def up(self):

        """
        update plots.

        Returns
        -------

        """

        # for blitting, first restore
        # self.ax.figure.canvas.restore_region(self.ax_cache)

        if self.plrz == 'Hz':
            # self.sf_pc.set_array(self.Hz)
            self.plotter.update_sf_intensity_plot(self.Hz)
        elif self.plrz == 'Ez':
            # self.sf_pc.set_array(self.Ez)
            self.plotter.update_sf_intensity_plot(self.Ez)
        else:
            print('What polarization is this?')

        # self.ax.draw_artist(self.sf_pc)
        self.an_t.set_text('time step : {:d}  '.format(self.t))

        # Re-plot the time step annotation window
        self.ax.draw_artist(self.an_t)
        for phc in self.phcs:
            self.ax.draw_artist(phc)

        self.fig.canvas.update()

        # blitting
        # self.ax.figure.canvas.blit(self.ax.bbox)
