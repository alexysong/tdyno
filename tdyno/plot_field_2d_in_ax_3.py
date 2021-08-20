#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song, June 2017

"""


import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


class PlotField2DinAx:

    def __init__(self,
                 ax,
                 title=None, xlabel='$x$', ylabel='$y$', fontsize_label=20, fontsize_axis=16, cmap=None, dire='v', invert_y=False, tick_spacing=None,
                 xmin=None, xmax=None, xres=None, ymin=None, ymax=None, yres=None,
                 x=None, y=None,
                 sf=None, if_scalar_field_intensity=False, scalar_field_part='magn', if_colorbar=True, vmin=None, vmax=None, if_phasor=False,
                 vfx=None, vfy=None, if_vector_field=False, vector_field_part='real',
                 qv_step=1, qv_power=1., qv_scale=1., qv_cap_magn=None,
                 quiverwidth=None, headwidth=None, headlength=None, headaxislength=None,
                 shaded_region=None, shaded_region_text=None, if_update_shading=True
                 ):
        """
        Plotting and updating of 2D field in an axes.

        Parameters
        ----------
        title                                                   :   str
        xlabel, ylabel                                          :   str
        cmap                                                    :   str
                                                                    colormap to use.
        dire                                                    :   str
                                                                    [ 'v' | 'h' ]   horizontal or vertical. Vertical means y axis is y, x axis is x.
        invert_y                                                :   bool
                                                                    if True, invert the y axis in the figure.
        tick_spacing                                            :   float
                                                                    optional, adjust the tick spacing in the main plot.
        xmin, xmax, xres, ymin, ymax, yres                      :   float
        x, y                                                    :   array
                                                                    each is a 1d array. x and y coordinates. If given, overrides xim etc.
        sf                                                      :   complex 2d ndarray
                                                                    The scalar field to plot.
        if_scalar_field_intensity, if_phasor, if_vector_field   :   bool
                                                                    if True, plot pcolormesh of scalar field (can choose real, imag, or magn in scalar_field_part)/ phasor of scalar field / vector field
        scalar_field_part                                       :   str
                                                                    ['real' | 'imag' | 'magn']  choose which part of scalar field to plot
        if_colorbar                                             :   bool
        vmin, vmax                                              :   float
                                                                    he maximum data range for colormap for colorbar.
        vfx, vfy                                                :   complex 2d ndarray
                                                                    Complex vector field to plot.
        vector_field_part                                       :   str
                                                                    [ 'real' | 'imag' ]   plot the real or imaginary part
        qv_step                                                 :   int
        qv_power                                                :   float
        qv_cap_magn                                             :   float
        qv_scale                                                :   float
        shaded_region                                           :   ndarray
                                                                    optional bool indexing array to a region that will be semi-transparent grey shaded.
        shaded_region_text                                      :   str
                                                                    If supplied, will show this text marking the shaded region.
        if_update_shading                                       :   bool
                                                                    if or not re-plot shading each time you update the plots.
        quiverwidth, headwidth, headlength, headaxislength      :   float
                                                                    parameters for the quiver plot arrow shape


        Methods
        -------
        update_plot
        """

        self.ax = ax
        self.dire = dire
        self.invert_y = invert_y

        self.qv_step = qv_step
        self.qv_power = qv_power
    
        self.if_scalar_field_intensity = if_scalar_field_intensity
        self.if_phasor = if_phasor
        self.if_vector_field = if_vector_field
        self.scalar_field_part = scalar_field_part
        self.vector_field_part = vector_field_part
        
        # fonts used
        mpl.rcParams['mathtext.fontset'] = 'cm'
        my_font = 'Arial'
        self.my_font = my_font

        cmap = self.define_cmap(cmap)
        self.cmap = cmap

        self.x = x
        self.y = y
        self.xres = xres
        self.xmin = xmin
        self.xmax = xmax
        self.yres = yres
        self.ymin = ymin
        self.ymax = ymax
        self.XX_pcm = None
        self.YY_pcm = None
        self.XX_qv = None
        self.YY_qv = None

        if (xres or xmin or xmax or yres or ymin or ymax) or (x or y):
            self.set_xy(xmin=xmin, xmax=xmax, xres=xres, ymin=ymin, ymax=ymax, yres=yres, x=x, y=y, qv_step=qv_step)

        if self.dire == 'v':
            Xlabel = xlabel
            Ylabel = ylabel
        else:
            Xlabel = ylabel
            Ylabel = xlabel
        self.Xlabel = Xlabel
        self.Ylabel = Ylabel

        self.fontsize_label = fontsize_label
        self.fontsize_axis = fontsize_axis

        title = title
        if title is not None:
            ax.set_title(title, fontname=my_font, fontsize=fontsize_label)
        ax.set_aspect('equal')
        ax.set_xlabel(Xlabel, fontname=my_font, fontsize=fontsize_label)
        ax.set_ylabel(Ylabel, fontname=my_font, fontsize=fontsize_label)
        ax.tick_params(axis='both', which='major', labelsize=fontsize_axis)

        self.sf_pc = None
        self.vmin = vmin
        self.vmax = vmax

        self.if_colorbar = if_colorbar
        self.cb = None
        self.cax = None
        if if_colorbar:
            divider = make_axes_locatable(self.ax)
            self.cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cax.tick_params(axis='y', which='major', labelsize=self.fontsize_axis)

        if if_scalar_field_intensity and (sf is not None):
            self.plot_sf_intensity(sf=sf, scalar_field_part=scalar_field_part, cmap=cmap, vmin=vmin, vmax=vmax)

        self.qv_step = qv_step
        self.qv_power = qv_power
        self.qv_scale = qv_scale
        self.qv_cap_magn = qv_cap_magn
        self.quiverwidth = quiverwidth
        self.headwidth = headwidth
        self.headlength = headlength
        self.headaxislength = headaxislength

        self.sf_qv_phasor = None
        if if_phasor and (sf is not None):
            self.plot_sf_phasor(sf=sf, qv_step=qv_step, qv_power=qv_power, qv_scale=qv_scale, qv_cap_magn=qv_cap_magn,quiverwidth=quiverwidth, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)

        self.vf_qv = None
        if if_vector_field and (vfx is not None) and (vfy is not None):
            self.plot_vf_qv(vfx=vfx, vfy=vfy, vector_field_part=vector_field_part, qv_step=qv_step, qv_power=qv_power, qv_scale=qv_scale, qv_cap_magn=qv_cap_magn,quiverwidth=quiverwidth, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)

        self.pc_shaded_region = None
        if (shaded_region is not None) and shaded_region.max():
            cmap = plt.cm.gist_yarg_r
            my_transparency_cmap = cmap(np.arange(cmap.N))
            my_transparency_cmap[:, -1] = np.linspace(1., 0., cmap.N)
            my_transparency_cmap = ListedColormap(my_transparency_cmap)
            self.pc_shaded_region = self.ax.pcolorfast([xmin, xmax], [ymin, ymax], (1. - shaded_region), alpha=.1, cmap=my_transparency_cmap, zorder=3)
            if shaded_region_text is not None:
                self.ax.annotate(shaded_region_text, fontsize=9, xy=(0., 0.), xycoords='data', xytext=(0.06, 0.96), textcoords='axes fraction', color=[0., 0., 0., 0.5])
        self.if_update_shading = if_update_shading

        if invert_y:
            ax.invert_yaxis()

        if tick_spacing is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        for tick in ax.get_xticklabels():
            tick.set_fontname(my_font)
        for tick in ax.get_yticklabels():
            tick.set_fontname(my_font)
        if self.cax is not None:
            for tick in self.cax.get_xticklabels():
                tick.set_fontname(my_font)
            for tick in self.cax.get_yticklabels():
                tick.set_fontname(my_font)

    def define_cmap(self, cmap):
        # pass
        return cmap

    def set_xy(self, xmin=None, xmax=None, xres=None, ymin=None, ymax=None, yres=None, x=None, y=None, qv_step=None):
        """
        Set x and y coordinates. If x and y are supplied, use them to override xmin etc. Otherwise, use xmin, etc. to generate x and y.

        Parameters
        ----------
        xmin
        xmax
        xres
        ymin
        ymax
        yres
        x
        y

        Returns
        -------

        """
        if qv_step is not None:
            self.qv_step = qv_step
        else:
            qv_step = self.qv_step

        if (x is not None) or (y is not None) or (xres is not None) or (xmin is not None) or (xmax is not None) or (yres is not None) or (ymin is not None) or (ymax is not None):
            if (x is not None) or (y is not None):
                xres = x[1] - x[0]
                xmin = x[0]
                xmax = x[-1] + xres
                yres = y[1] - y[0]
                ymin = y[0]
                ymax = y[-1] + yres
            else:
                Nx = int((xmax + xres / 1.e4 - xmin) / xres)
                Ny = int((ymax + yres / 1.e4 - ymin) / yres)
                x_n = np.arange(Nx, dtype=int)
                y_n = np.arange(Ny, dtype=int)
                # xx_n, yy_n = np.meshgrid(x_n, y_n)
                x = x_n * xres + xmin
                y = y_n * yres + ymin

            xx, yy = np.meshgrid(x, y)

            xx_qv, yy_qv = np.meshgrid(x[::qv_step], y[::qv_step])
            if self.dire == 'v':
                XX_pcm = xx
                YY_pcm = yy
                XX_qv = xx_qv
                YY_qv = yy_qv
            else:
                XX_pcm = yy
                YY_pcm = xx
                XX_qv = yy_qv
                YY_qv = xx_qv

            self.x = x
            self.y = y
            self.xres = xres
            self.xmin = xmin
            self.xmax = xmax
            self.yres = yres
            self.ymin = ymin
            self.ymax = ymax
            self.XX_pcm = XX_pcm
            self.YY_pcm = YY_pcm
            self.XX_qv = XX_qv
            self.YY_qv = YY_qv

    def plot_sf_intensity(self, sf=None, scalar_field_part=None, cmap=None, vmin=None, vmax=None, **kwargs):
        """
        Plot sf field intensity.

        Parameters
        ----------
        sf
        scalar_field_part
        cmap
        vmin
        vmax

        Keyword Arguments
        -----------------
        xmin
        xmax
        xres
        ymin
        ymax
        yres
        x
        y


        Returns
        -------

        """
        self.if_scalar_field_intensity = True

        self.set_xy(**kwargs)

        if cmap is not None:
            cmap = self.define_cmap(cmap)
            self.cmap = cmap
        else:
            cmap = self.cmap

        if scalar_field_part is None:
            scalar_field_part = self.scalar_field_part
        else:
            self.scalar_field_part = scalar_field_part
        if scalar_field_part == 'real':
            sf_plot = np.real(sf)
            cmap = cmap or 'RdBu_r'
        elif scalar_field_part == 'imag':
            sf_plot = np.imag(sf)
            cmap = cmap or 'RdBu_r'
        else:
            sf_plot = np.abs(sf)
            cmap = cmap or 'afmhot'

        if vmin is not None:
            self.vmin = vmin
        else:
            vmin = self.vmin

        if vmax is not None:
            self.vmax = vmax
        else:
            vmax = self.vmax

        # self.sf_pc = self.ax.pcolormesh(self.XX_pcm, self.YY_pcm, sf_plot, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
        self.sf_pc = self.ax.pcolorfast([self.xmin, self.xmax], [self.ymin, self.ymax], sf_plot, vmin=vmin, vmax=vmax, cmap=cmap)

        if self.if_colorbar:
            cb = plt.colorbar(self.sf_pc, cax=self.cax)
            # # for phase plot, use pi as unit
            # self.cb = plt.colorbar(self.sf_pc, cax=cax, ticks=[0, np.pi/4, np.pi/2, np.pi*3/4, np.pi])
            # self.cb.ax.set_yticklabels(['$0$', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', '$\pi$'])
        else:
            cb = None
        self.cb = cb

    def plot_sf_intensity_temp_pcm(self, ax=None, cax=None, sf=None, scalar_field_part=None, cmap=None, vmin=None, vmax=None, **kwargs):
        """
        Plot sf field intensity using pcolormesh (with gouraud shading) temporarily.

        Parameters
        ----------
        sf
        scalar_field_part
        cmap
        vmin
        vmax

        Keyword Arguments
        -----------------
        xmin
        xmax
        xres
        ymin
        ymax
        yres
        x
        y


        Returns
        -------

        """
        # self.if_scalar_field_intensity = True

        if ax is None:
            ax = self.ax

        self.set_xy(**kwargs)

        if cmap is not None:
            cmap = self.define_cmap(cmap)
        else:
            cmap = self.cmap

        if scalar_field_part is None:
            scalar_field_part = self.scalar_field_part
        if scalar_field_part == 'real':
            sf_plot = np.real(sf)
            cmap = cmap or 'RdBu_r'
        elif scalar_field_part == 'imag':
            sf_plot = np.imag(sf)
            cmap = cmap or 'RdBu_r'
        else:
            sf_plot = np.abs(sf)
            cmap = cmap or 'afmhot'

        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        pcm = ax.pcolormesh(self.XX_pcm, self.YY_pcm, sf_plot, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
        # self.ax.pcolorfast([self.xmin, self.xmax], [self.ymin, self.ymax], sf_plot, vmin=vmin, vmax=vmax, cmap=cmap)

        if cax is not None:
            cb = plt.colorbar(pcm, cax=cax)
            # # for phase plot, use pi as unit
            # self.cb = plt.colorbar(self.sf_pc, cax=cax, ticks=[0, np.pi/4, np.pi/2, np.pi*3/4, np.pi])
            # self.cb.ax.set_yticklabels(['$0$', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', '$\pi$'])
        else:
            cb = None

        return pcm, cb

    def update_sf_intensity_plot(self, sf):

        if self.scalar_field_part == 'real':
            f_plot = sf.real
        elif self.scalar_field_part == 'imag':
            f_plot = sf.imag
        else:
            f_plot = np.abs(sf)
        self.sf_pc.set_array(f_plot)
        # self.sf_pc.set_array(f_plot.ravel())
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.sf_pc)
        if self.if_update_shading and (self.pc_shaded_region is not None):
            self.ax.draw_artist(self.pc_shaded_region)

    def calc_sf_phasor_data(self, sf=None, qv_step=None, qv_power=None, qv_cap_magn=None):
        f_qv = sf[::qv_step, ::qv_step]
        fa_qv = np.abs(f_qv)
        fa_qv_power = np.power(fa_qv, qv_power)
        coef = fa_qv_power / fa_qv
        f_qv = f_qv * coef

        if qv_cap_magn is not None:
            fa_qv = np.abs(f_qv)
            too_large = fa_qv > qv_cap_magn
            f_qv[too_large] = f_qv[too_large] / fa_qv[too_large] * qv_cap_magn

        fr_qv = f_qv.real
        fi_qv = f_qv.imag

        return fr_qv, fi_qv

    def plot_sf_phasor(self, sf=None, qv_step=None, qv_power=None, qv_scale=None, qv_cap_magn=None, quiverwidth=None, headwidth=None, headlength=None, headaxislength=None, **kwargs):
        """

        Parameters
        ----------
        sf
        qv_step
        qv_power
        qv_scale
        qv_cap_magn
        quiverwidth
        headwidth
        headlength
        headaxislength

        Keyword Arguments
        -----------------
        xmin
        xmax
        xres
        ymin
        ymax
        yres
        x
        y

        Returns
        -------

        """

        self.if_phasor = True

        self.set_xy(**kwargs)

        if qv_step is None:
            qv_step = self.qv_step
        else:
            self.qv_step = qv_step
        if qv_power is None:
            qv_power = self.qv_power
        else:
            self.qv_power = qv_power
        if qv_cap_magn is None:
            qv_cap_magn = self.qv_cap_magn
        else:
            self.qv_cap_magn = qv_cap_magn
        if quiverwidth is None:
            quiverwidth = self.quiverwidth
        else:
            self.quiverwidth = quiverwidth
        if headwidth is None:
            headwidth = self.headwidth
        else:
            self.headwidth = headwidth
        if headlength is None:
            headlength = self.headlength
        else:
            self.headlength = headlength
        if headaxislength is None:
            headaxislength = self.headaxislength
        else:
            self.headaxislength = headaxislength

        fr_qv, fi_qv = self.calc_sf_phasor_data(sf=sf, qv_step=qv_step, qv_power=qv_power, qv_cap_magn=qv_cap_magn)

        self.sf_qv_phasor = self.ax.quiver(self.XX_qv, self.YY_qv, fr_qv, fi_qv, color=[1., .9, .9, 1.], scale_units='x', scale=(qv_cap_magn / max(self.xres, self.yres) / qv_step * qv_scale), units='x', width=0.075 * max(self.xres, self.yres) * qv_step)

    def update_phasor_plot(self, sf):

        fr_qv, fi_qv = self.calc_sf_phasor_data(sf=sf, qv_step=self.qv_step, qv_power=self.qv_power, qv_cap_magn=self.qv_cap_magn)
        self.sf_qv_phasor.set_UVC(fr_qv, fi_qv)
        self.ax.draw_artist(self.sf_qv_phasor)
        if self.if_update_shading and (self.pc_shaded_region is not None):
            self.ax.draw_artist(self.pc_shaded_region)

    def calc_vf_qv_data(self, vfx, vfy, vector_field_part=None, qv_step=None, qv_power=None, qv_cap_magn=None, dire=None, invert_y=None):

        if vector_field_part == 'real':
            fx_qv, fy_qv = [np.real(ff) for ff in [vfx, vfy]]
        else:
            fx_qv, fy_qv = [np.imag(ff) for ff in [vfx, vfy]]
        fx_qv, fy_qv = [ff[::qv_step, ::qv_step] for ff in [fx_qv, fy_qv]]
        fa_qv = np.sqrt(np.square(fx_qv) + np.square(fy_qv))
        fa_qv_power = np.power(fa_qv, qv_power)
        coef = fa_qv_power / fa_qv
        fx_qv, fy_qv = [ff * coef for ff in [fx_qv, fy_qv]]

        if qv_cap_magn is not None:
            fa_qv = np.sqrt(np.square(fx_qv) + np.square(fy_qv))
            too_large = fa_qv > qv_cap_magn
            fx_qv[too_large] = fx_qv[too_large] / fa_qv[too_large] * qv_cap_magn
            fy_qv[too_large] = fy_qv[too_large] / fa_qv[too_large] * qv_cap_magn

        if dire == 'v':
            FX_qv = fx_qv
            FY_qv = fy_qv
        else:
            FX_qv = fy_qv
            FY_qv = fx_qv
        if invert_y:
            FY_qv = -FY_qv

        return FX_qv, FY_qv

    def plot_vf_qv(self, vfx=None, vfy=None, vector_field_part=None, qv_step=None, qv_power=None, qv_scale=None, qv_cap_magn=None, quiverwidth=None, headwidth=None, headlength=None, headaxislength=None, **kwargs):
        """

        Parameters
        ----------
        vfx
        vfy
        vector_field_part
        qv_step
        qv_power
        qv_scale
        qv_cap_magn
        quiverwidth
        headwidth
        headlength
        headaxislength

        Keyword Arguments
        -----------------
        xmin
        xmax
        xres
        ymin
        ymax
        yres
        x
        y


        Returns
        -------

        """

        self.if_vector_field = True

        self.set_xy(**kwargs)

        if vector_field_part is None:
            vector_field_part = self.vector_field_part
        else:
            self.vector_field_part = vector_field_part

        if qv_step is None:
            qv_step = self.qv_step
        else:
            self.qv_step = qv_step
        if qv_power is None:
            qv_power = self.qv_power
        else:
            self.qv_power = qv_power
        if qv_cap_magn is None:
            qv_cap_magn = self.qv_cap_magn
        else:
            self.qv_cap_magn = qv_cap_magn
        if quiverwidth is None:
            quiverwidth = self.quiverwidth
        else:
            self.quiverwidth = quiverwidth
        if headwidth is None:
            headwidth = self.headwidth
        else:
            self.headwidth = headwidth
        if headlength is None:
            headlength = self.headlength
        else:
            self.headlength = headlength
        if headaxislength is None:
            headaxislength = self.headaxislength
        else:
            self.headaxislength = headaxislength

        FX_qv, FY_qv = self.calc_vf_qv_data(vfx=vfx, vfy=vfy, vector_field_part=vector_field_part, qv_step=qv_step, qv_power=qv_power, qv_cap_magn=qv_cap_magn, dire=self.dire, invert_y=self.invert_y)
        self.vf_qv = self.ax.quiver(self.XX_qv, self.YY_qv, FX_qv, FY_qv, color=[0., .1, .3, 1.], scale_units='x', scale=qv_scale, width=quiverwidth, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)

    def update_vf_qv_plot(self, vfx, vfy):

        FX_qv, FY_qv = self.calc_vf_qv_data(vfx=vfx, vfy=vfy, vector_field_part=self.vector_field_part, qv_step=self.qv_step, qv_power=self.qv_power, dire=self.dire, invert_y=self.invert_y)
        self.vf_qv.set_UVC(FX_qv, FY_qv)
        self.ax.draw_artist(self.vf_qv)
        if self.if_update_shading and (self.pc_shaded_region is not None):
            self.ax.draw_artist(self.pc_shaded_region)

    def update_sf(self, sf):
        """
        Update scalar field plots.

        Parameters
        ----------
        sf

        Returns
        -------

        """
        if self.if_scalar_field_intensity:
            self.update_sf_intensity_plot(sf)
        if self.if_phasor:
            self.update_phasor_plot(sf)

    def update_vf(self, vfx, vfy):
        """
        Update vector field plots.

        Parameters
        ----------
        vfx
        vfy

        Returns
        -------

        """
        if self.if_vector_field:
            self.update_vf_qv_plot(vfx, vfy)