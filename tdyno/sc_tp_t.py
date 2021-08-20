#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song, Jul 2017

"""

import numpy as np


class Hm:

    def __init__(self, c, omg, phi=0., t_bf=None, t_r=None):
        """
        Harmonic temporal profile.

        At the beginning, ramp up by a Gaussian profile. After hitting the Gaussian peak, constant amplitude.

        Parameters
        ----------
        c           :   Cnst
                        physical constants
        omg         :   float
                        frequency of the harmonic
        phi         :   float
                        phase shift of the harmonic
        t_bf        :   float
                        total time before the peak of the Gaussian.
        t_r         :   float
                        time-constant of the ramping, i.e. half-width of the Gaussian


        """

        if t_r is None:
            t_r = 10. / c.c0
        # ramp time
        self.t_r = t_r
        # time before
        if t_bf is None:
            self.t_bf = 6. * t_r
        else:
            self.t_bf = t_bf

        self.omg = omg
        self.phi = phi

        # the source field
        self.fld = None

    def f(self, t):

        """
        calculate the field for given times t

        Parameters
        ----------
        t   :   float or ndarray
                for points with t < 0, that's before the beginning of time, return 0
        Returns
        -------

        """
        if type(t) is float:
            t = np.array(t)

        # calculate source field
        self.fld = np.zeros(t.shape)
        i_t_bf = (t > 0.) * (t < self.t_bf)
        self.fld[i_t_bf] = np.exp(- np.square((t[i_t_bf] - self.t_bf) / self.t_r)) * np.sin(self.omg * t[i_t_bf] + self.phi)
        i_t_nm = (t >= self.t_bf)
        self.fld[i_t_nm] = np.sin(self.omg * t[i_t_nm] + self.phi)

        return self.fld


class HP:

    def __init__(self, c, omg, phi=0., t_spn=None, t_bf=None, t_r=None):
        """
        Harmonic Pulse temporal profile.

        At the beginning/end, use a Gaussian to ramp up/down. Constant amplitude in between for t_spn time.

        If t_spn is zero, then this is a Gaussian profile modulated harmonic wave centered at a given frequency omg.

        Parameters
        ----------
        c           :   Cnst
                        physical constants
        omg         :   float
                        frequency of the harmonic wave
        t_spn       :   float
                        length of pulse (the full amplitude section)
        t_bf        :   float
                        total time before the peak of the Gaussian.
        t_r         :   float
                        time constant of the ramping, i.e. half width of the Gaussian


        """

        if t_r is None:
            t_r = 10. / c.c0
        self.t_r = t_r
        if t_bf is None:
            self.t_bf = 6. * t_r
        else:
            self.t_bf = t_bf
        self.t_spn = t_spn

        self.omg = omg
        self.phi = phi

        # the source field
        self.fld = None

    def f(self, t):

        """
        calculate the field for given times t

        Parameters
        ----------
        t   :   float or ndarray
                for points with t < 0, that's before the beginning of time, return 0

        Returns
        -------

        """
        if type(t) is float:
            t = np.array(t)

        # calculate source field
        self.fld = np.zeros(t.shape)
        i_t_bf = (t > 0.) * (t < self.t_bf)
        self.fld[i_t_bf] = np.exp(- np.square((t[i_t_bf] - self.t_bf) / self.t_r)) * np.sin(self.omg * t[i_t_bf] + self.phi)

        i_t_nm = (t >= self.t_bf) * (t <= (self.t_bf + self.t_spn))
        self.fld[i_t_nm] = np.sin(self.omg * t[i_t_nm] + self.phi)
        i_t_af = (t > (self.t_bf + self.t_spn))
        self.fld[i_t_af] = np.exp(- np.square((t[i_t_af] - self.t_bf - self.t_spn) / self.t_r)) * np.sin(self.omg * t[i_t_af] + self.phi)

        return self.fld


class Gsn:

    def __init__(self, c, tau, t_bf=None):
        """
        A pure Gaussian pulse (no carrier frequency).

        Parameters
        ----------
        c       :   Cnst
                    Physical constants
        tau     :   float
                    width of the Gaussian
        t_bf    :   float
                    time before the peak of the Gaussian pulse

        Returns
        -------

        """

        self.tau = tau
        if t_bf is None:
            t_bf = 6. * tau
        self.t_bf = t_bf

        # the source field
        self.fld = None

    def f(self, t):

        """
        calculate the field for given times

        Parameters
        ----------
        t   :   float or ndarray
                for points with t < 0, that's before the beginning of time, return 0

        Returns
        -------

        """

        if type(t) is float:
            t = np.array(t)
        self.fld = np.zeros(t.shape)
        i_t = (t > 0.)
        self.fld[i_t] = np.exp(-np.square((t[i_t] - self.t_bf) / self.tau))

        return self.fld
