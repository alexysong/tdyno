# -*- coding: utf-8 -*-


class Cnst:

    def __init__(self, c0=1., mu0=1., epsi0=1.):
        """
        Physical constants.

        These constants are default to 1.

        c0 = 1 / np.sqrt(epsi0 * mu0) has to hold, otherwise PML will not function!

        Parameters
        ----------
        c0
        mu0
        epsi0
        """

        self.c0 = c0
        self.mu0 = mu0
        self.epsi0 = epsi0
