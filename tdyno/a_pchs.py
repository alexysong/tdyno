#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Y. Song, June 2017

"""

from matplotlib import patches, collections


def a_pchs(ax, pch_ls=None, dire='v', fc=None, alpha=None, lw=0.75, ec=None):
    """
    Add patches to represent dielectric objects.

    Parameters
    -------------
    ax          :   Axes
    pch_ls      :   list
                    list of patches.
    dire        :   str
                    [ 'v' | 'h' ]   for vertical or horizontal
    fc          :
                    facecolor
    alpha       :   float or None
                    transparency
    lw          :
                    linewidth
    ec          :
                    edge color

    Return
    --------
    phc

    """
    if fc is None:
        fc = [1, 1, 1, 0.]  # transparent
    if ec is None:
        ec = [0.05, 0.1, 0.3, 0.5]

    # generate matplotlib patch objects from patch as dict
    if type(pch_ls[0]) is dict:
        pch_ls_1 = []
        for pch in pch_ls:
            if pch['shp'] == 'rct':
                pa = patches.Rectangle(pch['xy'], pch['width'], pch['height'], facecolor=fc, alpha=alpha, lw=lw, ec=ec)
            elif pch['shp'] == 'ccl':
                pa = patches.Circle(pch['center'], pch['radius'], facecolor=fc, alpha=alpha, lw=lw, ec=ec)
            elif pch['shp'] == 'rng':
                pa = patches.Wedge(pch['center'], pch['radius'], 0, 360, width=pch['width'], facecolor=fc, alpha=alpha, lw=lw, ec=ec)
            elif pch['shp'] == 'wdg':
                pa = patches.Wedge(pch['center'], pch['radius'], pch['angles'][0], pch['angles'][1], width=pch['width'], facecolor=fc, alpha=alpha, lw=lw, ec=ec)
            else:
                pa = None
            pch_ls_1.append(pa)
        pch_ls = pch_ls_1

    phc = None
    if pch_ls:
        pch_collection = collections.PatchCollection(pch_ls, match_original=True, facecolor=fc, alpha=alpha, lw=lw, edgecolor=ec, zorder=3)
        phc = ax.add_collection(pch_collection)

    return phc
