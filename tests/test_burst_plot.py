# Author: Paul David Harris
# Purpose: Unit tests for burst_plot.py functions
# Created: 13 Sept 2022
"""
Tests of plotting functions

These are mostly smoke tests
"""

from collections import namedtuple
from itertools import product
import pytest
import numpy as np


import matplotlib
matplotlib.use('Agg')  # but if matplotlib is installed, use Agg
import matplotlib.pyplot as plt
# try:
#     import numba
# except ImportError:
#     has_numba = False
# else:
#     has_numba = True


import fretbursts.background as bg
import fretbursts.burstlib as bl
import fretbursts.burstlib_ext as bext
from fretbursts.burstlib import Data
from fretbursts import loader
from fretbursts import select_bursts
from fretbursts.ph_sel import Ph_sel
from fretbursts.phtools import phrates
import fretbursts.burst_plot as bplt



##
#  Multi-channel plot functions
#

def test_mch_plot_bg(data_mch):
    d = data_mch
    bplt.mch_plot_bg(d)
    plt.close()

def test_mch_plot_bg_ratio(data_mch):
    d = data_mch
    bplt.mch_plot_bg_ratio(d)
    plt.close()

def test_mch_plot_bsize(data_mch):
    d = data_mch
    bplt.mch_plot_bsize(d)
    plt.close()

##
#  Timetrace plots
#
@pytest.fixture(scope='module', params = (bplt.timetrace_single, bplt.ratetrace_single,
                                          bplt.hist_interphoton_single))
def ratetraces(request):
    return request.param

def test_trace_single(data, ratetraces):
    """Test plotting functions for time traces that do so on single Ph_sel"""
    d = data
    ph_sel_alex = (Ph_sel('all'), Ph_sel(Dex='Dem'), Ph_sel(Dex='Aem'), Ph_sel(Aex='Dem'), Ph_sel(Aex='Aem'))
    ph_sel_single = (Ph_sel('all'), Ph_sel(Dex='Dem'), Ph_sel(Dex='Aem'))
    ph_sel_list = ph_sel_alex if d.alternated else ph_sel_single
    for i in range(d.nch):
        for ph_sel in ph_sel_list:
            bplt.dplot(d, ratetraces, i=i, ph_sel=ph_sel)
    plt.close()

@pytest.fixture(scope='module', params = (bplt.timetrace, bplt.ratetrace, 
                                          bplt.timetrace_bg, bplt.timetrace_fret,
                                          bplt.timetrace_fret_scatter, bplt.time_ph))
def timetraces(request):
    return request.param

def test_trace(data, timetraces):
    """Test general time trace type functions"""
    d = data
    for i in range(d.nch):
        bplt.dplot(d, timetraces, i=i)
    plt.close()


@pytest.fixture(scope='module', params = (bplt.hist_size, bplt.hist_width,
                                          bplt.hist_brightness, bplt.hist_size_all,
                                          bplt.hist_fret, bplt.hist_interphoton,
                                          bplt.hist_ph_delays, bplt.hist_mdelays,
                                          bplt.hist_mrates, bplt.hist_sbr, 
                                          bplt.hist_burst_phrate, bplt.hist_burst_delays,
                                          bplt.hist_asymmetry))
def hists(request):
    return request.param

def test_hist(data, hists):
    d = data
    bplt.dplot(d, hists, i=None)
    for i in range(d.nch):
        bplt.dplot(d, hists, i=i)
    plt.close()

def test_hist_S(data_alex):
    d = data_alex
    bplt.dplot(d, bplt.hist_S, i=None)
    for i in range(d.nch):
        bplt.dplot(d, bplt.hist_S)
    plt.close()

@pytest.fixture(scope='module', params = (bplt.hist2d_alex, bplt.hexbin_alex,
                                          bplt.scatter_alex, bplt.scatter_naa_nt))
def ES_plots(request):
    return request.param

def test_ES_plots(data_alex, ES_plots):
    d = data_alex
    bplt.dplot(d, ES_plots, i=None)
    if ES_plots in (bplt.scatter_alex, bplt.scatter_naa_nt):
        bplt.dplot(d, ES_plots, i=0, color_style='kde')
    for i in range(d.nch):
        bplt.dplot(d, ES_plots, i=i)
    plt.close()

@pytest.fixture(scope='module', params = (bplt.scatter_width_size, bplt.scatter_rate_da,
                                          bplt.scatter_fret_size, bplt.scatter_fret_nd_na,
                                          bplt.scatter_fret_width, bplt.scatter_da,))
def scatterplots(request):
    return request.param

def test_scatterplots(data, scatterplots):
    d = data
    bplt.dplot(d, scatterplots, i=None)
    for i in range(d.nch):
        bplt.dplot(d, scatterplots, i=i)
    plt.close()
