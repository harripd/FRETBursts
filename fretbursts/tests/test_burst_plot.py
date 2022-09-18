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


try:
    import matplotlib
except ImportError:
    has_matplotlib = False  # OK to run tests without matplotlib
else:
    has_matplotlib = True
    matplotlib.use('Agg')  # but if matplotlib is installed, use Agg

try:
    import numba
except ImportError:
    has_numba = False
else:
    has_numba = True


import fretbursts.background as bg
import fretbursts.burstlib as bl
import fretbursts.burstlib_ext as bext
from fretbursts.burstlib import Data
from fretbursts import loader
from fretbursts import select_bursts
from fretbursts.ph_sel import Ph_sel
from fretbursts.phtools import phrates
if has_matplotlib:
    import fretbursts.burst_plot as bplt


# data subdir in the notebook folder
DATASETS_DIR = u'data/'


def _alex_process(d):
    loader.alex_apply_period(d)
    d.calc_bg(bg.exp_fit, time_s=30, tail_min_us=300)
    d.burst_search(L=10, m=10, F=7)

def load_dataset_1ch(process=True):
    fn = "0023uLRpitc_NTP_20dT_0.5GndCl.hdf5"
    fname = DATASETS_DIR + fn
    d = loader.photon_hdf5(fname)
    if process:
        _alex_process(d)
    return d

def load_dataset_1ch_nsalex(process=True):
    fn = "dsdna_d7_d17_50_50_1.hdf5"
    fname = DATASETS_DIR + fn
    d = loader.photon_hdf5(fname)
    if process:
        _alex_process(d)
    return d

def load_dataset_8ch():
    fn = "12d_New_30p_320mW_steer_3.hdf5"
    fname = DATASETS_DIR + fn
    d = loader.photon_hdf5(fname)
    d.calc_bg(bg.exp_fit, time_s=30, tail_min_us=300)
    d.burst_search(L=10, m=10, F=7)
    return d

def load_fake_pax():
    fn = "0023uLRpitc_NTP_20dT_0.5GndCl.hdf5"
    fname = DATASETS_DIR + fn
    d = loader.photon_hdf5(fname)
    d.add(ALEX=False, meas_type='PAX')
    loader.alex_apply_period(d)
    d.calc_bg(bg.exp_fit, time_s=30, tail_min_us='auto')
    d.burst_search(L=10, m=10, F=6, pax=True)
    return d

    
def load_dataset_grouped(process=True):
    fn = ['HP3_TE150_SPC630.hdf5', 'HP3_TE200_SPC630.hdf5', 'HP3_TE250_SPC630.hdf5', 'HP3_TE300_SPC630.hdf5']
    fn = [DATASETS_DIR + f for f in fn]
    d = loader.photon_hdf5(fn)
    if process:
        _alex_process(d)
    return d


@pytest.fixture(scope="module")
def data_8ch(request):
    d = load_dataset_8ch()
    return d

@pytest.fixture(scope="module")
def data_1ch(request):
    d = load_dataset_1ch()
    return d

@pytest.fixture(scope="module")
def data_1ch_nsalex(request):
    d = load_dataset_1ch_nsalex()
    return d

@pytest.fixture(scope="module")
def data_grouped(request):
    d = load_dataset_grouped()
    return d

@pytest.fixture(scope="module", params=[
                                    load_dataset_1ch,
                                    load_dataset_1ch_nsalex,
                                    load_dataset_8ch,
                                    load_dataset_grouped
                                    ])
def data(request):
    load_func = request.param
    d = load_func()
    return d

@pytest.fixture(scope="module", params=[
                                    load_dataset_8ch,
                                    load_dataset_grouped
                                    ])
def data_mch(request):
    load_func = request.param
    d = load_func()
    return d

@pytest.fixture(scope='module', params=[load_dataset_1ch,
                                        load_dataset_1ch_nsalex,
                                        load_dataset_grouped])
def data_alex(request):
    load_func = request.param
    d = load_func()
    return d

##
#  Multi-channel plot functions
#

def test_mch_plot_bg(data_mch):
    d = data_mch
    bplt.mch_plot_bg(d)

def test_mch_plot_bg_ratio(data_mch):
    d = data_mch
    bplt.mch_plot_bg_ratio(d)

def test_mch_plot_bsize(data_mch):
    d = data_mch
    bplt.mch_plot_bsize(d)

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

def test_hist_S(data_alex):
    d = data_alex
    bplt.dplot(d, bplt.hist_S, i=None)
    for i in range(d.nch):
        bplt.dplot(d, bplt.hist_S)

@pytest.fixture(scope='module', params = (bplt.hist2d_alex, bplt.hexbin_alex,
                                          bplt.scatter_alex, bplt.scatter_naa_nt))
def ES_plots(request):
    return request.param

def test_ES_plots(data_alex, ES_plots):
    d = data_alex
    bplt.dplot(d, ES_plots, i=None)
    for i in range(d.nch):
        bplt.dplot(d, ES_plots, i=i)

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