# Author: Paul David Harris
# Purpose: Unit tests for burstlib_ext.py functions
# Created: 13 Sept 2022
"""
Unit tests for the burstlib_ext module (bext)

Currently mostly just smoke tests

Running tests requires pytest
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

def test_load_group():
    """Smoke test to see if loading groupd data works"""
    fn = ['HP3_TE150_SPC630.hdf5', 'HP3_TE200_SPC630.hdf5', 'HP3_TE250_SPC630.hdf5', 'HP3_TE300_SPC630.hdf5']
    fn = [DATASETS_DIR + f for f in fn]
    d = loader.photon_hdf5(fn)
    assert d.nch == len(fn)
    
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

@pytest.mark.parametrize("data_ch, process", product([load_dataset_1ch, 
                                             load_dataset_1ch_nsalex, 
                                             load_dataset_8ch],
                                            [True, False]))
def test_group_data(data_ch, process):
    if data_ch is load_dataset_8ch:
        orig_d = data_ch()
    else:
        orig_d = data_ch(process=process)
    n = orig_d.nch
    d = bext.group_data([orig_d for _ in range(8)])
    assert n*8 == d.nch
    for field in Data.ph_fields:
        if hasattr(d, field):
            assert len(d[field]) == 8*n, f"{field} not correct length"
    if not process and orig_d.alternated:
        loader.alex_apply_period(d)
        for field in Data.ph_fields:
            if hasattr(d, field):
                assert len(d[field]) == 8*n, f"{field} not correct length"
    d.calc_bg(bg.exp_fit, time_s=30, tail_min_us='auto')
    d.burst_search(m=10, F=7, L=10)
    for field in Data.burst_fields:
        if hasattr(d, field):
            assert len(d[field]) == 8*n, f'{field} not correct length'

def test_join_data(data):
    """Smoke test for bext.join_data() function.
    """
    d = data
    dj = bext.join_data([d, d.copy()])
    assert (dj.num_bursts == 2 * d.num_bursts).all()
    for bursts in dj.mburst:
        assert (np.diff(bursts.start) > 0).all()

def test_burst_search_and_gate(data_1ch):
    """Test consistency of burst search and gate."""
    d = data_1ch
    assert d.alternated

    # Smoke tests
    bext.burst_search_and_gate(d, F=(6, 8))
    bext.burst_search_and_gate(d, m=(12, 8))
    bext.burst_search_and_gate(d, min_rate_cps=(60e3, 40e3))
    if d.nch > 1:
        mr1 = 35e3 + np.arange(d.nch) * 1e3
        mr2 = 30e3 + np.arange(d.nch) * 1e3
        bext.burst_search_and_gate(d, min_rate_cps=(mr1, mr2))

    # Consistency test
    d_dex = d.copy()
    d_dex.burst_search(ph_sel=Ph_sel(Dex='DAem'))
    d_aex = d.copy()
    d_aex.burst_search(ph_sel=Ph_sel(Aex='Aem'))
    d_and = bext.burst_search_and_gate(d)
    for bursts_dex, bursts_aex, bursts_and, ph in zip(
            d_dex.mburst, d_aex.mburst, d_and.mburst, d.iter_ph_times()):
        ph_b_mask_dex = bl.ph_in_bursts_mask(ph.size, bursts_dex)
        ph_b_mask_aex = bl.ph_in_bursts_mask(ph.size, bursts_aex)
        ph_b_mask_and = bl.ph_in_bursts_mask(ph.size, bursts_and)
        assert (ph_b_mask_and == ph_b_mask_dex * ph_b_mask_aex).all()

def test_burst_data(data):
    """Test for bext.burst_data()"""
    bext.burst_data(data, include_bg=True, include_ph_index=True)
    bext.burst_data(data, include_bg=False, include_ph_index=True)
    bext.burst_data(data, include_bg=True, include_ph_index=False)
    bext.burst_data(data, include_bg=False, include_ph_index=False)




def test_asymmetry(data):
    d = data
    for i in range(data.nch):
        asym = bext.asymmetry(data, i, dropnan=False)
        assert len(asym) == d.mburst[i].size
        if np.any(np.isnan(asym)):
            nan_count = np.isnan(asym).sum()
            asym = bext.asymmetry(d, i)
            assert len(asym) == d.mburst[i].size - nan_count


def test_calc_mdelays_hist(data):
    """Smoke test for calc_mdelays_hist"""
    d = data
    for i in range(d.nch):
        for ph_sel in [Ph_sel('all'), Ph_sel(Dex='Dem'), Ph_sel(Dex='Aem')]:
            bext.calc_mdelays_hist(d, ich=i)

def test_burst_fitter(data):
    d = data
    bext.bursts_fitter(d)
    assert hasattr(d, 'E_fitter')
    if d.alternated:
        bext.bursts_fitter(d, burst_data='S')
        assert hasattr(d, 'S_fitter')