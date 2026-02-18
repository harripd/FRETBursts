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
if has_matplotlib:
    import fretbursts.burst_plot as bplt


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
