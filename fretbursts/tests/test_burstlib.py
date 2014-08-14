#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
Module containing automated unit tests for FRETBursts.

Running the tests requires `py.test`.
"""

import pytest
import numpy as np

from fretbursts import loader
import fretbursts.background as bg
import fretbursts.burstlib as bl
from fretbursts.ph_sel import Ph_sel

# data subdir in the notebook folder
DATASETS_DIR = u'notebooks/data/'


def load_dataset_1ch():
    fn = "0023uLRpitc_NTP_20dT_0.5GndCl.sm"
    fname = DATASETS_DIR + fn
    d = loader.usalex(fname=fname, leakage=0.11, gamma=1.)
    d.add(det_donor_accept=(0, 1), alex_period=4000,
          D_ON=(2850, 580), A_ON=(900, 2580))
    loader.usalex_apply_period(d)

    d.calc_bg(bg.exp_fit, time_s=30, tail_min_us=300)
    d.burst_search_t(L=10, m=10, F=7)
    return d

def load_dataset_8ch():
    fn = "12d_New_30p_320mW_steer_3.dat"
    fname = DATASETS_DIR + fn
    leakage = 0.038
    gamma = 0.43
    d = loader.multispot8(fname=fname, leakage=leakage, gamma=gamma)
    d.calc_bg(bg.exp_fit, time_s=30, tail_min_us=300)
    d.burst_search_t(L=10, m=10, F=7)
    return d

@pytest.fixture(scope="module", params=[
                                    load_dataset_1ch,
                                    load_dataset_8ch,
                                    ])
def data(request):
    load_func = request.param
    d = load_func()
    return d


@pytest.fixture(scope="module")
def data_8ch(request):
    d = load_dataset_8ch()
    return d


##
# Test functions
#

def list_equal(list1, list2):
    """Test numerical equality of all the elements in the two lists.
    """
    return np.all([val1 == val2 for val1, val2 in zip(list1, list2)])

def list_array_equal(list1, list2, eq_func=np.all):
    """Test numerical equality between two lists of arrays.
    """
    return np.all([eq_func([arr1, arr2]) for arr1, arr2 in zip(list1, list2)])

def test_bg_calc(data):
    data.calc_bg(bg.exp_fit, time_s=30, tail_min_us=300)
    data.calc_bg(bg.exp_fit, time_s=30, tail_min_us='auto', F_bg=1.7)

def test_bg_from(data):
    """Test the method .bg_from() for all the ph_sel combinations.
    """
    d = data

    bg = d.bg_from(ph_sel=Ph_sel('all'))
    assert list_array_equal(bg, d.bg)
    bg = d.bg_from(ph_sel=Ph_sel(Dex='Dem'))
    assert list_array_equal(bg, d.bg_dd)
    bg = d.bg_from(ph_sel=Ph_sel(Dex='Aem'))
    assert list_array_equal(bg, d.bg_ad)

    if not d.ALEX:
        bg = d.bg_from(ph_sel=Ph_sel(Dex='DAem'))
        assert list_array_equal(bg, d.bg)
    else:
        bg = d.bg_from(ph_sel=Ph_sel(Aex='Dem'))
        assert list_array_equal(bg, d.bg_da)
        bg = d.bg_from(ph_sel=Ph_sel(Aex='Aem'))
        assert list_array_equal(bg, d.bg_aa)

        bg = d.bg_from(ph_sel=Ph_sel(Dex='DAem'))
        bg_c = [bg1 + bg2 for bg1, bg2 in zip(d.bg_dd, d.bg_ad)]
        assert list_array_equal(bg, bg_c)

        bg = d.bg_from(ph_sel=Ph_sel(Aex='DAem'))
        bg_c = [bg1 + bg2 for bg1, bg2 in zip(d.bg_da, d.bg_aa)]
        assert list_array_equal(bg, bg_c)

        bg = d.bg_from(ph_sel=Ph_sel(Dex='Dem', Aex='Dem'))
        bg_c = [bg1 + bg2 for bg1, bg2 in zip(d.bg_dd, d.bg_da)]
        assert list_array_equal(bg, bg_c)

        bg = d.bg_from(ph_sel=Ph_sel(Dex='Aem', Aex='Aem'))
        bg_c = [bg1 + bg2 for bg1, bg2 in zip(d.bg_ad, d.bg_aa)]
        assert list_array_equal(bg, bg_c)

        bg = d.bg_from(ph_sel=Ph_sel(Dex='DAem', Aex='Aem'))
        bg_c = [bg1 + bg2 + bg3 for bg1, bg2, bg3 in
                        zip(d.bg_dd, d.bg_ad, d.bg_aa)]
        assert list_array_equal(bg, bg_c)


def test_iter_ph_times(data):
    """Test method .iter_ph_times() for all the ph_sel combinations.
    """
    # TODO add all the ph_sel combinations like in test_bg_from()
    d = data

    #assert list_array_equal(d.ph_times_m, d.iter_ph_times())

    for ich, ph in enumerate(d.iter_ph_times()):
        assert (ph == d.ph_times_m[ich]).all()

    for ich, ph in enumerate(d.iter_ph_times(Ph_sel(Dex='Dem'))):
        if d.ALEX:
            assert (ph == d.ph_times_m[ich][d.D_em[ich]*d.D_ex[ich]]).all()
        else:
            assert (ph == d.ph_times_m[ich][-d.A_em[ich]]).all()

    for ich, ph in enumerate(d.iter_ph_times(Ph_sel(Dex='Aem'))):
        if d.ALEX:
            assert (ph == d.ph_times_m[ich][d.A_em[ich]*d.D_ex[ich]]).all()
        else:
            assert (ph == d.ph_times_m[ich][d.A_em[ich]]).all()

    if d.ALEX:
        for ich, ph in enumerate(d.iter_ph_times(Ph_sel(Aex='Aem'))):
            assert (ph == d.ph_times_m[ich][d.A_em[ich]*d.A_ex[ich]]).all()
    else:
        for ph1, ph2 in zip(d.iter_ph_times(Ph_sel('all')),
                            d.iter_ph_times(Ph_sel(Dex='DAem'))):
            assert ph1.size == ph2.size
            assert (ph1 == ph2).all()

def test_burst_search(data):
    data.burst_search_t(L=10, m=10, F=7, ph_sel=Ph_sel(Dex='Dem'))
    assert list_equal(data.bg_bs, data.bg_dd)
    data.burst_search_t(L=10, m=10, F=7, ph_sel=Ph_sel(Dex='Aem'))
    assert list_equal(data.bg_bs, data.bg_ad)

    if data.ALEX:
        data.burst_search_t(L=10, m=10, F=7,
                            ph_sel=Ph_sel(Dex='Aem', Aex='Aem'))
        bg_Aem = [b1 + b2 for b1, b2 in zip(data.bg_ad, data.bg_aa)]
        assert list_equal(data.bg_bs, bg_Aem)

    data.burst_search_t(L=10, m=10, F=7)

def test_b_functions(data):
    itstart, iwidth, inum_ph, iistart, iiend, itend = 0, 1, 2, 3, 4, 5
    d = data
    for mb in d.mburst:
        assert (bl.b_start(mb) == mb[:, itstart]).all()
        assert (bl.b_end(mb) == mb[:, itend]).all()
        assert (bl.b_width(mb) == mb[:, iwidth]).all()
        assert (bl.b_istart(mb) == mb[:, iistart]).all()
        assert (bl.b_iend(mb) == mb[:, iiend]).all()
        assert (bl.b_size(mb) == mb[:, inum_ph]).all()

        rate = 1.*mb[:, inum_ph]/mb[:, iwidth]
        assert (bl.b_rate(mb) == rate).all()

        separation = mb[1:, itstart] - mb[:-1, itend]
        assert (bl.b_separation(mb) == separation).all()

        assert (bl.b_end(mb) > bl.b_start(mb)).all()


def test_b_end_b_iend(data):
    """Test coherence between b_end() and b_iend()"""
    d = data
    for ph, mb in zip(d.ph_times_m, d.mburst):
        assert (ph[bl.b_iend(mb)] == bl.b_end(mb)).all()

def test_monotonic_burst_start(data):
    """Test for monotonic burst_start."""
    d = data
    for i in xrange(d.nch):
        assert (np.diff(bl.b_start(d.mburst[i])) > 0).all()

def test_monotonic_burst_end(data):
    """Test for monotonic burst_end."""
    d = data
    for mb in d.mburst:
        assert (np.diff(bl.b_end(mb)) > 0).all()

def test_burst_start_end_size(data):
    """Test consistency between burst istart, iend and size"""
    d = data
    for mb in d.mburst:
        size = mb[:, bl.iiend] - mb[:, bl.iistart] + 1
        assert (size == mb[:, bl.inum_ph]).all()
        size2 = bl.b_iend(mb) - bl.b_istart(mb) + 1
        assert (size2 == bl.b_size(mb)).all()

def test_burst_fuse_0ms(data):
    """Test that after fusing with ms=0 the sum of bursts sizes is that same
    as the number of ph in bursts (via burst selection).
    """
    d = data
    if not hasattr(d, 'fuse'):
        df = d.fuse_bursts(ms=0)
        for ph, mb in zip(df.ph_times_m, df.mburst):
            m = bl.ph_select(ph, mb)
            assert m.sum() == bl.b_size(mb).sum()

def test_get_burst_size(data):
    """Test that get_burst_size() returns nd + na when gamma = 1.
    """
    d = data
    for ich, (nd, na) in enumerate(zip(d.nd, d.na)):
        burst_size = bl.select_bursts.get_burst_size(d, ich)
        assert (burst_size == nd + na).all()

def test_expand(data):
    """Test method `expand()` for `Data()`."""
    d = data
    for ich, mb in enumerate(d.mburst):
        if mb.size == 0: continue  # if no bursts skip this ch
        nd, na, bg_d, bg_a, width = d.expand(ich, width=True)
        width2 = bl.b_width(mb)*d.clk_p
        period = d.bp[ich]
        bg_d2 = d.bg_dd[ich][period] * width2
        bg_a2 = d.bg_ad[ich][period] * width2
        assert (width == width2).all()
        assert (nd == d.nd[ich]).all() and (na == d.na[ich]).all()
        assert (bg_d == bg_d2).all() and (bg_a == bg_a2).all()


def test_burst_corrections(data):
    """Test background and bleed-through corrections."""
    d = data
    d.calc_ph_num(alex_all=True)
    d.corrections()
    leakage = d.get_leakage_array()

    for ich, mb in enumerate(d.mburst):
        if mb.size == 0: continue  # if no bursts skip this ch
        nd, na, bg_d, bg_a, width = d.expand(ich, width=True)
        burst_size_raw = bl.b_size(mb)

        lk = leakage[ich]
        if d.ALEX:
            nda, naa = d.nda[ich], d.naa[ich]
            period = d.bp[ich]
            bg_da = d.bg_da[ich][period]*width
            bg_aa = d.bg_aa[ich][period]*width
            burst_size_raw2 = nd + na + bg_d + bg_a + lk*nd + nda + naa + \
                              bg_da + bg_aa
            assert np.allclose(burst_size_raw, burst_size_raw2)
        else:
            burst_size_raw2 = nd + na + bg_d + bg_a + lk*nd
            assert np.allclose(burst_size_raw, burst_size_raw2)

def test_burst_size_da(data):
    """Test that nd + na with no corrections is equal to b_size(mburst).
    """
    d = data
    d.calc_ph_num(alex_all=True)
    if d.ALEX:
        for mb, nd, na, naa, nda in zip(d.mburst, d.nd, d.na, d.naa, d.nda):
            tot_size = bl.b_size(mb)
            tot_size2 = nd + na + naa + nda
            assert np.allclose(tot_size, tot_size2)
    else:
        for mb, nd, na in zip(d.mburst, d.nd, d.na):
            tot_size = bl.b_size(mb)
            assert (tot_size == nd + na).all()

def test_collapse(data_8ch):
    """Test the .collapse() method that joins the ch.
    """
    d = data_8ch
    dc1 = d.collapse()
    dc2 = d.collapse(update_gamma=False)

    for name in d.burst_fields:
        if name in d:
            assert np.allclose(dc1[name][0], dc2[name][0])

if __name__ == '__main__':
    pytest.main("-x -v fretbursts/tests/test_burstlib.py")
