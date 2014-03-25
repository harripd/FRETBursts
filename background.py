#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
Routines to compute the background from an array of timestamps. This module 
is imported as `bg` by `burstlib.py`.

The important functions are :func:`exp_fit` and :func:`exp_cdf_fit` that 
provide two (fast) algorithms to estimate the background without binning. 
These functions are not usually called directly but passed to 
:meth:`Data.calc_bg` to compute the background of a measurement.

See also :func:`exp_hist_fit` for background estimation using an histogram fit.
"""

import numpy as np
from utils.misc import pprint
from fit import exp_fitting
from fit.gaussian_fitting import gaussian_fit_hist


def raw_fit(ph, clk_p=12.5e-9, residuals=False):
    """Compute the "raw" rate: (number of ph / duration). """
    Lambda = ph.size/((ph[-1]-ph[0])*clk_p)
    if residuals:
        resid = exp_fitting.get_residuals(np.diff(ph), 1./(Lambda*clk_p))
        return Lambda, np.abs(resid).max()*100
    else:
        return Lambda


def _exp_fit_generic(ph, fit_fun, tail_min_us=None, tail_min_p=0.1, 
                     clk_p=12.5e-9):
    """Computes BG rates on timestamp delays above a min. value.
    
    Compute a background rate, selecting waiting-times (delays) larger than a 
    minimum threshold.
    
    You need to pass the specific fitting function as `fit_fun`.
    """
    dph = np.diff(ph)
    if tail_min_us is None:
        tail_min = dph.max()*tail_min_p
    else:
        tail_min = tail_min_us*1e-6/clk_p
    Lambda, residuals, s_size = fit_fun(dph, s_min=tail_min)
    Lambda /= clk_p
    #print s_size,
    return Lambda, np.abs(residuals).max()*100


def exp_fit(ph, tail_min_us=None, clk_p=12.5e-9):
    """Return a background rate using the MLE of mean waiting-times.
    
    Compute the background rate, selecting waiting-times (delays) larger 
    than a minimum threshold.

    This function performs a Maximum Likelihood (ML) fit. For
    exponentially-distributed waiting-times this is the empirical mean.

    Arguments:
        ph (array): timestamps array from which to extract the background
        tail_min_us (float): minimum waiting-time in micro-secs
        clk_p (float): clock period for timestamps in `ph`
    
    Returns:
        Estimated background rate in cps.

    See also:
        :func:`exp_cdf_fit`, :func:`exp_hist_fit`
    """
    return _exp_fit_generic(ph, fit_fun=exp_fitting.expon_fit, 
                            tail_min_us=tail_min_us, clk_p=clk_p)

def exp_cdf_fit(ph, tail_min_us=None, clk_p=12.5e-9):
    """Return a background rate fitting the empirical CDF of waiting-times.

    Compute the background rate, selecting waiting-times (delays) larger 
    than a minimum threshold.

    This function performs a least square fit of an exponential Cumulative 
    Distribution Function (CDF) to the empirical CDF of waiting-times.

    Arguments:
        ph (array): timestamps array from which to extract the background
        tail_min_us (float): minimum waiting-time in micro-secs
        clk_p (float): clock period for timestamps in `ph`
    
    Returns:
        Estimated background rate in cps.
        
    See also:
        :func:`exp_fit`, :func:`exp_hist_fit`
    """
    return _exp_fit_generic(ph, fit_fun=exp_fitting.expon_fit_cdf, 
                            tail_min_us=tail_min_us, clk_p=clk_p)


def exp_hist_fit(ph, tail_min_us, binw=50e-6, clk_p=12.5e-9, 
                  weights='hist_counts'):
    """Compute background rate with WLS histogram fit of waiting-times.

    Compute the background rate, selecting waiting-times (delays) larger 
    than a minimum threshold.

    This function performs a Weighed Least Squares (WLS) fit of the
    histogram of waiting times to an exponential decay.

    Arguments:
        ph (array): timestamps array from which to extract the background
        tail_min_us (float): minimum waiting-time in micro-secs
        binw (float): bin width for waiting times, in seconds.
        clk_p (float): clock period for timestamps in `ph`
        weights (None or string): if None no weights is applied.
            if is 'hist_counts', each bin has a weight equal to its counts
            if is 'inv_hist_counts', the weight is the inverse of the counts.
    
    Returns:
        Estimated background rate in cps.
        
    See also:
        :func:`exp_fit`, :func:`exp_cdf_fit`
    """
    assert np.size(ph) > 0
    dph = np.diff(ph)
    tail_min = tail_min_us*1e-6/clk_p
    binw_clk = binw/clk_p
    bins = np.arange(0, dph.max() - tail_min + 1, binw_clk)
    Lambda, residuals, s_size = exp_fitting.expon_fit_hist(dph, 
                                bins=bins, s_min=tail_min, weights=weights)    
    Lambda /= clk_p
    return Lambda, np.abs(residuals).max()*100

##
# Other functions
#
def histo(ph, bin_ms=10., t_max_s=None, clk_p=12.5e-9):
    """Returns an histogram and bins-centers of ph (ph arrival times)."""
    if t_max_s is not None:
        ph = ph[ph <= t_max_s/clk_p]
    bins = np.arange(ph[0], ph[-1]+1, (bin_ms*1e-3)/clk_p)
    H = np.histogram(ph, bins=bins)
    tt = H[0]
    ti = H[1][:-1]+0.5*(H[1][1]-H[1][0])
    return tt, ti

def gauss_fit(ph, bin_ms=10, clk_p=12.5e-9):
    """Returns the BG rate of ph calculated from the hist (PDF) of timetrace.
    """
    assert np.size(ph) > 0
    tt, ti = histo(ph=ph, bin_ms=bin_ms, clk_p=clk_p)
    #mu, sig = gaussian_fit(tt, mu_sigma_guess=[tt.mean(), tt.std()])
    mu, sig = gaussian_fit_hist(tt, mu0=tt.mean(), sigma0=tt.std())
    mask = (tt<(mu+3*sig))*(tt>(mu-3*sig))
    tt2 = tt[mask]
    #mu2, sig2 = gaussian_fit(tt2, mu_sigma_guess=[mu,sig])
    mu2, sig2 = gaussian_fit_hist(tt2, mu0=mu, sigma0=sig)
    return mu2/(bin_ms*1e-3)#, sig2/(bin_ms*1e-3)


##
# Experimental functions
#
def smart_bg(d, ich=0, bin_=50e-3, step=1):
    """BG calculation through binning (WARNING: very slow!)."""
    bg = []
    t = d.ph_times_m[ich]*d.clk_p
    t_max = np.floor(t.max())
    pprint(" Calculation started:")
    for s in np.arange(step, t_max, step):
        #if (s % (t_max/50) == 0): pprint(" %d %%" % (s/t_max*100))
        h = np.histogram(t[(t<s)*(t>(s-step))],
                bins=np.arange(s-step, s+1e-3, bin_))
        print h[0]
        bg.append(h[0].min())
    pprint('\n')
    return np.array(bg)/bin_

