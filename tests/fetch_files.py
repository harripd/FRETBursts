#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 07:24:07 2026

@author: paul
"""

import pooch

# data subdir in the notebook folder
DATASETS_DIR = u'data/'

alex1c = pooch.create(path=DATASETS_DIR, base_url='doi:10.6084/m9.figshare.1019906.v26')
alex1c.load_registry_from_doi()
_ = alex1c.fetch("12d_New_30p_320mW_steer_3.hdf5")

phdf5 = pooch.create(path=DATASETS_DIR, base_url='doi:10.6084/m9.figshare.1456362.v15')
phdf5.load_registry_from_doi()
_ = phdf5.fetch("0023uLRpitc_NTP_20dT_0.5GndCl.hdf5")

mphmm = pooch.create(path=DATASETS_DIR, base_url='doi:10.5281/zenodo.5902313')
mphmm.load_registry_from_doi()
hp3_files = ['HP3_TE150_SPC630.hdf5', 'HP3_TE200_SPC630.hdf5', 'HP3_TE250_SPC630.hdf5', 'HP3_TE300_SPC630.hdf5']
for fn in hp3_files:
    mphmm.fetch(fn)

