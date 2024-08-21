#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
Module containing automated unit tests for FRETBursts.

Running the tests requires `py.test`.
"""

from __future__ import division

from itertools import product, permutations
import pytest
from fretbursts.ph_sel import Ph_sel
import numpy  as np



det_dict_string = {'ex':{'D':(0,),'A':(1,),'DA':(0,1), '0':(0,), '1':(1,), '2':(2,), '[0,1,2]':(0,1,2)},
           'em':{'D':(0,),'A':(1,),'DA':(0,1), '0':(0,), '1':(1,), '2':(2,), '[0,1,2]':(0,1,2)},
           'pol':{'P':(0,), 'S':(1,), 'par':(0,), 'per':(1,), '0':(0,), '1':(1,)},
           'split':{'0':(0,), '1':(1,)}}

def make_det_map(det_strings):
    min_det = {key:np.unique(np.concatenate([v for v in val.values()])).size for key, val in det_strings.items()}
    num_det = np.product([v for v in min_det.values()])
    dets = np.random.randint(255, size=tuple(v for v in min_det.values()))
    while np.unique(dets).size < num_det:
        uni, whr = np.unique(dets, return_counts=True)
        nums = uni[whr!=1]
        for num in nums:
            mask = dets == num
            dets[mask] = [num] + list(np.random.randint(255, size=mask.sum()-1))

    det_map = {
        key:tuple(dets[tuple(slice(None) if k != i else j for k in range(len(min_det)))].reshape(-1) 
                  for j in range(min_det[key])) 
        for i, key in enumerate(min_det.keys())
    }
    det_map['all_streams'] = dets.reshape(-1)
    return det_map, dets

def intersect_multi(*args):
    if len(args) == 1:
        return args[0]
    intersect = np.intersect1d(args[0], args[1])
    for arr in args[2:]:
        intersect = np.intersect1d(intersect, arr)
    return intersect

def union_multi(*args):
    if len(args) == 1:
        return args[0]
    union = np.union1d(args[0], args[1])
    for arr in args[2:]:
        union = np.union1d(union, arr)
    return union

def make_1det_list(strings):
    streams = list()
    for l in range(1,len(strings)+1):
        for det_types in permutations(strings.keys(), l):
            det_dict = {key:strings[key] for key in det_types}
            det_map, dets = make_det_map(det_dict)
            for ex_types in product(*[strings[det].keys() for det in det_types]):
                stream = str().join([em + det for em, det in zip(ex_types, det_types)])
                expect = intersect_multi(*[union_multi(*[det_map[det][i] for i in det_dict[det][em]])
                                          for em, det in zip(ex_types, det_types)])
                streams.append((stream, det_map, expect))
    return streams

def make_1det_list_whole(strings):
    streams = list()
    for l in range(1,len(strings)+1):
        whole_det_map, whole_dets = make_det_map(strings)
        for det_types in permutations(strings.keys(), l):
            for ex_types in product(*[strings[det].keys() for det in det_types]):
                stream = str().join([em + det for em, det in zip(ex_types, det_types)])
                whole_expect = intersect_multi(*[union_multi(*[whole_det_map[det][i] for i in strings[det][em]])
                                                for em, det in zip(ex_types, det_types)])
                streams.append((stream, whole_det_map, whole_expect))
    return streams


def make_2det_list(strings):
    streams = list()
    for l in range(1,len(strings)+1):
        whole_det_map, whole_dets = make_det_map(strings)
        for det_types in permutations(strings.keys(), l):
            for ex_types0 in product(*[strings[det].keys() for det in det_types]):
                stream0 = str().join([em + det for em, det in zip(ex_types0, det_types)])
                whole_expect0 = intersect_multi(*[union_multi(*[whole_det_map[det][i] for i in strings[det][em]]) 
                                                  for em, det in zip(ex_types0, det_types)])
                cnt = 0
                for ex_types1 in product(*[strings[det].keys() for det in det_types]):
                    stream1 = str().join([em + det for em, det in zip(ex_types1, det_types)])
                    whole_expect1 = intersect_multi(*[union_multi(*[whole_det_map[det][i] for i in strings[det][em]])
                                                    for em, det in zip(ex_types1, det_types)])
                    if stream0 == stream1:
                        continue
                    else: 
                        stream = "_".join([stream0, stream1])
                        whole_expect = np.union1d(whole_expect0, whole_expect1)
                        if whole_expect.size == whole_expect1.size or whole_expect.size == whole_expect1.size:
                            continue
                        cnt += 1
                        if cnt > 2:
                            break
                    streams.append((stream, whole_det_map, whole_expect))
    return streams


@pytest.mark.parametrize("stream, dmap, expect", make_1det_list(det_dict_string) + make_1det_list_whole(det_dict_string) + 
                         make_2det_list(det_dict_string))
def test_Ph_sel_single(stream, dmap, expect):
    sel = Ph_sel(stream)
    assert np.all(np.sort(sel.get_det(dmap)) == np.sort(expect))
    # only run this test if ph_sel is equivalen to all
    if np.union1d(dmap['all_streams'], expect).size != expect.size:
        i = [i for i in dmap['all_streams'] if i not in expect][0]
        # make array of dets, 1st 10 in the selection, next outside of selection
        dets = np.concatenate([expect[np.random.randint(0, expect.size, size=10)], i * np.ones(10, dtype=int)])
        assert np.all(sel.get_mask(dmap, dets) == np.concatenate([np.ones(10,dtype=bool), np.zeros(10, dtype=bool)]))
    
