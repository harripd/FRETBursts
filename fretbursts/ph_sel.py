#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:33:05 2022

@author: paul
"""

import re
import numpy as np
import functools
import warnings
from itertools import combinations
from collections.abc import Iterable

def _check_kwargs(func):
    @functools.wraps(func)
    def kwarg_sort(*args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, Iterable) and np.all([np.issubdtype(type(v), np.integer) and v >=0 for v in value]):
                kwargs[key] = tuple(value)
            elif isinstance(value, int) and value >= 0:
                kwargs[key] = (value, )
            elif value is not None:
                raise ValueError(f"{key} received non-int or negative value(s)")
        return func(*args, **kwargs)
    return kwarg_sort


_char_dict = {'D':(0, ), 'A':(1, ), 'DA':(0,1), 
              'par':(0, ), 'per':(1, ), 'P':(0, ), 'S':(1, ), 
              "0":(0, ), "1":(1, )}

_inv_dict = {'ex':('D', 'A'), 'em':('D', 'A'), 'pol':('P', 'S'), 'split':('0', '1')}

_ph_keys = {'ex','em','pol','split'}

_stream_regex = dict(em = re.compile(r'((\[(\d+,)*\d+\])|(\d+|A|D|DA))em'), 
                    ex = re.compile(r'((\[(\d+,)*\d+\])|(\d+|A|D|DA))ex'), 
                    pol = re.compile(r'(S|P|par|per|0|1)pol'), 
                    split = re.compile(r'\d+split')) # list of regex for stream identifiers

_find_id_exem = re.compile(r'(\d+|DA|D|A)')

_find_id = {'ex':_find_id_exem, 'em':_find_id_exem,
            'pol':re.compile(r'([01SP]|par|per)'), 'split':re.compile(r'[01]')}

_trim = re.compile(r'[,_]$')

_stream_patern = re.compile(r'(((\[(\d+,)*\d+\])|(\d+|A|D|DA))em|((\[(\d+,)*\d+\])|(\d+|A|D|DA))ex|(S|P|par|per|0|1)pol|\d+split)+[,_]?')

def _inv_det(key, val):
    det = str()
    if val is not None:
        if np.any([v > 1 for v in val]):
            if len(val) == 1:
                det += str(val[0])
            else:
                det += '['
                for v in val:
                    det += f'{v},'
                det = det[:-1] + f']{key}'
        else:
            for v in val:
                det += _inv_dict[key][v]
            det += key
    return det
            

class Ph_stream:
    __slots__ = ('__ex', '__em', '__pol', '__split', '__hash')
    @_check_kwargs
    def __init__(self, ex=None, em=None, pol=None, split=None):
        self.__ex = ex
        self.__em = em
        self.__pol = pol
        self.__split = split
        self.__hash = hash((ex, em, pol, split))
    
    def __iter__(self):
        for stream in self.__slots__[:-1]:
            stream = stream[2:] # odd [2:] to remove double underscore of protected attributes in slots
            idxs = getattr(self, stream)
            if idxs is not None:
                yield stream, idxs
    
    def __hash__(self):
        return self.__hash
    
    def __eq__(self, other):
        if isinstance(other, (Ph_stream, Ph_sel)):
            return hash(self) == hash(other)
        else:
            return False
    
    def __contains__(self, other):
        if self == other:
            return True
        for stream, idxs in self: # check all values in self also in other
            oids = getattr(other, stream)
            if oids is None or np.intersect1d(idxs, oids).size < len(oids):
                return False
        else:
            return True
    
    def __str__(self):
        string = str()
        for key, val in self:
            string += _inv_det(key, val)
        return string
    
    def __repr__(self):
        return "Ph_stream: " + self.__str__()
    
    @property
    def ex(self):
        return self.__ex
    
    @property
    def em(self):
        return self.__em
    
    @property
    def pol(self):
        return self.__pol
    
    @property
    def split(self):
        return self.__split
    
    @property
    def Dex(self):
        if self.__ex is None:
            return True
        else:
            return 0 in self.__ex
    
    @property
    def Aex(self):
        if self.__ex is None:
            return True
        else:
            return 1 in self.__ex
    
    @property
    def Dem(self):
        if self.__em is None:
            return True
        else:
            return 0 in self.__em
    
    @property
    def Aem(self):
        if self.__em is None:
            return True
        else:
            return 0 in self.__em
    
    @_check_kwargs
    def update(self, **kwargs):
        mkdict = {key:val for key, val in self}
        mkdict.update(**kwargs)
        return Ph_stream(**mkdict)
    
    def get_det(self, stream_map):
        # get streams of importance
        streams = [np.concatenate([stream_map[stream][idx] for idx in idxs]) for stream, idxs in self]
        if len(streams) > 1:
            streams = tuple(np.intersect1d(*streams))
        elif len(streams) == 1:
            streams = tuple(streams[0])
        else:
            streams = tuple(stream_map['all_streams'])
        return streams
    
    def get_mask(self, stream_map, dets):
        mask = np.zeros(dets.shape, dtype=bool)
        for det in self.get_det(stream_map):
            mask += det == dets
        return mask


def _convert_str(ph_str):
    if ph_str in _char_dict:
        return _char_dict[ph_str]
    elif ph_str.isnumeric():
        return (int(ph_str), )
    else:
        raise ValueError(f"Unknown stream: {ph_str}")
    

def _tuple_dets(dets, key):
    tdet = list()
    for attr in _find_id[key].finditer(dets[0]):
        tdet += list(_convert_str(attr.group()))
    return tuple(tdet)


def _make_dets(ph_id, key, regex):
    dets = [s.group(1) for s in regex.finditer(ph_id)]
    ldets = len(dets)
    if ldets == 0:
        return None
    elif ldets == 1:
        return _tuple_dets(dets, key)
    else:
        raise ValueError(f"Cannot specify stream {key} type multiple times per stream")

    
def _process_str(ph_str):
    if ph_str == "all":
        return (Ph_stream(), )
    # loop over each set of stream identifiers
    streams = list()
    ph_len = 0
    for ph_id in _stream_patern.finditer(ph_str):
        ph_id = ph_id.group()
        ph_len += len(ph_id)
        ph_id = _trim.sub('', ph_id) # get rid of , and _ separators
        
        streams.append(Ph_stream(**{key:_make_dets(ph_id, key, regex) for key, regex in _stream_regex.items()}))
    if ph_len != len(ph_str):
        raise ValueError(f"Non-parsable characters in string {ph_str}")
    return streams


def _process_dict(ph_dict):
    if "Dex" in ph_dict or "Aem" in ph_dict:
        if len(ph_dict) > 1 and  not ("Dex" in ph_dict and "Aex" in ph_dict):
            raise ValueError("Cannot mix old 'Dex'/'Aex' kwargs with new 'em'/'ex' kwargs")
        warnings.warn("Specifiying Ph_sel with Dex/Aex kwargs discourated, and will be deprecated. Prefered method to specify with string",
                      DeprecationWarning)
        streams = [Ph_stream(ex=_char_dict[key[:-2]], em=_char_dict[val[:-2]]) for key, val in ph_dict.items()]
    else:
        streams = [Ph_stream(**ph_dict)]
    return streams

def _tuplesub(phs0, phs1, key):
    tup0, tup1 = getattr(phs0, key), getattr(phs1, key)
    if None in (tup0, tup1):
        sub0, sub1 = tup0 is None, tup1 is None
    else:
        sub0 = np.all([t0 in tup1 for t0 in tup0])
        sub1 = np.all([t1 in tup0 for t1 in tup1])
    return sub0, sub1

def _tupunion(phs0, phs1, key):
    tup0, tup1 = getattr(phs0, key), getattr(phs1, key)
    if None in (tup0, tup1):
        return None
    else:
        return tuple(set(tup0) | set(tup1))

def _comp_break(comp, subcnt0, subcnt1):
    if comp > 1: 
        return True
    elif comp == 1 and (subcnt0 > 0 or subcnt1 > 0):
        return True
    elif subcnt0 > 0 and subcnt1 > 0:
        return True
    else:
        return False

def parallel_stream(sel0, sel1):
    if not np.all([isinstance(sel, Ph_stream) for sel in (sel0, sel1)]):
        raise TypeError(f"parallel_stream only take Ph_stream arguments, got {type(sel0)} and {type(sel1)}")
    comp, subcnt0, subcnt1, par = 0, 0, 0, True
    for key in Ph_stream.__slots__[:-1]:
        sub0, sub1 = _tuplesub(sel0, sel1, key[2:])
        if sub0 and sub1:
            continue
        elif not (sub0 or sub1):
            comp += 1
        else:
            subcnt0 += sub0
            subcnt1 += sub1
        # check for break conditions
        if _comp_break(comp, subcnt0, subcnt1):
            par = False
            break
    # generate dicationary of union of two streams
    udict = {key[2:]:_tupunion(sel0, sel1, key[2:]) for key in Ph_stream.__slots__[:-1]}
    return par, udict


class Ph_sel:
    __slots__ = ("__streams", "__hash")
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError("Cannot specify streams as args and kwargs at the same time")
        elif len(args) > 0:
            streams = list()
            for arg in args:
                streams += _process_str(arg) if isinstance(arg, str) else _process_dict(arg)
        elif len(kwargs) > 0:
            streams = _process_dict(kwargs)
        else:
            raise ValueError("Must specify a stream or all")
        # check no streams identical or substreams of others
        repl_mask = [True for _ in range(len(streams))]
        for (i, streama), (j, streamb) in combinations(enumerate(streams), 2):
            # check if any streams can be combined
            par, new_stream = parallel_stream(streama, streamb)
            if par:
                warnings.warn(f"Streams {streama.__str__()} and {streamb.__str__()} can be represented as single stream {new_stream.__str__()}")
                streams[i] = Ph_stream(**new_stream)
                repl_mask[j] = False
        # drop second stream of 2 streams that were merged
        streams = [stream for stream, m in zip(streams, repl_mask) if m]
        # set order of streams so universal
        stream_sort = np.argsort([hash(stream) for stream in streams])        
        self.__streams = tuple(streams[i] for i in stream_sort)
        self.__hash = hash(self.__streams) if len(self.__streams) > 1 else hash(self.__streams[0])
    
    def __hash__(self):
        return self.__hash
    
    def __str__(self):
        string = str()
        for stream in self.__streams:
            string += stream.__str__() + "_"
        string = string[:-1]
        return string
    
    def __repr__(self):
        rep = 'Ph_sel: ' + self.__streams[0].__str__() + '\n'
        for stream in self.__streams[1:]:
            rep += '        ' + stream.__str__() + '\n'
        rep = rep[:-1]
        return rep
    
    def __len__(self):
        return len(self.__streams)
    
    def __eq__(self, other):
        if isinstance(other, (Ph_stream, Ph_sel)):
            return hash(self) == hash(other)
    
    def __contains__(self, other):
        compare = (other, ) if isinstance(other, Ph_stream) else other
        contain = False
        for comp in compare:
            for stream in self.__streams:
                if comp in stream:
                    contain = True
                    break
            if contain:
                break
        return contain
    
    @property
    def Dex(self):
        return np.any([stream.Dex for stream in self.__streams])
    
    @property
    def Dem(self):
        return np.any([stream.Dem for stream in self.__streams])
    @property
    def Aex(self):
        return np.any([stream.Aex for stream in self.__streams])
    
    @property
    def Aem(self):
        return np.any([stream.Dem for stream in self.__streams])
    
    @property
    def streams(self):
        return self.__streams
            
    def get_det(self, stream_map):
        dets = self.__streams[0].get_det(stream_map)
        for stream in self.__streams[1:]:
            dets = np.union1d(dets, stream.get_det(stream_map))
        return tuple(dets)
    
    def get_mask(self, stream_map, dets):
        mask = np.zeros(dets.shape, dtype=bool)
        for det in self.get_det(stream_map):
            mask += det == dets
        return mask
    
    