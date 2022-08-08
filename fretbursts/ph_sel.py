#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Paul David Harris
# created: Aug 1 2022
"""
This module is dedicated to handling and specifying *photon streams* in an abstract way.
*Photon streams* are defined by the nature of the detector, and during which excitation 
period the photon arrived, *photon streams*.
While detectors are stored as simple indices, during conversion *to PhotonHDF5*
the setup of the detectors was defined.

This assigns photons some of the following categories:

=============    =============================================
Detector type    Descrition
=============    =============================================
ex               excitation period (usually specrally defined)
em               emmission spectral channel of arriving photon
pol              Polarization of emission
split            detector part of split channel
=============    =============================================

These streams are given integer indices. For *ex* and *em* these can be arbitrarily
large to support 3+ color setups. 
*pol* and *split* each can only be indexed as 0 or 1.

For *ex/em* index 0 corresponds to Donor, and index 1 to Acceptor (for standard 2 color setups)
For *pol* index 0 corresponds to parallel emission, and index 1 to perpendicular emission.

To make code more readabe, *ex/em* and *pol* indices have default aliasses.

=====   =====   ========
Type    Index   Aliases
=====   =====   ========
ex/em   0       D
ex/em   1       A
pol     0       P or par
pol     1       S or per
=====   =====   ========

Photon selectiosn are achieved through 2 classes, a base class the user usually doesn't
interact with, and a wrapper class that is the general way of specifying and selecting
photon streams.

#. :class:`Ph_stream` the foundational photon selection class, which defines a set of detectors with certain traits in common
#. :class:`Ph_sel` the higher level class, which allows any arbitrary selection of streams, by concatenating multiple :class:`Ph_stream` objects

With :class:`Ph_sel` , any stream can be specified, and any combination can be specified.

The synax is `[streamcode1]typecode1[streamcode2]typecode2...` ::
    
    PhDA = Ph_sel('DexAem')

.. note::
    
    The above is the new form of specifying ::
        
        Ph_sel(Dex='Aem')
    
    from version 0.7


Selections are defined by strings, and if you want the union of several selections
you can specify multiple sub-streams by separating them with a single underscore::
    
    Ph_sel('DexDem_AexAem')
    
If any of the *ex*, *em*, *pol*, or *split* are not specified, then the selection
will not disitinguish photons based on that stream.

For instance::
    
    Ph_sel('Dex')
    
will take all photons during Donor excitation, regarless of whether they came in 
the Donor or Acceptor emission channels, or polarization, or split.

Finally, :class:`Ph_stream` and :class:`Ph_sel` are immutable and hashable, and 
thus can be used as dictionary keys.
"""

import re
import numpy as np
import functools
import warnings
from itertools import combinations
from collections.abc import Iterable

from .utils.misc import intersect_multi, union_multi

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

_short_dict_exem = {None:'n', 0:'d', 1:'a'}

_short_dict_pol = {None:'', 0:'p', 1:'s'}

_short_dict_split = {None:'', 0:'0', 1:'1'}

_short_dict = {'ex':_short_dict_exem, 'em':_short_dict_exem,
               'pol':_short_dict_pol, 'split':_short_dict_split}

_ph_keys = {'ex','em','pol','split'}

_stream_regex = dict(em = re.compile(r'((\[(\d+,)*\d+\])|(\d+|A|D|DA))em'), 
                    ex = re.compile(r'((\[(\d+,)*\d+\])|(\d+|A|D|DA))ex'), 
                    pol = re.compile(r'(S|P|par|per|0|1)pol'), 
                    split = re.compile(r'(\d+)split')) # list of regex for stream identifiers

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

def _fuse_splitpol(tup):
    if tup is not None:
        if np.any([t > 1 for t in tup]):
            raise ValueError('pol and split must be None, 0 or 1')
        elif len(tup) > 1:
            tup = None
    return tup


class Ph_stream:
    """
    Base class defining a uniform set of of photon streams.
    
    Usually used only by :class:`Ph_sel`, and not by the user 
    """
    __slots__ = ('__ex', '__em', '__pol', '__split', '__hash')
    @_check_kwargs
    def __init__(self, ex=None, em=None, pol=None, split=None):
        pol = _fuse_splitpol(pol)
        split = _fuse_splitpol(split)
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
        if len(string) == 0:
            string = 'all'
        return string
    
    def __repr__(self):
        return "Ph_stream: " + self.__str__()
    
    @property
    def ex(self):
        """Tuple of int representing the excitation channel(s) of selection"""
        return self.__ex
    
    @property
    def em(self):
        """Tuple of int representing the emission spectral channel(s) of selection"""
        return self.__em
    
    @property
    def pol(self):
        """Tuple indicating polarization, (0,) for parallel, (1,) for perpendicular, and None if both"""
        return self.__pol
    
    @property
    def split(self):
        """Tuple (0,) (1,) for split channel or None if both"""
        return self.__split
    
    @property
    def Dex(self):
        """Boolean if stream in incldued Donor excitation"""
        if self.__ex is None:
            return True
        else:
            return 0 in self.__ex
    
    @property
    def Aex(self):
        """Boolean if stream in incldued acceptor excitation"""
        if self.__ex is None:
            return True
        else:
            return 1 in self.__ex
    
    @property
    def Dem(self):
        """Boolean if stream in incldued Donor emission"""
        if self.__em is None:
            return True
        else:
            return 0 in self.__em
    
    @property
    def Aem(self):
        """Boolean if stream in incldued Acceptor emission"""
        if self.__em is None:
            return True
        else:
            return 0 in self.__em
    
    @_check_kwargs
    def update(self, **kwargs):
        """Return a new :class:`Ph_stream` merging input stream"""
        mkdict = {key:val for key, val in self}
        mkdict.update(**kwargs)
        return Ph_stream(**mkdict)
    
    def get_det(self, stream_map):
        """Returns the detector indice(s) of :class:`Ph_stream` based on stream_map dictionary"""
        # get streams of importance
        try:
            streams = [np.concatenate([stream_map[stream][idx] for idx in idxs]) for stream, idxs in self]
        except KeyError as k:
            raise ValueError(f"Data does not differntiate {k} photons")
        if len(streams) > 1:
            streams = tuple(intersect_multi(*streams))
        elif len(streams) == 1:
            streams = tuple(streams[0])
        else:
            streams = tuple(stream_map['all_streams'])
        return streams
    
    def get_mask(self, stream_map, dets):
        """Returns mask of photons in dets of the stream in :class:`Ph_stream` based on stream_map dictionary"""
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
    if "Dex" in ph_dict or "Aex" in ph_dict:
        if len(ph_dict) > 1 and  not ("Dex" in ph_dict and "Aex" in ph_dict):
            raise ValueError("Cannot mix old 'Dex'/'Aex' kwargs with new 'em'/'ex' kwargs")
        warnings.warn("Specifiying Ph_sel with Dex/Aex kwargs discourated, and will be deprecated. Prefered method to specify with string",
                      DeprecationWarning)
        ph_dict = {key:val for key, val in ph_dict.items() if val is not None}
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
    """
    Identify if streams are "parallel" meaning they could be merged into a single
    stream, and return dictioary specifying the union of thes treams

    Parameters
    ----------
    sel0 : Ph_stream
        First stream to compare.
    sel1 : Ph_stream
        First stream to compare.

    Raises
    ------
    TypeError
        Wrong data type given.

    Returns
    -------
    par : bool
        If streams are parallel or not, if True, then udict will represent the
        merging of the two streams, otherwise, udict will include additional streams
    udict : dict
        dictionary the can be used as input to make new Ph_stream.

    """
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
    """
    Class used to describe a selection of photons
    Defined by a string, specifying index and type of index, repeated for as many
    types of indices desired. Then for another non-paralllel definition, an 
    underscore followed by the next definition.
    
    
    """
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
        compare = (other, ) if isinstance(other, Ph_stream) else other.streams
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
        """Bool if any stream includes Donor excitation"""
        return np.any([stream.Dex for stream in self.__streams])
    
    @property
    def Dem(self):
        """Bool if any stream includes Donor emission"""
        return np.any([stream.Dem for stream in self.__streams])
    @property
    def Aex(self):
        """Bool if any stream includes Acceptor excitation"""
        return np.any([stream.Aex for stream in self.__streams])
    
    @property
    def Aem(self):
        """Bool if any stream includes Acceptor emission"""
        return np.any([stream.Dem for stream in self.__streams])
    
    @property
    def pol(self):
        """Which polarization stream specified.
        None indicates no polization ever selected,
        False inidcates inconsistent polarization selected in streams
        Otherwise (0,) for parallel, and (1,) for perpendicular
        """
        pol = self.__streams[0].pol
        pols = [stream.pol == pol for stream in self.__streams]
        pol = pol[0] if pol is not None else None
        if not np.all(pols):
            pol = False
        return pol
    
    @property
    def ex(self):
        """Union of all excitation stream indices specified in all streams in this :calss:`Ph_sel`"""
        ex = tuple()
        for stream in self.__streams:
            ex = set(ex) | set(stream.ex)
        return tuple(ex)
    
    @property
    def em(self):
        """Union of all emmision stream indices specified in all streams in this :class:`Ph_sel`"""
        em = tuple()
        for stream in self.__streams:
            em = set(em) | set(stream.em)
        return tuple(em)
    
    @property
    def streams(self):
        """List of Ph_stream objects that define the :class:`Ph_sel`"""
        return self.__streams
    
    @property
    def short_name(self):
        """Short string representation of :class:`Ph_sel`, does not incldue full em/ex etc."""
        name = str()
        if np.any([stream == Ph_sel('all') for stream in self.__streams]):
            name = 'all'
        elif len(self.__streams) == 1:
            single = np.all([len(val) < 2 for _, val in self.__streams[0]])
            norm = np.all(np.concatenate(([val for _, val in self.__streams[0]])) < 2)
            if single and norm:
                for stream, val in self.__streams[0]:
                    name += _short_dict[stream][val[0]]
            else:
                name += self.__str__()
        else:
            name += self.__str__()
        return name
            
    def get_det(self, stream_map):
        """Returns the detector indice(s) of :class:`Ph_sel` based on stream_map dictionary"""
        dets = union_multi(*[stream.get_det(stream_map) for stream in self.__streams])
        return tuple(dets)
    
    def get_mask(self, stream_map, dets):
        """Returns mask of photons in dets of photons in :class:`Ph_sel` based on stream_map dictionary"""
        mask = np.zeros(dets.shape, dtype=bool)
        for det in self.get_det(stream_map):
            mask += det == dets
        return mask
    
    