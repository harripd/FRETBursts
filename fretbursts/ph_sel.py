#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2020
#               Paul David Harris <harripd@gmail.com>
#
"""
Created on Sat Aug  7 20:36:22 2021

@author: Paul David Harris
"""

import warnings
import numpy as np
import re

def _process_dict(ph_dict):
    char_map = dict(D=np.array([0],dtype=np.uint8),A=np.array([1],dtype=np.uint8),DA=None,
                    P=np.array([0],dtype=np.uint8),S=np.array([1],dtype=np.uint8),
                    par=np.array([0],dtype=np.uint8),per=np.array([1],dtype=np.uint8))
    stream_dict = []
    for ph in ph_dict:
        stream_map = dict(ex=None,em=None,pol=None,split=None)
        for key, val in ph.items():
            if key not in stream_map:
                raise ValueError(f"Stream {key} not valid")
            elif str(val) in char_map:
                stream_map[key] = char_map[val]
            elif val is not None:
                stream_map[key] = np.atleast_1d(val).astype(np.uint8)
        stream_dict.append(stream_map)
    return stream_dict

def _process_str(ph_str):
    stream_map = dict(ex=None,em=None,pol=None,split=None)
    # dictionary of regex that each match the definitiion of a particular stream parameter
    stream_regex = dict(em = re.compile(r'((\[(\d+,)*\d+\])|(\d+|A|D|DA))em'),
                    ex = re.compile(r'((\[(\d+,)*\d+\])|(\d+|A|D|DA))ex'), 
                    pol = re.compile(r'(S|P|par|per|0|1)pol'), 
                    split = re.compile(r'\d+split')) # list of regex for stream identifiers
    char_map = dict(D=np.array([0],dtype=np.uint8),A=np.array([1],dtype=np.uint8),DA=None,
                    P=np.array([0],dtype=np.uint8),S=np.array([1],dtype=np.uint8),
                    par=np.array([0],dtype=np.uint8),per=np.array([1],dtype=np.uint8))
    str_match = dict(em=re.compile(r'(\d+|DA|D|A)'),ex=re.compile(r'(\d+|DA|D|A)'),
                     pol=re.compile(r'([01SP]|par|per)'),split=re.compile(r'[01]'))
    stream_split = re.compile(r'(((\[(\d+,)*\d+\])|(\d+|A|D|DA))em|((\[(\d+,)*\d+\])|(\d+|A|D|DA))ex|(S|P|par|per|0|1)pol|\d+split)+[,_]?')
    trim = re.compile('[,_]$')
    if ph_str[0] == 'all':
        return [stream_map]
    stream_dict = []
    for ph_unsplit in ph_str:
        ph_untrim = [match.group() for match in stream_split.finditer(ph_unsplit)]
        ph_re = ''
        for ph_string in ph_untrim:
            ph_re += ph_string
        if ph_re != ph_unsplit:
            raise ValueError("Unreadable string definition")
        ph_split = [trim.sub('',ph_string) for ph_string in ph_untrim]
        for ph_id in ph_split:
            stream_temp = dict(ex=None,em=None,pol=None,split=None)
            for str_key, regex in stream_regex.items():
                for i, str_iter in enumerate(regex.finditer(ph_id)):
                    ems = np.empty(0,dtype=np.uint8)
                    if i > 0:
                        raise ValueError(f"Cannot specify more than one {str_key} per stream")
                    else:
                        for arr in str_match[str_key].finditer(str_iter.group()):
                            if arr.group() in char_map:
                                ems = char_map[arr.group()]
                            else:
                                ems = np.append(ems,np.uint8(arr.group()))
                        stream_temp[str_key] = ems
                ph_id = regex.sub('',ph_id)
            if len(ph_id) != 0:
                raise ValueError(f'Could not process stream ID, unidentified characters: {ph_id}')
            stream_dict.append(stream_temp)
    return stream_dict

def _process_kwargs(ph_kwargs):
    stream_map = dict(ex=None,em=None,pol=None,split=None)
    char_map = dict(D=np.array([0],dtype=np.uint8),A=np.array([1],dtype=np.uint8),DA=None,
                    P=np.array([0],dtype=np.uint8),S=np.array([1],dtype=np.uint8),
                    par=np.array([0],dtype=np.uint8),per=np.array([1],dtype=np.uint8))
    depr_keys = ['Dex','Aex'] # to be deprecated 
    depr_count = 0
    # loop might be deprecated, used for converting Dex/Aex arguments to em/ex/pol/split type
    for key, val in ph_kwargs.items():
        if key in depr_keys:
            stream_map = []
            depr_count += 1
            stream_map_temp = dict(ex=None,em=None,pol=None,split=None)
            if key == 'Dex':
                stream_map_temp['ex'] = np.array([0])
            else:
                stream_map_temp['ex'] = np.array([1])
            if val == 'Dem':
                stream_map_temp['em'] = np.array([0])
            elif val == 'Aem':
                stream_map_temp['em'] = np.array([1])
            elif val == 'DAem':
                stream_map_temp['em'] = None
            else:
                raise ValueError(f"Invalid {key} value")
            if 'Pol' in ph_kwargs:
                if ph_kwargs['Pol'] == 'Pem':
                    stream_map_temp['pol'] = 0
                elif ph_kwargs['Pol'] == 'Sem':
                    stream_map_temp['pol'] = 1
                elif ph_kwargs['Pol'] == 'SPem' or ph_kwargs['Pol'] == 'PSem' :
                    stream_map_temp['pol'] = None
                else:
                    raise ValueError("Invalid Pol value")
            stream_map.append(stream_map_temp)
        elif depr_count != 0:
            raise ValueError("Cannot mix Dex/Aex and em/ex/pol arguments")
    # check if the old method was used, otherwise, process with the new method
    if depr_count != 0:
        warnings.warn("Using kwargs as inputs will be depricated in future release")
    else:
        for key, val in ph_kwargs.items():
            if key not in stream_map:
                raise ValueError(f"Cannot process kwarg: {key}")
            if val in char_map:
                stream_map[key] = char_map[val]
            else:
                stream_map[key] = int(val)
        stream_map = [stream_map]
    return stream_map

def _invert_key(stream_key,stream_val):
    invert_dict = {'ex':{0:'D',1:'A'},'em':{0:'D',1:'A'},'pol':{0:'P',1:'S'},'split':{0:'0',1:'1'}}
    if stream_val.size != 1:
        out = '['
        for i, val in enumerate(stream_val):
            out += str(val)
            if i +1 < stream_val.size:
                out += ','
        out += ']'
        return out
    elif stream_val[0] in invert_dict[stream_key]:
        return invert_dict[stream_key][stream_val[0]]
    else:
        return str(stream_val[0])

class Ph_sel():
    def __init__(self,*args,**kwargs):
        if len(args) != 0 and len(kwargs) != 0:
            raise ValueError("Cannot specify Ph_sel with both arguments and keyword arguments")
        if len(args) == 0: # case of only kwargs, will be deprecated
            self._stream_dict = _process_kwargs(kwargs)
        elif np.all([type(arg)==dict for arg in args]): # case when set of dicts passed 
            self._stream_dict = _process_dict(args)
        elif np.all([type(arg)==str for arg in args]): # case when set of strings passed (prefered method)
            self._stream_dict = _process_str(args)
        else:
            raise ValueError("Must specify as strings, dictionaries, or Keyword Arguments")
        # check if multiple streams define same streams
        self._stream_dict = [stream for n, stream  in enumerate(self._stream_dict) if stream not in self._stream_dict[:n]]
    def get_det(self,data_map):
        streams = np.empty(0,dtype=np.uint8)
        for stream_dict in self._stream_dict:
            stream_num = data_map['all_streams']
            for key, val in stream_dict.items():
                if val is not None and key not in data_map:
                    raise ValueError(f"{key} not implemented for this Data object, must use different Ph_sel object")
                elif val is not None:
                    if val.max() > len(data_map[key]): raise ValueError("Detector number not impolemented")
                    vals = [data_map[key][v] for v in val]
                    stream_num = np.intersect1d(stream_num,vals)
            streams = np.concatenate((streams,stream_num))
        return streams
    def get_mask(self,data_map,dets):
        det_id = self.get_det(data_map)
        stream_mask = np.zeros(dets.shape,dtype=bool)
        for index in det_id:
            stream_mask += dets == index
        return stream_mask
    def __str__(self):
        out = ''
        for i, stream_dict in enumerate(self._stream_dict):
            if stream_dict == {'ex':None,'em':None,'pol':None,'split':None}:
                return 'all'
            for stream_key, stream_val in stream_dict.items():
                if stream_val is not None:
                    out += _invert_key(stream_key,stream_val) + stream_key
            if i+1 < len(self._stream_dict):
                out += '_'
        return out
    def __len__(self):
        return len(self._stream_dict)
    def __eq__(self,comp_sel):
        if len(self) != len(comp_sel):
            return False
        else:
            res = [True if c  in self._stream_dict else False for c in comp_sel._stream_dict]
            return np.all(res)