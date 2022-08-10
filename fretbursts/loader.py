#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014-2016 The Regents of the University of California,
#               Antonino Ingargiola <tritemio@gmail.com>
#
"""
The `loader` module contains functions to load each supported data format.
The loader functions load data from a specific format and
return a new :class:`fretbursts.burstlib.Data()` object containing the data.

This module contains the high-level function to load a data-file and
to return a `Data()` object. The low-level functions that perform the binary
loading and preprocessing can be found in the `dataload` folder.
"""

import os
import re
import numpy as np
import tables

from phconvert.smreader import load_sm
from .dataload.spcreader import load_spc
from .burstlib import Data
from .utils.misc import selection_mask
from . import loader_legacy
import phconvert as phc

import logging
log = logging.getLogger(__name__)


def _is_multich(h5data):
    if 'photon_data' in h5data:
        return False
    elif 'photon_data0' in h5data:
        return True
    else:
        msg = 'Cannot find a photon_data group.'
        raise phc.hdf5.Invalid_PhotonHDF5(msg)


def _append_data_ch(d, name, value):
    if name not in d:
        d.add(**{name: [value]})
    else:
        d[name].append(value)


def _load_from_group(d, group, name, dest_name, multich_field=False,
                     ondisk=False, allow_missing=True):
    if allow_missing and name not in group:
        return

    node_value = group._f_get_child(name)
    if not ondisk:
        node_value = node_value.read()
    if multich_field:
        _append_data_ch(d, dest_name, node_value)
    else:
        d.add(**{dest_name: node_value})


def _append_empty_ch(data):
    # Empty channel, fill it with empty arrays
    ph_times = np.array([], dtype='int64')
    _append_data_ch(data, 'ph_times_m', ph_times)

    a_em = np.array([], dtype=bool)
    _append_data_ch(data, 'A_em', a_em)


def _get_measurement_specs(ph_data, setup):
    meas_dict = dict(spec=2,pol=False,split=False,alt=False,splitspec=False,
                     lifetime=False, PAX=False)
    if 'measurement_specs' not in ph_data:
        # No measurement specs, we will load timestamps and set them all in a
        # conventional photon stream (acceptor emission)
        meas_type = 'smFRET-1color'
        meas_specs = None
    else:
        assert 'measurement_type' in ph_data.measurement_specs
        meas_specs = ph_data.measurement_specs
        meas_type = meas_specs.measurement_type.read().decode()
    
    num_spectral_ch = setup.num_spectral_ch.read()
    num_polarization_ch = setup.num_polarization_ch.read()
    num_split_ch = setup.num_split_ch.read()
    if num_split_ch > 1 and num_spectral_ch == 1:
        # in this case data will be loaded as "spectral".
        log.warning('Loading split channels as spectral channels.')
        meas_dict['splitspec'] = True
        meas_dict['spec'] = num_split_ch
        num_spectral = num_split_ch
    else:
        num_spectral = num_spectral_ch
        meas_dict['spec'] = num_spectral
        meas_dict['split'] = True if num_split_ch == 2 else False
        if num_split_ch > 2:
            raise phc.hdf5.Invalid_PhotonHDF5(
                "The field `/setup/num_split_ch` indicates {num_split_ch} split channels, can only support 1 or 2")
    
    if num_polarization_ch > 2:
        raise phc.hdf5.Invalid_PhotonHDF5(
            f"The field `/setup/num_polarization_ch` indicates {num_polarization_ch} polarization channels, can only support 1 or 2")
    
    alt_period = np.zeros(num_spectral,dtype=bool)
    for i in range(num_spectral):
        alt_period[i] = True if 'alex_excitation_period%d' % (i+1) in meas_specs else False
    meas_dict['alt'] = alt_period
    if alt_period.size ==1 and alt_period == np.array([False,True],dtype=bool):
        meas_dict['PAX'] = True
        print("Warning, this verion of FRETBursts has not been tested with PAX measuremnts")
    if setup.num_polarization_ch.read() == 2:
        meas_dict['pol'] = True
    elif setup.num_polarization_ch.read() == 1:
        meas_dict['pol'] = False
    else:
        raise phc.hdf5.Invalid_PhotonHDF5(
            f"The field /setup/num_polarization_ch indicates {num_polarization_ch} channels, maxiumum of 2 allowed")
    meas_dict['lifetime'] = True if setup.lifetime.read() else False
    # Check consistency of polarization specs
    if meas_specs is not None:
        det_specs = meas_specs.detectors_specs
        if setup.num_polarization_ch.read() == 1:
            if (
                'polarization_ch1' in det_specs
                and 'polarization_ch2' in det_specs
            ):
                msg = ("The field `/setup/num_polarization_ch` indicates "
                       "no polarization.\nHowever, the fields "
                       "`detectors_specs/polarization_ch*` are present.")
                raise phc.hdf5.Invalid_PhotonHDF5(msg)
        else:
            if not (
                'polarization_ch1' in det_specs
                and 'polarization_ch2' in det_specs
            ):
                msg = ("The field `/setup/num_polarization_ch` indicates "
                       "more than one polarization.\nHowever, some "
                       "`detectors_specs/polarization_ch*` fields are "
                       "missing.")
                raise phc.hdf5.Invalid_PhotonHDF5(msg)
    
    # Check if the measurement type is a valid string
    # regex matches first smFRET or generic, then checks for subsequent ALEX, the number of colors, then polarization
    valid_regex = re.compile(r'(smFRET|generic|PAX)((-(ns|us)ALEX)?(-(\d+)(color|c))?(-([12])?pol)?)?') 
    meas_regex = valid_regex.match(meas_type)
    if meas_regex is None or meas_regex.group() != meas_type:
        raise NotImplementedError('Measurement type "%s" not supported'
                                  ' by FRETBursts.' % meas_type)
    if meas_regex.group(1) == 'smFRET': 
        # smFRET, check for consistency between colors, polarization etc.
        meas_type_colors = 2 if meas_regex.group(6) is None else int(meas_regex.group(6))
        meas_type_pol = 1 if meas_regex.group(8) is None else int(meas_regex.group(9))
        if meas_type_colors != num_spectral:
            log.warning(f"Measurement type and num_spectral_ch indicate different number of colors: {meas_type_colors} and {num_spectral}")
        if meas_regex.group(8) is None and num_polarization_ch != 1:
            print("Polarization supported, but not included in measurement type definition")
        elif meas_type_pol != num_polarization_ch:
            log.warning(f"Measurement type and num_polarization_ch indicate different number of colors: {meas_type_pol} and {num_polarization_ch}")
        
        # add details on the data to measurement type, ALEX, color, polarization
        if meas_regex.group(4) is not None:
            meas_type += '-nsALEX' if meas_regex.group(4) == 'ns' else '-usALEX'
        if num_polarization_ch == 2:
            meas_type += '-2pol'
        elif num_polarization_ch > 2:
            raise phc.hdf5.Invalid_PhotonHDF5("The field '/setup/num_polarization_ch'"
                                              "Indicates more than two polarizations.\n")
    # Currently checking not implemented for PAX measurements
    elif meas_regex.group(1) == 'PAX':
        meas_dict['PAX'] = True
        print("Warning, this verion of FRETBursts has not been tested with PAX measuremnts")
        
    return meas_dict, meas_specs, meas_type


def _load_photon_data_arrays(data, ph_data, ondisk=False):
    assert 'timestamps' in ph_data

    # Build mapping to convert Photon-HDF5 to FRETBursts names
    # fields not mapped use the same name on both Photon-HDF5 and FRETBursts
    mapping = {'timestamps': 'ph_times_m','detectors': 'det_m',
               'nanotimes': 'nanotimes', 'particles': 'particles'}
    if data.alternated:
        mapping = {'timestamps': 'ph_times_t', 'detectors': 'det_t',
                   'nanotimes': 'nanotimes_t', 'particles': 'particles_t'}

    # Load all photon-data arrays
    for name in ph_data._v_leaves:
        dest_name = mapping.get(name, name)
        _load_from_group(data, ph_data, name, dest_name=dest_name,
                         multich_field=True, ondisk=ondisk)

    # Timestamps are always present, and their units are always present too
    data.add(clk_p=ph_data.timestamps_specs.timestamps_unit.read())


def _load_nanotimes_specs(data, ph_data):
    nanot_specs = ph_data.nanotimes_specs
    nanotimes_params = {}
    for name in ['tcspc_unit', 'tcspc_num_bins', 'tcspc_range']:
        value = nanot_specs._f_get_child(name).read()
        nanotimes_params.update(**{name: value})
    if 'user' in nanot_specs:
        for name in ['tau_accept_only', 'tau_donor_only',
                     'tau_fret_donor', 'inverse_fret_rate']:
            if name in nanot_specs.user:
                value = nanot_specs.user._f_get_child(name).read()
                nanotimes_params.update(**{name: value})
    _append_data_ch(data, 'nanotimes_params', nanotimes_params)

def _add_usALEX_specs(data, meas_specs,ich):
    # Used for both us-ALEX and PAX
    try:
        offset = meas_specs.alex_offset.read()
    except tables.NoSuchNodeError:
        log.warning('    No offset found, assuming offset = 0.')
        offset = 0
    _append_data_ch(data,'offset',offset)
    _append_data_ch(data,'alex_period',meas_specs.alex_period.read())
    _load_alex_periods_donor_acceptor(data, meas_specs,ich)


def _load_alex_periods_donor_acceptor(data, meas_specs,ich):
    # Used for both us- and ns-ALEX and PAX
        # Try to load alex period definitions
    alt_ON = []
    for i in range(1,data.num_colors+1):
        if 'alex_excitation_period%d' % i in meas_specs:
            alt_ON.append(meas_specs['alex_excitation_period%d' % i].read())
        else:
            alt_ON.append([])

            # But if it fails it's OK, those fields are optional
            msg = f"""
            The current file lacks the alternation period definition for
            stream {i}.
            You will need to manually add this info using:
    
              d.add(alt_ON[ich][{i}]=alt_ON)
    
            where `d` is a Data object and D_ON/A_ON is a tuple with start/stop
            values defining the D/A excitation excitation period. Values are in
            raw timestamps units.
            """
            log.warning(msg)
    _append_data_ch(data, 'alt_ON', alt_ON)



# This is deprecated, switch to make the stream_dict
# FAIRLY sure this should be deprecated, and replaced with the _generate_stream_dict
# def _compute_acceptor_emission_mask(data, ich, ondisk):
#     """For non-ALEX measurements."""
#     if data.detectors[ich].dtype.itemsize != 1:
#         raise NotImplementedError('Detectors dtype must be 1-byte.')
#     donor, accept = data._det_donor_accept_multich[ich]

#     # Remove counts not associated with D or A channels
#     det_ich = data.detectors[ich][:]  # load the data in case ondisk = True
#     num_detectors = len(np.unique(det_ich))
#     if not ondisk and num_detectors > donor.size + accept.size:
#         mask = (selection_mask(det_ich, donor) +
#                 selection_mask(det_ich, accept))
#         data.detectors[ich] = det_ich[mask]
#         data.ph_times_m[ich] = det_ich[mask]
#         if 'nanotimes' in data:
#             data.nanotimes[ich] = data.nanotimes[ich][:][mask]

#     # From `detectors` compute boolean mask `A_em`
#     if not ondisk and donor.size == 1 and 0 in (accept, donor):
#         # In this case we create the boolean mask in-place
#         # using the detectors array
#         _append_data_ch(data, 'A_em', data.detectors[ich].view(dtype=bool))
#         if accept == 0:
#             np.logical_not(data.A_em[ich], out=data.A_em[ich])
#     else:
#         # Create the boolean mask as a new array
#         _append_data_ch(data, 'A_em',
#                         selection_mask(det_ich, accept))


def _compute_stream_dict(data, ich):
    """Computer _stream_map, mainly for non-alternated data"""
    if data.det_m[ich].dtype.itemsize !=1:
        raise NotImplementedError("Detectors dtype must be 1 byte")
    det_ich = data.det_m[ich]
    unique_det = np.unique(det_ich)
    spec_map = data.det_spectral[ich] if hasattr(data,'det_spectral') else None
    pol_map = data.det_p_s_pol[ich] if hasattr(data,'det_p_s_pol') else None
    split_map = data.det_split[ich] if hasattr(data,'det_split') else None
    spec_union = _union_of_list(spec_map)
    pol_union = _union_of_list(pol_map)
    split_union = _union_of_list(split_map)
    det_union = np.empty(0, dtype=np.uint8)
    union_list = [spec_union, pol_union, split_union]
    for i in range(len(union_list)):
        if union_list[i].size == 0:
            union_list[i] = det_union
        elif det_union.size == 0:
            det_union = union_list[i]
        elif det_union.size != union_list[i].size:
            det_union = np.union1d(det_union, union_list[i])
            log.warning("Incomplete mapping of detectors in HDF5 file to Spectral/Polarization/split channels")
        elif np.any(det_union != union_list[i]):
            log.warning("Some detector indexes lack a Spectral, Polarization or Split identity, likely one or more channels overdefined")
            det_union = np.union1d(det_union, union_list[i])
    if det_union.size < unique_det.size:
        raise ValueError("Underdefined detector indeces in HDF5 file")
    elif det_union.size != np.union1d(det_union,unique_det).size:
        raise ValueError("Undefined detector indeces in HDF5 file")
    stream_map = {'ex':(det_union, )}
    if spec_map is not None: stream_map['em'] = spec_map
    if pol_map is not None: stream_map['pol'] = pol_map
    if split_map is not None: stream_map['split'] = split_map
    stream_map['all_streams'] = det_union
    _append_data_ch(data, '_stream_map', stream_map)


_IPH_error_msg = "/photon_data{nchan} conflicts with other /ph_data groups in field {name}"

_IPH_error_dict = {'alt':'alternated', 'nanotimes':'lifetime', 'pol':'polarization',
              'spec':'spectral', 'PAX':'ALEX'}

def _check_meas_dict(meas_dict0, meas_dict1, ich):
    for key0, val0 in meas_dict0.items():
        if not np.any(meas_dict1[key0] != val0):
            phc.hdf5.Invalid_PhotonHDF5(_IPH_error_msg.format(nchan=ich, name=_IPH_error_dict.get(key0, key0)))

def _photon_hdf5_1ch(h5data, data, ondisk=False, nch=1, ich=0, loadspecs=True):
    
    ph_data_name = '/photon_data' if nch == 1 else '/photon_data%d' % ich
    ph_data = h5data._f_get_child(ph_data_name)
    # Handle the case of missing channel (e.g. dead pixel)
    if ph_data_name not in h5data:
        _append_empty_ch(data)
        return
    # fields that are universal to a data-object, so only add once, at the beginning
    if ich == 0:
        data.add(nch=nch)
        # Load photon_data group and measurement_specs (if present)
        meas_dict, meas_specs, meas_string = _get_measurement_specs(ph_data, h5data.setup)
        # regex identifies number of colors in the meas_type
        # Set some `data` flags
        data.add(meas_type=meas_string)
        data.add(meas_dict=meas_dict)
        data.add(ALEX= np.any(meas_dict['alt']) and not meas_dict['PAX'])  # True for usALEX, nsALEX, but not PAX
        data.add(alternated=np.any(meas_dict['alt'])) # True for usALEX, nsALEX and PAX
        data.add(lifetime='nanotimes' in ph_data)
        data.add(polarization= meas_dict['pol'])
        data.add(spectral= meas_dict['spec'] > 1)
        data.add(num_colors = meas_dict['spec'])
    else: # check that all data fields have matching types
        meas_dict, meas_specs, meas_string = _get_measurement_specs(ph_data, h5data.setup)
        # regex identifies number of colors in the meas_type
        if data.nch != nch:
            raise ValueError("nch changed between channel loads")
        _check_meas_dict(data.meas_dict, meas_dict, ich)
    
    # Load photon_data arrays
    _load_photon_data_arrays(data, ph_data, ondisk=ondisk)

    # If nanotimes are present load their specs
    if data.lifetime:
        _load_nanotimes_specs(data, ph_data)

    # Unless 1-color, load donor and acceptor info
    det_specs = meas_specs.detectors_specs
    if data.spectral:
        spec_chans = []
        if not meas_dict['splitspec']:
            for ch in range(1,data.num_colors+1):
                spec_chans.append(np.atleast_1d(det_specs['spectral_ch%d' % ch ].read()))
        else:
            for ch in range(data.num_colors):
                spec_chans.append(np.atleast_1d(det_specs.split_ch1.read()))
        _append_data_ch(data, 'det_spectral', tuple(spec_chans))

    if data.polarization:
        pol_chans = (np.atleast_1d(det_specs.polarization_ch1.read()), 
                     np.atleast_1d(det_specs.polarization_ch2.read()))
        _append_data_ch(data, 'det_p_s_pol', pol_chans)
    
    if data.meas_dict['split']:
        split_chans = []
        for sp in range(1,data.meas_dict['split']+1):
            split_chans.append(np.atleast_1d(det_specs['split_ch%d' % sp].read()))
        _append_data_ch(data,'det_split',tuple(split_chans))
    
    if data.spectral and not data.alternated:
        # No alternation, we can compute the emission masks right away
        _compute_stream_dict(data, ich)

    if loadspecs and data.spectral and data.alternated and not data.lifetime:
        # load alternation metadata for usALEX or PAX
        _add_usALEX_specs(data, meas_specs,ich)

    if loadspecs and data.lifetime:
        data.add(laser_repetition_rate=meas_specs.laser_repetition_rate.read())
        if data.ALEX:
            # load alternation metadata for nsALEX
            _load_alex_periods_donor_acceptor(data, meas_specs,ich)


def _photon_hdf5_multich(h5data, data, ondisk=True):
    ph_times_dict = phc.hdf5.photon_data_mapping(h5data._v_file)
    nch = np.max(list(ph_times_dict.keys())) + 1
    _photon_hdf5_1ch(h5data, data, ondisk=ondisk, nch=nch, ich=0)
    for ich in range(1, nch):
        _photon_hdf5_1ch(h5data, data, ondisk=ondisk, nch=nch, ich=ich,
                         loadspecs=False)


def photon_hdf5(filename, ondisk=False, require_setup=True, validate=False, fix_order=True):
    """Load a data file saved in Photon-HDF5 format version 0.3 or higher.

    Photon-HDF5 is a format for a wide range of timestamp-based
    single molecule data. For more info please see:

    http://photon-hdf5.org/

    Arguments:
        filename (str or pathlib.Path): path of the data file to be loaded.
        ondisk (bool): if True, do not load the timestamps in memory
            using instead references to the HDF5 arrays. Default False.
        require_setup (bool): if True (default) the input file needs to
            have a setup group or won't be loaded. If False, accept files
            with missing setup group. Use False only for testing or
            DCR files.
        validate (bool): if True validate the Photon-HDF5 file on loading.
            If False skip any validation.
        fix_order (bool): if True then reorder-photons so macrotimes are in
            perfect ascending order.

    Returns:
        :class:`fretbursts.burstlib.Data` object containing the data.
    """
    filename = str(filename)
    assert os.path.isfile(filename), 'File not found.'
    version = phc.hdf5._check_version(filename)
    if version == u'0.2':
        return loader_legacy.hdf5(filename)

    h5file = tables.open_file(filename)
    # make sure the file is valid
    if validate and version.startswith(u'0.4'):
        phc.v04.hdf5.assert_valid_photon_hdf5(h5file,
                                              require_setup=require_setup,
                                              strict_description=False)
    elif validate:
        phc.hdf5.assert_valid_photon_hdf5(h5file, require_setup=require_setup,
                                          strict_description=False)
    # Create the data container
    h5data = h5file.root
    d = Data(fname=filename, data_file=h5data._v_file)

    for grp_name in ['setup', 'sample', 'provenance', 'identity']:
        if grp_name in h5data:
            d.add(**{grp_name:
                     phc.hdf5.dict_from_group(h5data._f_get_child(grp_name))})

    for field_name in ['description', 'acquisition_duration']:
        if field_name in h5data:
            d.add(**{field_name: h5data._f_get_child(field_name).read()})

    if _is_multich(h5data):
        _photon_hdf5_multich(h5data, d, ondisk=ondisk)
    else:
        _photon_hdf5_1ch(h5data, d, ondisk=ondisk)
    if fix_order:
        if hasattr(d, 'ph_times_t'):
            for i in range(d.nch):
                order = np.argsort(d.ph_times_t[i])
                d.ph_times_t[i] = d.ph_times_t[i][order]
                d.det_t[i] = d.det_t[i][order]
                if d.lifetime:
                    d.nanotimes_t[i] = d.nanotimes_t[i][order]
        else:
            for i in range(d.nch):
                order = np.argsort(d.ph_times_m[i])
                d.ph_times_m[i] = d.ph_times_m[i][order]
                d.det_m[i] = d.det_m[i][order]
                if d.lifetime:
                    d.nanotimes[i] = d.nanotimes[i][order]
    if not ondisk:
        h5file.close()

    return d


##
# Multi-spot loader functions
#

##
# usALEX loader functions
#

# Build masks for the alternating periods
def _select_outer_range(times, period, edges):
    return ((times % period) >= edges[0]) + ((times % period) < edges[1])


def _select_inner_range(times, period, edges):
    return ((times % period) >= edges[0]) * ((times % period) < edges[1])


def _select_range(times, period, edges):
    """
    Retern a mask of photons within an excitation period for usALEX measurements
    Arguments
    ---------
    times : np.ndarray
        the times ( of photons
    period : int
        The alternation period (one full Dex/Dem cycle)
    edges : array-like
        The edges of the excitation window, 2 element array
    """
    return _select_inner_range(times, period, edges) if edges[0] < edges[1] \
        else _select_outer_range(times, period, edges)


def usalex(fname, leakage=0, gamma=1., header=None, BT=None):
    """Load usALEX data from a SM file and return a Data() object.

    This function returns a Data() object to which you need to apply
    an alternation selection before performing further analysis (background
    estimation, burst search, etc.).

    The pattern to load usALEX data is the following::

        d = loader.usalex(fname=fname)
        d.add(D_ON=(2850, 580), A_ON=(900, 2580), alex_period=4000)
        plot_alternation_hist(d)

    If the plot looks good, apply the alternation with::

        loader.alex_apply_period(d)

    Now `d` is ready for further processing such as background estimation,
    burst search, etc...
    """
    if BT is not None:
        log.warning('`BT` argument is deprecated, use `leakage` instead.')
        leakage = BT
    if header is not None:
        log.warning('    `header` argument ignored. '
                    '    The header length is now computed automatically.')
    print(" - Loading '%s' ... " % fname)
    ph_times_t, det_t, labels = load_sm(fname, return_labels=True)
    print(" [DONE]\n")

    DONOR_ON = (2850, 580)
    ACCEPT_ON = (930, 2580)
    alex_period = 4000

    dx = Data(fname=fname, clk_p=12.5e-9, nch=1, leakage=leakage, gamma=gamma,
              ALEX=True, lifetime=False, alternated=True,
              meas_type='smFRET-usALEX', polarization=False,
              D_ON=DONOR_ON, A_ON=ACCEPT_ON, alex_period=alex_period,
              ph_times_t=[ph_times_t], det_t=[det_t],
              det_donor_accept=(np.atleast_1d(0), np.atleast_1d(1)),
              ch_labels=labels)
    return dx

def _union_of_list(det_map):
    """ 
    Makes unions of lists of numpy arrays
    """
    if det_map is not None:
        det_union = np.empty(0,dtype=np.uint8)
        for dets in det_map:
            det_union = np.union1d(det_union,dets)
        return det_union
    else:
        return np.empty(0, dtype=np.uint8)

def _reallocate_det_maps(det_map,unique_det):
    """
    Reasign the values in a detector map
    """
    if det_map is not None:
        new_map = []
        for det in det_map:
            new_det = np.zeros(det.shape,dtype=np.uint8)
            for i in range(unique_det.size):
                new_det[unique_det[i] == det] = i
            new_map.append(new_det)
        return new_map
    else:
        return None

def _append_alt_maps(det_map,i,num_dets):
    """
    append newly assigned alternation period indeces to the map arrays in lists
    """
    if det_map is not None:
        new_map_alt = []
        for det in det_map:
            new_map = np.zeros(0,dtype=np.uint8)
            for j in range(i):
                new_map = np.concatenate((new_map,det + j * num_dets))
            new_map_alt.append(new_map)
        return new_map_alt
    else:
        return None

def _apply_period_1ch(d, ph_times_t, det_t, valid_mask, ex_period,ich=0):
    """
    does the majority of work reading detector specs and generating alternation
    period adjusted indeces
    """
    spec_map = d.det_spectral[ich] if hasattr(d,'det_spectral') else None
    pol_map = d.det_p_s_pol[ich] if hasattr(d,'det_p_s_pol') else None
    split_map = d.det_split[ich] if hasattr(d,'det_split') else None
    # Identify all detector indexes in each of the maps
    spec_union = _union_of_list(spec_map)
    pol_union = _union_of_list(pol_map)
    split_union = _union_of_list(split_map)
    det_union = np.union1d(spec_union,pol_union)
    det_union = np.union1d(det_union,split_union)
    if spec_union.size != 0 and spec_union.size != det_union.size:
        log.warning("Spectral identity of some detectors is missing")
    if pol_union.size != 0 and pol_union.size != det_union.size:
        log.warning("Polarization identity of some detectors is missing")
    if split_union.size != 0 and split_union.size != det_union.size:
        log.warning("Split identity of some detectors is missing")
    # Finally, np.unique re-allocates the detector indexes to be arranged from 0 to n with no gaps
    unique_det, det_m = np.unique(det_t, return_inverse=True)
    # check that spectral, poliarization and split specs match the actual indexes in the detectors array
    if det_union.size < unique_det.size:
        raise ValueError("Undefined detectors in HDF5 file")
    elif np.any(det_union != unique_det):
        raise ValueError("Undefined detectors in HDF5 file")
    # generate new mapings for reduced detector numbers
    spec_map = _reallocate_det_maps(spec_map, unique_det)
    pol_map = _reallocate_det_maps(pol_map, unique_det)
    split_map = _reallocate_det_maps(split_map, unique_det)
    # identify all channels in the new mapping (note, this could be just np.arange(0,det_union.size,dtype=np.uint8))
    all_streams = _union_of_list(spec_map) if spec_map is not None else np.empty(0,dtype=np.uint8)
    all_streams = np.union1d(all_streams,_union_of_list(pol_map)) if pol_map is not None else all_streams
    all_streams = np.union1d(all_streams,_union_of_list(split_map)) if split_map is not None else all_streams
    num_dets = all_streams.size
    # remove photons not during an excitation period
    det_m = det_m[valid_mask]
    ph_times_m = ph_times_t[valid_mask]
    # now apply excitation period re-assignment based on ex_period
    alt_periods = []
    spec_map_alt = [] if spec_map is not None else None
    pol_map_alt = [] if pol_map is not None else None
    split_map_alt = [] if split_map is not None else None
    det_m += ex_period * num_dets
    # update the maps for new detector channels
    for i in range(len(d.alt_ON[ich])):
        alt_periods.append(all_streams + (i * num_dets))
    spec_map_alt = _append_alt_maps(spec_map,len(d.alt_ON[ich]),num_dets)
    pol_map_alt = _append_alt_maps(pol_map,len(d.alt_ON[ich]),num_dets)
    split_map_alt = _append_alt_maps(split_map,len(d.alt_ON[ich]),num_dets)
    stream_map = {}
    if spec_map is not None: stream_map['em'] = spec_map_alt
    if pol_map is not None: stream_map['pol'] = pol_map_alt
    if split_map is not None: stream_map['split'] = split_map_alt
    stream_map['ex'] = alt_periods
    stream_map['all_streams'] = _union_of_list(alt_periods)
    _append_data_ch(d, '_stream_map', (stream_map))
    _append_data_ch(d, 'det_m', det_m)
    _append_data_ch(d, 'ph_times_m', ph_times_m)


def _usalex_apply_period_1ch(d, delete_ph_t=True, ich=0):
    """Applies to the Data object `d` the alternation period previously set.

    This function operates on a single-channel.
    See :func:`usalex_apply_period` for details.
    """
    # generate mask for photon based on times
    ph_times_t = d.ph_times_t[ich]
    det_t = d.det_t[ich]
    if 'offset' in d:
        ph_times_t -= d.offset[ich]
    valid = np.zeros((len(d.alt_ON[ich]),ph_times_t.shape[0]),dtype=np.uint8)
    for i, ON in enumerate(d.alt_ON[ich]):
        valid[i,:] = _select_range(ph_times_t,d.alex_period[ich],ON)
    # check if any streams assigned to multiple excitation periods
    valid_sum = valid.sum(axis=0)
    if np.any(valid_sum>1):
        raise ValueError('Overlapping excitation channels')
    valid_mask = valid_sum == 1
    # valid_red is only photons inside an excitation period
    valid_red = valid[:,valid_mask]
    ex_period = np.empty(valid_red.shape[1],dtype=np.uint8)
    # make an array of which excitation period each "valid" photon belongs to
    for i in range(valid_red.shape[0]):
        ex_period[valid_red[i,:]==1] = i
    _apply_period_1ch(d, ph_times_t, det_t, valid_mask, ex_period, ich=ich)
    
    if delete_ph_t:
        d.delete('ph_times_t')
        d.delete('det_t')
    
    ### The following is the old code ###
    # donor_ch, accept_ch = d.det_spectral[ich] ## donor_ch, accept_ch with single det_ch
    # D_ON, A_ON = d._D_ON_multich[ich], d._A_ON_multich[ich] # same, replace with alt_ON = d.alt_ON
    # # Remove eventual ch different from donor or acceptor
    # det_t = d.det_t[ich][:]
    # ph_times_t = d.ph_times_t[ich][:] ## dito
    # d_ch_mask_t = selection_mask(det_t, donor_ch) ## make for loop
    # a_ch_mask_t = selection_mask(det_t, accept_ch)
    # valid_det = d_ch_mask_t + a_ch_mask_t # again for loop
    
    # # Build masks for excitation windows
    # d_ex_mask_t = _select_range(ph_times_t, d.alex_period, D_ON) ## make for loop
    # a_ex_mask_t = _select_range(ph_times_t, d.alex_period, A_ON)
    # # Safety check: each ph is either D or A ex (not both)
    # assert not (d_ex_mask_t * a_ex_mask_t).any()

    # # Select alternation periods, removing transients and invalid detectors
    # DexAex_mask = (d_ex_mask_t + a_ex_mask_t) * valid_det

    # # Reduce photons to the DexAex_mask selection
    # ph_times = ph_times_t[DexAex_mask]
    # d_em = d_ch_mask_t[DexAex_mask] # thes can be replaced with det_m
    # a_em = a_ch_mask_t[DexAex_mask]
    # d_ex = d_ex_mask_t[DexAex_mask]
    # a_ex = a_ex_mask_t[DexAex_mask]
    # assert d_ex.sum() == d_ex_mask_t.sum()
    # assert a_ex.sum() == a_ex_mask_t.sum()

    # if remove_d_em_a_ex:
    #     # Removes donor-ch photons during acceptor excitation
    #     mask = a_em + d_em * d_ex
    #     assert (mask == -(a_ex * d_em)).all()
    #     ph_times = ph_times[mask]
    #     d_em = d_em[mask]
    #     a_em = a_em[mask]
    #     d_ex = d_ex[mask]
    #     a_ex = a_ex[mask]

    # assert d_em.sum() + a_em.sum() == ph_times.size
    # assert (d_em + a_em).all()       # masks fill the total array
    # assert not (d_em * a_em).any()   # no photon is both D and A
    # assert a_ex.size == a_em.size == d_ex.size == d_em.size == ph_times.size
    # _append_data_ch(d, 'ph_times_m', ph_times)
    # _append_data_ch(d, 'D_em', d_em)
    # _append_data_ch(d, 'A_em', a_em)
    # _append_data_ch(d, 'D_ex', d_ex)
    # _append_data_ch(d, 'A_ex', a_ex)
    # assert (len(d.ph_times_m) == len(d.D_em) == len(d.A_em) ==
    #         len(d.D_ex) == len(d.A_ex) == ich + 1)

    # if 'particles_t' in d:
    #     particles_t = d.particles_t[ich][:]
    #     particles = particles_t[DexAex_mask]
    #     _append_data_ch(d, 'particles', particles)

    # assert d.ph_times_m[ich].size == d.A_em[ich].size

    # if d.polarization:
    #     # We also have polarization data
    #     p_pol_ch, s_pol_ch = d._det_p_s_pol_multich[ich]
    #     p_em, s_em = _get_det_masks(det_t, p_pol_ch, s_pol_ch, DexAex_mask,
    #                                 mask_ref=valid_det, ich=ich)
    #     _append_data_ch(d, 'P_em', p_em)
    #     _append_data_ch(d, 'S_em', s_em)

    # if delete_ph_t:
    #     d.delete('ph_times_t')
    #     d.delete('det_t')
    # return d

def usalex_apply_period(d, delete_ph_t=True):
    """Applies to the Data object `d` the alternation period previously set.

    Note that you first need to load the data in a variable `d` and then
    set the alternation parameters using `d.add(D_ON=..., A_ON=...)`.

    The typical pattern for loading ALEX data is the following::

        d = loader.photon_hdf5(fname=fname)
        d.add(D_ON=(2850, 580), A_ON=(900, 2580))
        alex_plot_alternation(d)

    If the plot looks good, apply the alternation with::

        loader.alex_apply_period(d)

    Now `d` is ready for further processing such as background estimation,
    burst search, etc...

    *See also:* :func:`alex_apply_period`.
    """
    for ich in range(d.nch):
        _usalex_apply_period_1ch(d, ich=ich,
                                 delete_ph_t=False)
    if delete_ph_t:
        d.delete('ph_times_t', 'det_t')
    # d.set_immutable('ph_times_m', 'det_m')
    d.add(alternation_applied=True)
    return d

##
# nsALEX loader functions
#

def nsalex(fname):
    """Load nsALEX data from a SPC file and return a Data() object.

    This function returns a Data() object to which you need to apply
    an alternation selection before performing further analysis (background
    estimation, burst search, etc.).

    The pattern to load nsALEX data is the following::

        d = loader.nsalex(fname=fname)
        d.add(D_ON=(2850, 580), A_ON=(900, 2580))
        alex_plot_alternation(d)

    If the plot looks good apply the alternation with::

        loader.alex_apply_period(d)

    Now `d` is ready for further processing such as background estimation,
    burst search, etc...
    """
    ph_times_t, det_t, nanotimes = load_spc(fname)

    DONOR_ON = (10, 1500)
    ACCEPT_ON = (2000, 3500)
    nanotimes_nbins = 4095

    dx = Data(fname=fname, clk_p=50e-9, nch=1, ALEX=True, lifetime=True,
              D_ON=DONOR_ON, A_ON=ACCEPT_ON,
              nanotimes_nbins=nanotimes_nbins,
              nanotimes_params=[{'tcspc_num_bins': nanotimes_nbins}],
              ph_times_t=[ph_times_t], det_t=[det_t], nanotimes_t=[nanotimes],
              det_donor_accept=(np.atleast_1d(4), np.atleast_1d(6)))
    return dx

def _nsalex_apply_period_1ch(d, ich=0, delete_ph_t=True):
    # extract main specs from each channel
    ph_times_t = d.ph_times_t[ich]
    det_t = d.det_t[ich]
    nanotimes_t = d.nanotimes_t[ich]
    # identify photons in nanotime excitation ranges
    valid = np.zeros((len(d.alt_ON[ich]),nanotimes_t.shape[0]),dtype=bool)
    for i, ON in enumerate(d.alt_ON[ich]):
        valid[i,:] = (nanotimes_t > ON[0]) * (nanotimes_t < ON[1])
    valid_sum = valid.sum(axis=0)
    if np.any(valid_sum > 1):
        raise ValueError("Overlapping excitation periods")
    valid_mask = valid_sum == 1
    valid_red = valid[:,valid_mask]
    # make array of excitation periods
    ex_period = np.empty(valid_red.shape[1],dtype=np.uint8)
    for i in range(valid_red.shape[0]):
        ex_period[valid_red[i,:]] = i
    _apply_period_1ch(d, ph_times_t, det_t, valid_mask, ex_period)
    nanotimes = nanotimes_t[valid_mask]
    _append_data_ch(d, 'nanotimes', nanotimes)
    if delete_ph_t:
        d.delete('ph_times_t')
        d.delete('det_t')
        d.delete('nanotimes_t')
    
    
    # donor_ch, accept_ch = d._det_donor_accept_multich[ich]
    # D_ON_multi, A_ON_multi = d._D_ON_multich[ich], d._A_ON_multich[ich]
    # D_ON = [(D_ON_multi[i], D_ON_multi[i + 1])
    #         for i in range(0, len(D_ON_multi), 2)]
    # A_ON = [(A_ON_multi[i], A_ON_multi[i + 1])
    #         for i in range(0, len(A_ON_multi), 2)]
    # Mask for donor + acceptor detectors (discard other detectors)
    # det_t = d.det_t[ich][:]
    # d_ch_mask_t = selection_mask(det_t, donor_ch)
    # a_ch_mask_t = selection_mask(det_t, accept_ch)
    # da_ch_mask_t = d_ch_mask_t + a_ch_mask_t

    # # Masks for excitation periods
    # d_ex_mask_t = np.zeros(d.nanotimes_t[ich].size, dtype='bool')
    # for d_on in D_ON:
    #     d_ex_mask_t += (d.nanotimes_t[ich] > d_on[0]) * (d.nanotimes_t[ich] < d_on[1])

    # a_ex_mask_t = np.zeros(d.nanotimes_t[ich].size, dtype='bool')
    # for a_on in A_ON:
    #     a_ex_mask_t += (d.nanotimes_t[ich] > a_on[0]) * (d.nanotimes_t[ich] < a_on[1])

    # ex_mask_t = d_ex_mask_t + a_ex_mask_t  # Select only ph during Dex or Aex

    # # Total mask: D+A photons, and only during the excitation periods
    # valid = da_ch_mask_t * ex_mask_t  # logical AND

    # # Apply selection to timestamps and nanotimes
    # ph_times = d.ph_times_t[ich][:][valid]
    # nanotimes = d.nanotimes_t[ich][:][valid]

    # # Apply selection to the emission masks
    # d_em = d_ch_mask_t[valid]
    # a_em = a_ch_mask_t[valid]
    # assert (d_em + a_em).all()       # masks fill the total array
    # assert not (d_em * a_em).any()   # no photon is both D and A

    # # Apply selection to the excitation masks
    # d_ex = d_ex_mask_t[valid]
    # a_ex = a_ex_mask_t[valid]
    # assert (d_ex + a_ex).all()
    # assert not (d_ex * a_ex).any()

    # d.add(ph_times_m=[ph_times], nanotimes=[nanotimes],
    #       D_em=[d_em], A_em=[a_em], D_ex=[d_ex], A_ex=[a_ex],
    #       alternation_applied=True)

    # if d.polarization:
    #     # We also have polarization data
    #     p_polariz_ch, s_polariz_ch = d._det_p_s_pol_multich[ich]
    #     p_em, s_em = _get_det_masks(det_t, p_polariz_ch, s_polariz_ch, valid,
    #                                 ich=ich)
    #     d.add(P_em=[p_em], S_em=[s_em])
    


def nsalex_apply_period(d, delete_ph_t=True, ich=0):
    """Applies to the Data object `d` the alternation period previously set.

    Note that you first need to load the data in a variable `d` and then
    set the alternation parameters using `d.add(D_ON=..., A_ON=...)`.

    The typical pattern for loading ALEX data is the following::

        d = loader.photon_hdf5(fname=fname)
        d.add(alt_ON=[(2850, 580),(900, 2580)])
        alex_plot_alternation(d)

    If the plot looks good, apply the alternation with::

        loader.alex_apply_period(d)

    Now `d` is ready for further processing such as background estimation,
    burst search, etc...

    *See also:* :func:`alex_apply_period`.
    """
    for ich in range(d.nch):
        _nsalex_apply_period_1ch(d, ich=ich,
                                 delete_ph_t=False)
    if delete_ph_t:
        d.delete('ph_times_t', 'det_t', 'nanotimes_t')
    # d.set_immutable('ph_times_m', 'det_m', 'nanotimes')
    d.add(alternation_applied=True)
    




def _get_det_masks(det_t, det_ch1, det_ch2, valid, mask_ref=None, ich=0):
    """
    Returns two masks for photons detected by `det_ch1` and `det_ch2`,
    and being also `valid`. `valid` is a bool mask, same size as `det_t`,
    which selects the photons considered "valid".
    The returned masks have same size as `det_t` (and as `valid`).
    """
    ch1_mask_t = selection_mask(det_t, det_ch1)
    ch2_mask_t = selection_mask(det_t, det_ch2)
    if mask_ref is not None:
        both_ch_mask_t = ch1_mask_t + ch2_mask_t
        assert all(both_ch_mask_t == mask_ref)
    # Apply "valid" selection to the channel masks
    ch1_mask = ch1_mask_t[valid]
    ch2_mask = ch2_mask_t[valid]
    assert (ch1_mask + ch2_mask).all()       # masks fill the total array
    assert not (ch1_mask * ch2_mask).any()   # no photon is both channels
    return ch1_mask, ch2_mask


def alex_apply_period(d, delete_ph_t=True):
    """Apply the ALEX period definition set in D_ON and A_ON attributes.

    This function works both for us-ALEX and ns-ALEX data.

    Note that you first need to load the data in a variable `d` and then
    set the alternation parameters using `d.add(D_ON=..., A_ON=...)`.

    The typical pattern for loading ALEX data is the following::

        d = loader.photon_hdf5(fname=fname)
        d.add(D_ON=(2850, 580), A_ON=(900, 2580))
        alex_plot_alternation(d)

    If the plot looks good, apply the alternation with::

        loader.alex_apply_period(d)

    Now `d` is ready for further processing such as background estimation,
    burst search, etc...
    """
    if not d.alternated:
        print('No alternation found. Nothing to apply.')
        return
    if 'alternation_applied' in d and d['alternation_applied']:
        print('Alternation already applied, Cannot be reapplied. \n'
              'Reload the data if you need to change alternation parameters.')
        return

    if d.lifetime:
        apply_period_func = nsalex_apply_period
    else:
        apply_period_func = usalex_apply_period
    apply_period_func(d, delete_ph_t=delete_ph_t)
    ph_data_size = d.ph_data_sizes.sum()
    msg = ('# Total photons (after ALEX selection):  {:12,}\n'.format(ph_data_size))
    phot_count = {ph_str:0 for ph_str in d.ph_streams}
    d._time_min = d._time_reduce(last=False, func=min)
    d._time_max = d._time_reduce(last=True, func=max)
    for ich, str_dict in enumerate(d.ph_streams_inv_dict):
        for sel_str, index in str_dict.items():
            phot_count[sel_str] += (d.det_m[ich] == index).sum()
    for sel_str, count in phot_count.items():
        msg += f'# of photons in {sel_str} stream: ' +'{:12,}\n'.format(count)
    print(msg)
    
def sm_single_laser(fname):
    """Load SM files acquired using single-laser and 2 detectors.
    """
    print(" - Loading '%s' ... " % fname)
    ph_times_t, det_t, labels = load_sm(fname, return_labels=True)
    print(" [DONE]\n")

    a_em = (det_t == 1)
    dx = Data(fname=fname, clk_p=12.5e-9, nch=1,
              ALEX=False, lifetime=False, alternated=False,
              meas_type='smFRET',
              ph_times_m=[ph_times_t], det_donor_accept=(0, 1), A_em=[a_em],
              ch_labels=labels)
    dx.add(acquisition_duration=np.round(dx.time_max - dx.time_min, 1))
    return dx
