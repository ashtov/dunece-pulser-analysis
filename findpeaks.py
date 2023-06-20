import logging
import sys
import pathlib

import numpy as np
import scipy
import pandas as pd

from hdf5libs import HDF5RawDataFile
import detchannelmaps
import daqdataformats
import fddetdataformats
import rawdatautils.unpack.wib2

## debug for now
#fname = 'np02_bde_coldbox_run020331_0000_dataflow0_datawriter_0_20230309T180027.hdf5'
#infnames = [fname]
#dacgains = [63]
#print(infnames)

# Channel map maps from internal DAQ numbering by WIB to official channel
# number. Should be the same always.
CHANNEL_MAP_NAME = 'VDColdboxChannelMap'
CHANNEL_MAP = detchannelmaps.make_map(CHANNEL_MAP_NAME)

# Are these really always the same???
CHANNELS_PER_CRP = 3072
TIMESTAMPS_PER_FRAME = 8192
MAX_PEAKS_PER_EVENT = 4

BAD_CHANNELS = [1958, 2250, 2868]

def peak_heights(arr, cutoff: int):
    """Find heights of peaks in 1d numpy array arr with heights greater than
    the height of the highest - cutoff."""
    #peaks, peakinfo = scipy.signal.find_peaks(arr, height=np.max(arr) - cutoff, prominence=(None, None), wlen=20)
    peaks, peakinfo = scipy.signal.find_peaks(arr, distance=2000, prominence=(cutoff, None), wlen=20, width=3.5)
    logging.debug('* * * * Peaks: %s\n* * * * Peak info: %s', str(peaks), str(peakinfo))
    return peakinfo['prominences'], peaks, peakinfo

def apply_mask(arr):
    """Applies mask to 2d array of peak heights
    Blocks out zeros and known bad channels"""
    arr_masked = np.ma.masked_equal(arr, 0, copy=False)
    for i in BAD_CHANNELS:
        arr_masked[i] = np.ma.masked
    return arr_masked

def read_parameters(parametersfname):
    """Read parameters file into pandas dataframe"""
    parameters = pd.read_csv(parametersfname, sep='\t', header=0, index_col=0, dtype={'Run number': 'Int32', 'Filename': 'string', 'Pulser DAC': 'Int8', 'Peaksfile': 'string'})
    return parameters

def read_peaksfile(fname):
    """Read file with peaks data"""
    peaks = np.loadtxt(fname, dtype=np.int16, delimiter='\t')
    assert peaks.shape[0] == CHANNELS_PER_CRP and peaks.shape[1] % MAX_PEAKS_PER_EVENT == 0
    peaks_masked = apply_mask(peaks)
    return peaks_masked

def write_peaksfile(fname, arr):
    """Write array of peaks to file"""
    assert arr.shape[0] == CHANNELS_PER_CRP and arr.shape[1] % MAX_PEAKS_PER_EVENT == 0
    np.savetxt(fname, arr, fmt='%d', delimiter='\t')

def main():
    if len(sys.argv) >= 3:
        loglevel = sys.argv[2]
    else:
        loglevel = 'INFO'
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)
    #logging.basicConfig(filename='log.log', level=logging.INFO)
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} FILELIST')
        return 0
    parametersfname = pathlib.Path(sys.argv[1])
    if not parametersfname.exists():
        print(f'No such file: {parametersfname}')
        return 0

    logging.info('Reading parameters file %s . . .', parametersfname)
    parameters = read_parameters(parametersfname)
    logging.info('Successfully read parameters file')

    for run_number in parameters.index:
        fname = parameters.at[run_number, 'Filename']
        logging.info('Reading file %s . . .', fname)
        h5_file = HDF5RawDataFile(fname)
        # One "record" is one triggered event, stored as a tuple (n, 0)
        # (not sure what the 0 is for)
        #records: list[tuple[int, int]]  = h5_file.get_all_record_ids()
        records = h5_file.get_all_record_ids()
        # assuming no more than 4 peaks per frame ???
        allpeakheights = np.zeros((CHANNELS_PER_CRP, len(records), MAX_PEAKS_PER_EVENT), dtype=np.int16)
        first = True
        for rec in records:
            logging.info('* Record: %s', str(rec))
            assert rec[1] == 0
            # One "fragment" is either data and metadata from a single WIB or some
            # other metadata about the event.
            fragpaths: list[str] = h5_file.get_fragment_dataset_paths(rec)
            for fragpath in fragpaths:
                logging.debug('* * Fragment: %s', fragpath)
                frag = h5_file.get_frag(fragpath)
                fragheader = frag.get_header()
                assert fragheader.version == 5
                # Only attempt to process WIB data, not metadata fragments
                logging.debug('* * * FragmentType: %s', fragheader.fragment_type)
                if fragheader.fragment_type == daqdataformats.FragmentType.kWIB.value:
                    # WIB2Frame object stores data and another level of metadata
                    # for a single WIB
                    frame = fddetdataformats.WIB2Frame(frag.get_data())
                    frameheader = frame.get_header()
                    assert frameheader.version == 4
                    data = rawdatautils.unpack.wib2.np_array_adc(frag).T
                    assert data.shape == (256, TIMESTAMPS_PER_FRAME)
                    if first:
                        cutoff = round(np.std(data[0]) * 6)
                        logging.info('Using cutoff: %d', cutoff)
                        first = False
                    firstchan = True    # this solves problem with first channel of 256 not having reasonable peaks
                    for i in range(256):
                        chnum = CHANNEL_MAP.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, i)
                        logging.debug('* * * * Channel number: %d', chnum)
                        heights, peak_locations, fullpeakinfo = peak_heights(data[i], cutoff)
                        l = len(heights)
                        if firstchan:
                            expected_peaks = peak_locations
                            expected_peaks_len = len(expected_peaks)
                            if expected_peaks_len not in (3, 4):
                                logging.warning('Number of peaks in first channel in fragment not 3 or 4? Probably wrong! Got peak locations %s, heights %s', str(peak_locations), str(heights))
                                continue    # good idea???
                            firstchan = False
                        #elif not np.array_equal(expected_peaks, peak_locations):
                        elif l != expected_peaks_len or np.any(np.abs(expected_peaks - peak_locations) > 1):
                            if chnum in BAD_CHANNELS:
                                logging.debug('Known bad channel: Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights))
                            else:
                                logging.warning('Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s, full peak info: %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights), str(fullpeakinfo))
                            # easy, just don't write bad ones at this point
                            continue
                            #l2 = min(l, MAX_PEAKS_PER_EVENT)
                            #allpeakheights[chnum, rec[0] - 1, :l2] = heights[:l2]
                        # set to 0 (effectively delete) "peaks" too close to
                        # signal edge to be accurately measured
                        if peak_locations[0] < 7:
                            heights[0] = 0
                            logging.debug('First peak too close to edge! Ignoring it . . .')
                        elif peak_locations[-1] > TIMESTAMPS_PER_FRAME - 9:
                            heights[-1] = 0
                            logging.debug('Last peak too close to edge! Ignoring it . . .')
                        # oops I had this in an 'else' and that always skipped writing channel 0 data
                        allpeakheights[chnum, rec[0] - 1, :l] = heights
                            #if l > MAX_PEAKS_PER_EVENT:
                            #    logging.warning('More than %d peaks found in channel %d (fragment %s, record %s)', MAX_PEAKS_PER_EVENT, chnum, fragpath, str(rec))
                            #    l = MAX_PEAKS_PER_EVENT
                            #    allpeakheights[chnum, rec[0] - 1] = heights[:l]
                            #else:
                            #    allpeakheights[chnum, rec[0] - 1, :l] = heights

                    ## probably don't need to regenerate this each time
                    #chnums = CHANNEL_MAP.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, np.arange(data.shape[1]))
        allpeakheights_flatter = allpeakheights.reshape((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT))
        allpeakheights_masked = apply_mask(allpeakheights_flatter)
        write_peaksfile(parameters.at[run_number, 'Peaksfile'], allpeakheights_masked)
        first = False

if __name__ == '__main__':
    main()

