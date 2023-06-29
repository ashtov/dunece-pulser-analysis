import logging
import sys
import pathlib

import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt    # DEBUG

from hdf5libs import HDF5RawDataFile
import detchannelmaps
import daqdataformats
import fddetdataformats
import rawdatautils.unpack.wib2

import filtfunc

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
TIME_PER_SAMPLE = 0.512 # microseconds

#BAD_CHANNELS = [1958, 2250, 2868]
#BAD_CHANNELS = [182, 1424, 1849, 2949, 2997, 3019]

def peak_heights(arr, cutoff: int):
    """Find heights of peaks in 1d numpy array arr with heights greater than
    the height of the highest - cutoff."""
    #peaks, peakinfo = scipy.signal.find_peaks(arr, height=np.max(arr) - cutoff, prominence=(None, None), wlen=20)
    peaks, peakinfo = scipy.signal.find_peaks(arr, distance=2000, prominence=(cutoff, None), wlen=20, width=3.5)
    logging.debug('* * * * Peaks: %s\n* * * * Peak info: %s', str(peaks), str(peakinfo))
    return peakinfo['prominences'], peaks, peakinfo

def fit_peak(arr, height):
    """Attempts to fit filtfunc.f() to the peak beginning at index peakpos in
    arr and returns fit parameters."""
    # or the peak contained entirely in arr, with nothing else?
    # hardcoded one step = 0.5 ms?
    #popt, pcov = scipy.optimize.curve_fit(filtfunc.f, np.arange(len(arr)) / 2, arr, p0=[height, 2., 0., arr[0]], bounds=((0., 1.5, -0., 0.), (163840, 2.5, +2., 16384)))
    popt, pcov = scipy.optimize.curve_fit(filtfunc.f, np.arange(len(arr)) * TIME_PER_SAMPLE, arr, p0=[height, 2., 0., arr[0]], bounds=((0., 1.5, -1., arr[0] - 25), (163840, 2.5, +4., arr[0] + 25)))
    return popt, pcov

def apply_mask(arr, BAD_CHANNELS):
    """Applies mask to 2d array of peak heights
    Blocks out zeros and known bad channels"""
    arr_masked = np.ma.masked_equal(arr, 0, copy=False)
    for i in BAD_CHANNELS:
        arr_masked[i] = np.ma.masked
    return arr_masked

def read_parameters(parametersfname):
    """Read parameters file into pandas dataframe"""
    logging.info('Reading parameters file %s . . .', parametersfname)
    parameters = pd.read_csv(parametersfname, sep='\t', header=0, index_col=0, dtype={'Run number': 'Int32', 'Filename': 'string', 'Pulser DAC': 'Int8', 'Peaksfile': 'string'})
    logging.info('Successfully read parameters file')
    return parameters

def read_config(configfname):
    """Read config file"""
    logging.info('Reading config file %s . . .', configfname)
    configdf = pd.read_csv(configfname, sep='\t', header=0, index_col=0, dtype={'Parameter': 'string', 'Value': 'string'}, usecols=[0, 1])
    config = configdf.to_dict()['Value']
    config['bad_channels'] = [int(i) for i in config['bad_channels'].split(',')]
    logging.info('Successfully read config file')
    logging.debug('Config: %s', str(config))
    return config

def read_peaksfile(fname, config):
    """Read file with peaks data"""
    peaks = np.loadtxt(fname, dtype=np.int16, delimiter='\t')
    assert peaks.shape[0] == CHANNELS_PER_CRP and peaks.shape[1] % MAX_PEAKS_PER_EVENT == 0
    peaks_masked = apply_mask(peaks, config['bad_channels'])
    return peaks_masked

def write_peaksfile(fname, arr):
    """Write array of peaks to file"""
    assert arr.shape[0] == CHANNELS_PER_CRP and arr.shape[1] % MAX_PEAKS_PER_EVENT == 0
    np.savetxt(fname, arr, fmt='%d', delimiter='\t')

def set_loglevel():
    if len(sys.argv) >= 3:
        loglevel = sys.argv[2]
    else:
        loglevel = 'INFO'
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)
    #logging.basicConfig(filename='log.log', level=logging.INFO)

def frames_in_file_iter(fname: str):
    """Iterable for looping over frames in a file"""
    raise NotImplementedError()

def main():
    set_loglevel()
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} CONFIGFILE')
        return 0
    configfname = pathlib.Path(sys.argv[1])
    if not configfname.exists():
        print(f'No such file: {configfname}')
        return 0
    config = read_config(configfname)
    parametersfname = pathlib.Path(config['parametersfname'])
    if not parametersfname.exists():
        print(f'No such file: {parametersfname}')
        return 0
    parameters = read_parameters(parametersfname)
    BAD_CHANNELS = config['bad_channels']

    for run_number in parameters.index:
        pulserDAC = parameters.at[run_number, 'Pulser DAC']
        fname = parameters.at[run_number, 'Filename']
        logging.info('Reading file %s . . .', fname)
        h5_file = HDF5RawDataFile(fname)
        # One "record" is one triggered event, stored as a tuple (n, 0)
        # (not sure what the 0 is for)
        #records: list[tuple[int, int]]  = h5_file.get_all_record_ids()
        records = h5_file.get_all_record_ids()
        # assuming no more than 4 peaks per frame ???
        allpeakheights = np.zeros((CHANNELS_PER_CRP, len(records), MAX_PEAKS_PER_EVENT), dtype=np.int16)
        #alldata = []    # new pedestal-finding terrible
        first = True
        #for rec in records:
        for rec in records[:2]: # DEBUG
            logging.info('* Record: %s', str(rec))
            assert rec[1] == 0
            # One "fragment" is either data and metadata from a single WIB or some
            # other metadata about the event.
            fragpaths: list[str] = h5_file.get_fragment_dataset_paths(rec)
            for fragnum, fragpath in enumerate(fragpaths):
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
                    # new pedestal-finding
                    #alldata.append(data)
                    if data.shape != (256, TIMESTAMPS_PER_FRAME):
                        logging.warning('Bad data shape? shape: %s for fragment %s, record %s', str(data.shape), fragpath, str(rec))
                    #logging.debug('* * * data shape: %s', str(data.shape))
                    #assert data.shape == (256, TIMESTAMPS_PER_FRAME)
                    if first:
                        cutoff = round(np.std(data[0]) * 6)
                        logging.info('Using cutoff: %d', cutoff)
                        first = False
                    firstchan = True    # this solves problem with first channel of 256 not having reasonable peaks
                    #for i in range(256):
                    for i in range(1):
                        chnum = CHANNEL_MAP.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, i)
                        logging.debug('* * * * Channel number: %d', chnum)
                        heights, peak_locations, fullpeakinfo = peak_heights(data[i], cutoff)
                        # set to 0 (effectively delete) "peaks" too close to
                        # signal edge to be accurately measured
                        if peak_locations[0] < 7:
                            #heights[0] = 0
                            heights = heights[1:]
                            peak_locations = peak_locations[1:]
                            logging.debug('First peak too close to edge! Ignoring it . . .')
                        elif peak_locations[-1] > TIMESTAMPS_PER_FRAME - 9:
                            #heights[-1] = 0
                            heights = heights[:-1]
                            peak_locations = peak_locations[:-1]
                            logging.debug('Last peak too close to edge! Ignoring it . . .')
                        l = len(heights)
                        if firstchan:
                            expected_peaks = peak_locations
                            expected_peaks_len = len(expected_peaks)
                            if expected_peaks_len not in (3, 4):
                                logging.warning('Number of peaks in first channel in fragment not 3 or 4? Probably wrong! Got peak locations %s, heights %s', str(peak_locations), str(heights))
                                continue    # good idea???
                            firstchan = False
                        elif l != expected_peaks_len or np.any(np.abs(expected_peaks - peak_locations) > 1):
                            if chnum in BAD_CHANNELS:
                                logging.debug('Known bad channel: Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights))
                            else:
                                logging.warning('Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s, full peak info: %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights), str(fullpeakinfo))
                            # easy, just don't write bad ones at this point
                            continue
                        # oops I had this in an 'else' and that always skipped writing channel 0 data
                        allpeakheights[chnum, rec[0] - 1, :l] = heights
                        
                        # pulse fitting
                        for loc, height in zip(peak_locations, heights):
                            WINDOW_BEFORE = -20
                            WINDOW_BEFORE_FIT = -WINDOW_BEFORE - 5
                            WINDOW_AFTER = 30
                            WINDOW_AFTER_FIT = -WINDOW_BEFORE + 7
                            # off by one or not? think about it
                            if loc < -WINDOW_BEFORE or loc > TIMESTAMPS_PER_FRAME - WINDOW_AFTER:
                                continue
                            logging.debug("loc: %s, height: %s", str(loc), str(height))
                            #window = data[i][loc - 6:loc + 7]# - data[i][loc - 6]
                            # size of window to draw on plot and to use for fit
                            window = data[i][loc + WINDOW_BEFORE:loc + WINDOW_AFTER].astype(np.float64)# - data[i][loc - 6]
                            fitwindow = window[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT]
                            popt, pcov = fit_peak(fitwindow, height)   # only fit rising edge?
                            real_amplitude = popt[0] / 10.11973588
                            logging.debug("popt: %s\npcov: %s", str(popt), str(pcov))
                            # DEBUG
                            x = (np.arange(len(window)) - WINDOW_BEFORE_FIT) * TIME_PER_SAMPLE
                            x2 = np.linspace(x[0], x[-1], 512)
                            y = filtfunc.f(x2, popt[0], popt[1], popt[2], popt[3])
                            logging.debug("Min vs. max fitted: %.2f", np.max(y) - np.min(y))
                            fig, ax = plt.subplots(figsize=(12, 8))
                            ax.scatter(x, window, s=10, label='Data')
                            ax.scatter(x[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT], fitwindow, s=10, label='Data used for fit')
                            ax.plot(x2, y, linewidth=0.5, label='Fit')
                            pmax = np.max(window)
                            ax.text(7, pmax, f'A0: {popt[0]:8.2f}\nrA: {real_amplitude:8.2f}\ntp: {popt[1]:8.2f}\nh:  {popt[2]:8.2f}\nb:  {popt[3]:8.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
                            ax.legend()
                            ax.set(
                                    title=f'Pulser DAC = {pulserDAC}, Record {rec[0]}, Channel {chnum}, Peak at {loc}',
                                    xlabel='Time (microseconds)',
                                    ylabel='ADC Counts',
                                    xticks=np.arange(round(x[0]), round(x[-1]), 2),
                                    )
                            #fig.savefig(f'debug/DEBUGPLOT_{i}_{loc}.png')
                            fig.savefig(f'fitplots_crp4/{pulserDAC}_{rec[0]}_{chnum}_{loc}.png')
                            plt.close()
                            break   # DEBUG

        # COMMENTED FOR PLOTFITTING TO FIRST CHANNEL ONLY
        #allpeakheights_flatter = allpeakheights.reshape((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT))
        #allpeakheights_masked = apply_mask(allpeakheights_flatter, BAD_CHANNELS)
        #write_peaksfile(parameters.at[run_number, 'Peaksfile'], allpeakheights_masked)
        first = False

if __name__ == '__main__':
    main()

