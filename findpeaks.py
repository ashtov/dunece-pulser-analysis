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

MAGIC_CONVERSION_FACTOR = 10.11973588 
MAX_HEIGHT = 16384 * MAGIC_CONVERSION_FACTOR


#BAD_CHANNELS = [1958, 2250, 2868]
#BAD_CHANNELS = [182, 1424, 1849, 2949, 2997, 3019]

def peak_heights(arr, cutoff: int):
    """Find heights of peaks in 1d numpy array arr with heights greater than
    the height of the highest - cutoff."""
    #peaks, peakinfo = scipy.signal.find_peaks(arr, height=np.max(arr) - cutoff, prominence=(None, None), wlen=20)
    peaks, peakinfo = scipy.signal.find_peaks(arr, distance=2000, prominence=(cutoff, None), wlen=20, width=3.5)
    #logging.debug('* * * * Peaks: %s\n* * * * Peak info: %s', str(peaks), str(peakinfo))
    return peakinfo['prominences'], peaks, peakinfo

def fit_peak(arr, height, base=None):
    """Attempts to fit filtfunc.f() to the peak beginning at index peakpos in
    arr and returns fit parameters."""
    # FIX THIS MESS LATER, filtfunc should probably not have h and b in it
    # or the peak contained entirely in arr, with nothing else?
    # hardcoded one step = 0.5 ms?
    #popt, pcov = scipy.optimize.curve_fit(filtfunc.f, np.arange(len(arr)) / 2, arr, p0=[height, 2., 0., arr[0]], bounds=((0., 1.5, -0., 0.), (163840, 2.5, +2., 16384)))
    if base:
        f = lambda t, A0, tp, h: filtfunc.f(t, A0, tp, h, base)
        popt, pcov = scipy.optimize.curve_fit(f, np.arange(len(arr)) * TIME_PER_SAMPLE, arr, p0=[height * MAGIC_CONVERSION_FACTOR, 2., -0.3], bounds=((0., 1.5, -1.), (MAX_HEIGHT, 2.5, +1.)))
    else:
        popt, pcov = scipy.optimize.curve_fit(filtfunc.f, np.arange(len(arr)) * TIME_PER_SAMPLE, arr, p0=[height * MAGIC_CONVERSION_FACTOR, 2., 0., arr[0]], bounds=((0., 1.5, -1., arr[0] - 25), (MAX_HEIGHT, 2.5, +4., arr[0] + 25)))
    return popt, pcov

def fit_peak2(arr, height):
    """Attempts to fit filtfunc.f() to the peak beginning at index peakpos in
    arr and returns fit parameters."""
    popt, pcov = scipy.optimize.curve_fit(filtfunc.f_no_b, np.arange(len(arr)) * TIME_PER_SAMPLE, arr, p0=[height * MAGIC_CONVERSION_FACTOR, 2., -0.3], bounds=((0., 1.5, -1.), (MAX_HEIGHT, 2.5, +1.)))
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
    if config['pulserDACs'] == 'all':
        config['pulserDACs'] = list(range(64))
    else:
        config['pulserDACs'] = [int(i) for i in config['pulserDACs'].split(',')]
    logging.info('Successfully read config file')
    #logging.debug('Config: %s', str(config))
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

    if len(sys.argv) >= 4:
        alldatafname = sys.argv[4]
        logging.info('alldatafname from argv: %s', alldatafname)
    else:
        alldatafname = config['alldatafname']
        logging.info('alldatafname from config: %s', alldatafname)

    negative = config['negative'] == 'yes'
    logging.info('Using negative data: %s', str(negative))

    all_pulserDACs_data = {}

    # very DEBUG
    plot = False

    for run_number in parameters.index:
        pulserDAC = parameters.at[run_number, 'Pulser DAC']
        if pulserDAC not in config['pulserDACs']:     # not debug anymore?
            continue
        if negative and pulserDAC > 31:
            logging.info('Negative pulses selected, skipping Pulser DAC = %d', pulserDAC)
            continue
        fname = parameters.at[run_number, 'Filename']
        logging.info('Reading file %s . . .', fname)
        h5_file = HDF5RawDataFile(fname)
        # One "record" is one triggered event, stored as a tuple (n, 0)
        # (not sure what the 0 is for)
        #records: list[tuple[int, int]]  = h5_file.get_all_record_ids()
        records = h5_file.get_all_record_ids()
        # assuming no more than 4 peaks per frame ???
        allpeakheights = np.zeros((CHANNELS_PER_CRP, len(records), MAX_PEAKS_PER_EVENT), dtype=np.int16)

        # super jank
        DATA_TO_STORE = ['Peak Position', 'Peak Prominence', 'Amplitude', 'Peaking Time', 'Horizontal Offset', 'Baseline', 'Baseline Standard Deviation', 'Peak Area']
        alldata = {}
        for datanameind, dataname in enumerate(DATA_TO_STORE):
            if datanameind < 2:
                alldata[dataname] = np.zeros((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT), dtype=np.int16)
            else:
                alldata[dataname] = np.zeros((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT), dtype=np.float64)

        #alldata = np.zeros((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT), dtype=[('Peak Position', 'i2'), ('Peak Prominence', 'i2'), ('Amplitude', 'f8'), ('Peaking Time', 'f8'), ('Horizontal Offset', 'f8'), ('Baseline', 'f8'), ('Baseline Standard Deviation', 'f8'), ('Peak Area', 'f8')])

        #alldata = []    # new pedestal-finding terrible
        first = True
        #for rec in records[:1]: # DEBUG
        for rec in records:
            logging.info('* Record: %s', str(rec))
            assert rec[1] == 0
            # One "fragment" is either data and metadata from a single WIB or some
            # other metadata about the event.
            fragpaths: list[str] = h5_file.get_fragment_dataset_paths(rec)
            for fragnum, fragpath in enumerate(fragpaths):
                #if fragnum == 0:    # DEBUG
                #    continue
                #logging.debug('* * Fragment: %s', fragpath)
                frag = h5_file.get_frag(fragpath)
                fragheader = frag.get_header()
                assert fragheader.version == 5
                # Only attempt to process WIB data, not metadata fragments
                #logging.debug('* * * FragmentType: %s', fragheader.fragment_type)
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
                    ##logging.debug('* * * data shape: %s', str(data.shape))
                    #assert data.shape == (256, TIMESTAMPS_PER_FRAME)
                    if first:
                        cutoff = round(np.std(data[0]) * 6)
                        logging.info('Using cutoff: %d', cutoff)
                        first = False
                    firstchan = True    # this solves problem with first channel of 256 not having reasonable peaks
                    if negative:
                        data = -data - 49152    # flip over data for negative peaks
                    #for i in range(1):
                    ## DEBUG
                    #n = 245 if fragnum == 1 else 0
                    for i in range(256):
                        chnum = CHANNEL_MAP.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, i)
                        #logging.debug('* * * * Channel number: %d', chnum)
                        ## DEBUG!!
                        #if not (chnum in range(1870, 1885) or chnum in range(2850, 2880)):
                        #    continue
                        if negative and chnum > 1903:
                            #logging.debug('Negative selected! Skipping collection channel %d . . .', chnum)
                            continue
                        # DEBUG?
                        if chnum in BAD_CHANNELS:
                            #logging.debug('Known bad channel %d. Skipping . . .', chnum)
                            continue
                        heights, peak_locations, fullpeakinfo = peak_heights(data[i], cutoff)
                        # set to 0 (effectively delete) "peaks" too close to
                        # signal edge to be accurately measured
                        if peak_locations[0] < 7:
                            #heights[0] = 0
                            heights = heights[1:]
                            peak_locations = peak_locations[1:]
                            #logging.debug('First peak too close to edge! Ignoring it . . .')
                        elif peak_locations[-1] > TIMESTAMPS_PER_FRAME - 9:
                            #heights[-1] = 0
                            heights = heights[:-1]
                            peak_locations = peak_locations[:-1]
                            #logging.debug('Last peak too close to edge! Ignoring it . . .')
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
                                #logging.debug('Known bad channel: Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights))
                                pass
                            else:
                                logging.warning('Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s, full peak info: %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights), str(fullpeakinfo))
                            # easy, just don't write bad ones at this point
                            continue
                        # oops I had this in an 'else' and that always skipped writing channel 0 data
                        allpeakheights[chnum, rec[0] - 1, :l] = heights
                        
                        # pulse fitting
                        #for loc, height in zip(peak_locations, heights):
                        for ind, loc in enumerate(peak_locations):
                            height = heights[ind]
                            WINDOW_BEFORE = -200
                            WINDOW_BEFORE_FIT = -WINDOW_BEFORE - 4
                            WINDOW_AFTER = 200
                            WINDOW_AFTER_FIT = -WINDOW_BEFORE + 6
                            WINDOW_DISPLAY_BEFORE = -20
                            WINDOW_DISPLAY_AFTER = WINDOW_AFTER
                            # off by one or not? think about it
                            if loc < -WINDOW_BEFORE or loc > TIMESTAMPS_PER_FRAME - WINDOW_AFTER:
                                continue
                            #logging.debug("loc: %s, height: %s", str(loc), str(height))
                            #window = data[i][loc - 6:loc + 7]# - data[i][loc - 6]
                            # size of window to draw on plot and to use for fit
                            window = data[i][loc + WINDOW_BEFORE:loc + WINDOW_AFTER].astype(np.float64)# - data[i][loc - 6]
                            fitwindow = window[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT]
                            # window for baseline
                            bwindow = window[:WINDOW_BEFORE_FIT]
                            baseline = np.mean(bwindow)
                            baselinestd = np.std(bwindow)
                            #popt, pcov = fit_peak(fitwindow, height)   # only fit rising edge?
                            #popt, pcov = fit_peak(fitwindow, height, baseline)
                            popt, pcov = fit_peak2(fitwindow - baseline, height)
                            #real_amplitude = popt[0] / 10.11973588
                            #logging.debug("popt: %s\npcov: %s", str(popt), str(pcov))
                            area = scipy.integrate.quad(filtfunc.f_fast, 0., 8., args=(popt[0], popt[1]))
                            #logging.debug("area: %s", str(area))

                            # saving data to giant arrays
                            recind = MAX_PEAKS_PER_EVENT * (rec[0] - 1) + ind
                            alldata['Peak Position'][chnum, recind] = loc
                            alldata['Peak Prominence'][chnum, recind] = height
                            alldata['Amplitude'][chnum, recind] = popt[0]
                            alldata['Peaking Time'][chnum, recind] = popt[1]
                            alldata['Horizontal Offset'][chnum, recind] = popt[2]
                            alldata['Baseline'][chnum, recind] = baseline
                            alldata['Baseline Standard Deviation'][chnum, recind] = baselinestd
                            alldata['Peak Area'][chnum, recind] = area[0]
                            # is this really the best way to do this?

                            if plot:
                                # DEBUG / plot-drawing
                                x = (np.arange(len(window)) - WINDOW_BEFORE_FIT) * TIME_PER_SAMPLE
                                # fix to actually use WINDOW_DISPLAY_AFTER ?
                                x2 = np.linspace(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE], x[-1], 2048)
                                #y = filtfunc.f(x2, popt[0], popt[1], popt[2], popt[3])
                                y = filtfunc.f(x2, popt[0], popt[1], popt[2], baseline)
                                #logging.debug("Min vs. max fitted: %.2f", np.max(y) - np.min(y))
                                fig, ax = plt.subplots(figsize=(24, 8))
                                ax.scatter(x[WINDOW_AFTER_FIT:], window[WINDOW_AFTER_FIT:], s=10, label='Data')
                                ax.scatter(x[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT], fitwindow, s=10, label='Data used for fit')
                                ax.scatter(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE:WINDOW_BEFORE_FIT], bwindow[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE:], s=10, label='Data used for baseline')
                                ax.plot(x2, y, linewidth=0.5, label='Fit')
                                pmax = np.max(window)
                                #ax.text(7, pmax, f'A0: {popt[0]:8.2f}\nrA: {real_amplitude:8.2f}\ntp: {popt[1]:8.2f}\nh:  {popt[2]:8.2f}\nb:  {popt[3]:8.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
                                ax.text(7, pmax, f'A0: {popt[0]:8.2f}\nrA: {real_amplitude:8.2f}\ntp: {popt[1]:8.2f}\nh:  {popt[2]:8.2f}\nb:  {baseline:8.2f}\nbs: {baselinestd:8.2f}\na: {area[0]:9.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
                                ax.legend(loc='upper left')
                                ax.set(
                                        title=f'Pulser DAC = {pulserDAC}, Record {rec[0]}, Channel {chnum}, Peak at {loc}',
                                        xlabel='Time (microseconds)',
                                        ylabel='ADC Counts',
                                        xticks=np.arange(round(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE]), round(x[-1]), 2),
                                        )
                                #fig.savefig(f'debug/DEBUGPLOT_{i}_{loc}.png')
                                fig.savefig(f'{config["fitplotsdir"]}{pulserDAC}_{rec[0]}_{chnum}_{loc}.png')
                                plt.close()
                                #break   # DEBUG

        alldata2 = {k: pd.DataFrame(v).stack() for k, v in alldata.items()}
        all_pulserDACs_data[pulserDAC] = pd.DataFrame(alldata2)
        # COMMENTED FOR PLOTFITTING TO FIRST CHANNEL ONLY
        #allpeakheights_flatter = allpeakheights.reshape((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT))
        #allpeakheights_masked = apply_mask(allpeakheights_flatter, BAD_CHANNELS)
        #write_peaksfile(parameters.at[run_number, 'Peaksfile'], allpeakheights_masked)
        first = False
    all_pulserDACs_df = pd.concat(all_pulserDACs_data)
    all_pulserDACs_df.to_pickle(f'{alldatafname}.pkl')
    # DEBUG
    all_pulserDACs_df.to_csv(f'{alldatafname}.tsv', sep='\t')#, float_format='.2f')

if __name__ == '__main__':
    main()

