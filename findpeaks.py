import logging
import sys
import pathlib

import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import click

from hdf5libs import HDF5RawDataFile
import detchannelmaps
import daqdataformats
import fddetdataformats
import rawdatautils.unpack.wib2

import filtfunc

# Channel map maps from internal DAQ numbering by WIB to official channel
# number. Should be the same always for CRP.
CHANNEL_MAP_NAME = 'VDColdboxChannelMap'
CHANNEL_MAP = detchannelmaps.make_map(CHANNEL_MAP_NAME)

CHANNELS_PER_CRP = 3072
TIMESTAMPS_PER_FRAME = 8192     # in CRP5 data, this is sometimes 3073?
MAX_PEAKS_PER_EVENT = 4
TIME_PER_SAMPLE = 0.512 # microseconds

# conversion factor to correct LArASIC pulse response function "amplitude" to
# amplitude in ADC counts
MAGIC_CONVERSION_FACTOR = 10.11973588 
MAX_HEIGHT = 16384 * MAGIC_CONVERSION_FACTOR

# constants for plotting and undershoot-finding
WINDOW_BEFORE = -200
WINDOW_BEFORE_FIT = -WINDOW_BEFORE - 4      # relative to window start
WINDOW_AFTER = 200
WINDOW_AFTER_FIT = -WINDOW_BEFORE + 6
WINDOW_DISPLAY_BEFORE = -20                 # not relative ot window start
WINDOW_DISPLAY_AFTER = WINDOW_AFTER
WINDOW_EXTREMAL_START = WINDOW_AFTER_FIT    # relative to window start
WINDOW_UNDERSHOOT_LEN = 25                  # window for undershoot position only
WINDOW_EXTREMAL_END = WINDOW_EXTREMAL_START + 150   # window for recovery position
WINDOW_EXTREMAL_CORRECTION = WINDOW_EXTREMAL_START - WINDOW_BEFORE_FIT  # not used correctly in v3!!!

def peak_heights(arr, cutoff: int):
    """Find heights of peaks in 1d numpy array arr with heights greater than
    the height of the highest - cutoff."""
    peaks, peakinfo = scipy.signal.find_peaks(arr, distance=2000, prominence=(cutoff, None), wlen=20, width=3.5)
    logging.debug('* * * * Peaks: %s\n* * * * Peak info: %s', str(peaks), str(peakinfo))
    return peakinfo['prominences'], peaks, peakinfo

# old version, unused?
def fit_peak(arr, height, base=None):
    """Attempts to fit filtfunc.f() to the peak beginning at index peakpos in
    arr and returns fit parameters."""
    # FIX THIS MESS LATER, filtfunc should probably not have h and b in it
    # or the peak contained entirely in arr, with nothing else?
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

def apply_mask(arr, skip_channel):
    """Applies mask to 2d array of peak heights
    Blocks out zeros and known bad channels"""
    arr_masked = np.ma.masked_equal(arr, 0, copy=False)
    for i in skip_channel:
        arr_masked[i] = np.ma.masked
    return arr_masked

def read_parameters(parametersfname):
    """Read parameters file into pandas dataframe
    Parameters file stores correspondence between data files, Pulser DAC
    settings, and peaks result file locations."""
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
    logging.debug('Config: %s', str(config))
    return config

def read_peaksfile(fname, config):
    """Read file with peaks data
    This function is not used in this file, but is provided for the
    convenience of other scripts which may read peaks files."""
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

def frames_in_file_iter(fname: str):
    """Iterable for looping over frames in a file"""
    raise NotImplementedError()

@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output; useful for debugging.')
#@click.option('--config-file', type=click.Path(exists=True, dir_okay=False), help='Location of config file. See documentation for format.')
@click.option('--plot', '-p', is_flag=True, help='Plot raw pulse waveforms. --plots-dir must also be specified. --firstonly recommended as generating plots is relatively very slow.')
@click.option('--plots-dir', type=click.Path(exists=True, file_okay=False, writable=True, readable=False), help='Directory to plot pulse waveforms. No effect if --plot is not specified. Plots will have filenames like [pulserDAC]_[record #]_[channel]_[pulse location].png')
@click.option('--existing-data', type=click.Path(exists=True, dir_okay=False), help='Do not process data, just generate plots. Only useful with --plot.')
@click.option('--firstonly', '-f', is_flag=True, help='Process only first pulse for each channel, Pulser DAC setting. Useful for testing or generating plots.')
@click.option('--output', '-o', type=click.Path(dir_okay=False, writable=True, readable=False), required=True, help='Filename for output (in pandas pickle format).')
@click.option('--inputs-file', type=click.Path(exists=True, dir_okay=False), required=True, help='Location of file with list of input files and their corresponding Pulser DAC setting.')
@click.option('--skip-channel', '-s', type=click.IntRange(0, CHANNELS_PER_CRP), multiple=True, help='Channels to skip processing in input data. Useful to ignore bad channels with garbage data. Can be specified multiple times.')
@click.option('--pulser-dac', '-d', type=click.IntRange(0, 63), multiple=True, help='Pulser DAC setting to process. Can be specified multiple times, in which case they will be processed in the order specified. If not specified, process all Pulser DAC settings in inputs file.')
@click.option('--negative', '-n', is_flag=True, help='Process negative pulse data instead of positive.')
def main(verbose, plot, plots_dir, existing_data, firstonly, output, inputs_file, skip_channel, pulser_dac, negative):
    if verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel)
    if plot and not plots_dir:
        raise click.BadOptionUsage('plot', 'Must specify --plots-dir when --plot is used')
    parameters = read_parameters(inputs_file)
    all_pulserDACs_data = {}

    for run_number in parameters.index:
        pulserDAC = parameters.at[run_number, 'Pulser DAC']
        if pulserDAC not in pulser_dac:     # not debug anymore?
            continue
        if negative and pulserDAC > 31:
            logging.info('Negative pulses selected, skipping Pulser DAC = %d', pulserDAC)
            continue
        fname = parameters.at[run_number, 'Filename']
        logging.info('Reading Pulser DAC %d in file %s . . .', pulserDAC, fname)
        h5_file = HDF5RawDataFile(fname)
        # One "record" is one triggered event, stored as a tuple (n, 0)
        # (not sure what the 0 is for)
        records = h5_file.get_all_record_ids()
        if firstonly:
            records = records[:1]

        DATA_TO_STORE = ['Peak Position', 'Peak Prominence', 'Amplitude', 'Peaking Time', 'Horizontal Offset', 'Baseline', 'Baseline Standard Deviation', 'Peak Area', 'Undershoot', 'Undershoot Area', 'Undershoot Position', 'Recovery Position', 'Undershoot Start']
        alldata = {}
        for datanameind, dataname in enumerate(DATA_TO_STORE):
            if datanameind < 2 or datanameind >= 10:
                alldata[dataname] = np.zeros((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT), dtype=np.int16)
            else:
                alldata[dataname] = np.zeros((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT), dtype=np.float64)

        if existing_data:
            raise NotImplementedError('--existing-data not implemented yet!')
            all_pulserDACs_df = pd.read_pickle(existing_data)

        first = True
        for rec in records:
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
                    if data.shape != (256, TIMESTAMPS_PER_FRAME):
                        logging.warning('Bad data shape? shape: %s for fragment %s, record %s', str(data.shape), fragpath, str(rec))
                    # calculate peak height cutoff only for the first channel
                    # at each pulser DAC setting
                    if first:
                        cutoff = round(np.std(data[0]) * 6)
                        logging.info('Using cutoff: %d', cutoff)
                        first = False
                    firstchan = True    # this solves problem with first channel of 256 not having reasonable peaks
                    if negative:
                        data = -data - 49152    # flip over data for negative peaks
                    # for CRP5 misconfigured COLDATAs (probably)
                    chrange = (0, 256)
                    #if crp5 and frameheader.link:
                    #    if frameheader.slot == 0:
                    #        chrange = (64, 256)
                    #        logging.debug('Skipping FEMB 9 ASICs 4-7')
                    #    elif frameheader.slot == 2:
                    #        chrange = (0, 192)
                    #        logging.debug('Skipping FEMB 2 ASICs 0-3')

                    for i in range(*chrange):
                        chnum = CHANNEL_MAP.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, i)
                        logging.debug('* * * * Channel number: %d', chnum)
                        if negative and chnum > 1903:
                            logging.debug('Negative selected! Skipping collection channel %d . . .', chnum)
                            continue
                        # DEBUG?
                        if chnum in skip_channel:
                            logging.debug('Known bad channel %d. Skipping . . .', chnum)
                            continue

                        #if existing_data:
                        #    relevant_ind = MAX_PEAKS_PER_EVENT * (rec[0] - 1)
                        #    relevant_data = all_pulserDACs_df.loc[(pulserDAC, chnum, slice(relevant_ind, relevant_ind + MAX_PEAKS_PER_EVENT - 1)), slice(None)].droplevel([0, 1])
                        #else:
                        heights, peak_locations, fullpeakinfo = peak_heights(data[i], cutoff)

                        # set to 0 (effectively delete) "peaks" too close to
                        # signal edge to be accurately measured
                        if peak_locations[0] < 7:
                            heights = heights[1:]
                            peak_locations = peak_locations[1:]
                            logging.debug('First peak too close to edge! Ignoring it . . .')
                        elif peak_locations[-1] > TIMESTAMPS_PER_FRAME - 9:
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
                            if chnum in skip_channel:
                                logging.debug('Known bad channel: Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights))
                                pass
                            else:
                                logging.warning('Mismatched peaks in channel %d (fragment %s, record %s): expected %s, got %s, heights %s, full peak info: %s', chnum, fragpath, str(rec), str(expected_peaks), str(peak_locations), str(heights), str(fullpeakinfo))
                            # easy, just don't write bad ones at this point
                            continue
                        
                        # pulse fitting
                        for ind, loc in enumerate(peak_locations):
                            height = heights[ind]
                            # off by one or not? think about it
                            # skip pulse fitting for peaks too close to ends of signal
                            if loc < -WINDOW_BEFORE or loc > TIMESTAMPS_PER_FRAME - WINDOW_AFTER:
                                continue
                            logging.debug("loc: %s, height: %s", str(loc), str(height))
                            # size of window to draw on plot and to use for fit
                            # need np.float64 for fitting, according to scipy?
                            window = data[i][loc + WINDOW_BEFORE:loc + WINDOW_AFTER].astype(np.float64)
                            fitwindow = window[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT]
                            # window for baseline
                            bwindow = window[:WINDOW_BEFORE_FIT]
                            baseline = np.mean(bwindow)
                            baselinestd = np.std(bwindow)
                            popt, pcov = fit_peak2(fitwindow - baseline, height)
                            amp, tp, h = popt
                            real_amplitude = popt[0] / 10.11973588
                            logging.debug("popt: %s\npcov: %s", str(popt), str(pcov))
                            area_tuple = scipy.integrate.quad(filtfunc.f_fast, 0., 8., args=(amp, tp))
                            logging.debug("area: %s", str(area_tuple))
                            area = area_tuple[0]

                            # undershoot / overshoot finding
                            # Algorithm:
                            # Undershoot is the lowest point below baseline in the window between
                            # first point after pulse that's below baseline and first point after
                            # that that's back above baseline.
                            # Overshoot is the highest point above baseline between the first point
                            # above baseline after the first local minimum after pulse and the
                            # first point after that that's back below baseline.
                            # The recorded "undershoot" is the larger in magnitude of these two.
                            postwindow = window[WINDOW_EXTREMAL_START:WINDOW_EXTREMAL_END]
                            postwindow_diff = postwindow - baseline
                            # undershoot-finding:
                            try:
                                us_start = np.flatnonzero(postwindow_diff < 0)[0]
                            except IndexError:
                                us_start = postwindow_diff.size - 1
                                logging.warning('Could not find undershoot start in window! Pulser DAC %d, Record %s, Channel %d, peak at %d', pulserDAC, rec, chnum, loc)
                            try:
                                us_end = np.flatnonzero(postwindow_diff[us_start:] > 0)[0] + us_start
                            except IndexError:
                                us_end = postwindow_diff.size - 1
                                logging.warning('Could not find undershoot end in window! Pulser DAC %d, Record %s, Channel %d, peak at %d', pulserDAC, rec, chnum, loc)
                            us_window = postwindow_diff[us_start:us_end]
                            try:
                                us_min_loc = np.argmin(us_window) + us_start
                                us_min = postwindow_diff[us_min_loc]
                            except ValueError:
                                # should happen if us_start is at end? maybe somehow fold this into one try-except?
                                us_min_loc = -1
                                us_min = 1
                            us_area = np.sum(us_window) * TIME_PER_SAMPLE
                            # overshoot-finding:
                            first_lm = np.flatnonzero((postwindow_diff[:-1] - postwindow_diff[1:]) < 0)[0]  # first local minimum
                            os_start = first_lm if first_lm < us_start else us_end
                            try:
                                os_end = np.flatnonzero(postwindow_diff[os_start:] < 0)[0] + os_start
                            except IndexError:
                                os_end = postwindow_diff.size - 1
                                logging.warning('Could not find overshoot end in window! Pulser DAC %d, Record %s, Channel %d, peak at %d', pulserDAC, rec, chnum, loc)
                            os_window = postwindow_diff[os_start:os_end]
                            try:
                                os_max_loc = np.argmax(os_window) + os_start
                                os_max = postwindow_diff[os_max_loc]
                            except ValueError:
                                # numpy throws ValueError when argmax of an empty array. this means somehow both os_start and os_end are at the end of array.
                                # most likely because undershoot does not end within window
                                os_max_loc = -1
                                os_max = -1         # this should make undershoot always get chosen below
                            os_area = np.sum(os_window) * TIME_PER_SAMPLE
                            # choosing one (necessary?)
                            if os_max > -us_min:
                                extremal_pos, extremal_val, extremal_start, extremal_end, extremal_area = os_max_loc + WINDOW_EXTREMAL_CORRECTION, os_max, os_start + WINDOW_EXTREMAL_CORRECTION, os_end + WINDOW_EXTREMAL_CORRECTION, os_area
                            else:
                                extremal_pos, extremal_val, extremal_start, extremal_end, extremal_area = us_min_loc + WINDOW_EXTREMAL_CORRECTION, us_min, us_start + WINDOW_EXTREMAL_CORRECTION, us_end + WINDOW_EXTREMAL_CORRECTION, us_area

                            # saving data to giant arrays
                            recind = MAX_PEAKS_PER_EVENT * (rec[0] - 1) + ind
                            alldata['Peak Position'][chnum, recind] = loc
                            alldata['Peak Prominence'][chnum, recind] = height
                            alldata['Amplitude'][chnum, recind] = amp
                            alldata['Peaking Time'][chnum, recind] = tp
                            alldata['Horizontal Offset'][chnum, recind] = h
                            alldata['Baseline'][chnum, recind] = baseline
                            alldata['Baseline Standard Deviation'][chnum, recind] = baselinestd
                            alldata['Peak Area'][chnum, recind] = area
                            alldata['Undershoot'][chnum, recind] = extremal_val
                            alldata['Undershoot Position'][chnum, recind] = extremal_pos
                            alldata['Recovery Position'][chnum, recind] = extremal_end
                            alldata['Undershoot Start'][chnum, recind] = extremal_start
                            alldata['Undershoot Area'][chnum, recind] = extremal_area
                            # is this really the best way to do this?

                            if plot:
                                # DEBUG / plot-drawing
                                x = (np.arange(len(window)) - WINDOW_BEFORE_FIT) * TIME_PER_SAMPLE
                                # fix to actually use WINDOW_DISPLAY_AFTER ?
                                x2 = np.linspace(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE], x[-1], 2048)
                                y = filtfunc.f(x2, amp, tp, h, baseline)
                                logging.debug("Min vs. max fitted: %.2f", np.max(y) - np.min(y))
                                fig, ax = plt.subplots(figsize=(24, 8), layout='constrained')
                                ax.scatter(x[WINDOW_AFTER_FIT:], window[WINDOW_AFTER_FIT:], s=10, label='Data')
                                ax.scatter(x[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT], fitwindow, s=10, label='Data used for fit')
                                ax.scatter(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE:WINDOW_BEFORE_FIT], bwindow[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE:], s=10, label='Data used for baseline')
                                ax.scatter(x=x[extremal_start + WINDOW_BEFORE_FIT], y=postwindow[extremal_start - WINDOW_EXTREMAL_CORRECTION], s=10, label='Undershoot start')
                                ax.scatter(x=x[extremal_pos + WINDOW_BEFORE_FIT], y=postwindow[extremal_pos - WINDOW_EXTREMAL_CORRECTION], s=10, label='Undershoot')
                                ax.scatter(x=x[extremal_end + WINDOW_BEFORE_FIT], y=postwindow[extremal_end - WINDOW_EXTREMAL_CORRECTION], s=10, label='Recovery')
                                ax.plot(x2, y, linewidth=0.5, label='Fit')
                                pmax = np.max(window)
                                ax.text(7, pmax, f'A0: {amp:9.2f}\nrA: {real_amplitude:9.2f}\ntp: {tp:9.2f}\nh:  {h:9.2f}\nb:  {baseline:9.2f}\nbs: {baselinestd:9.2f}\na: {area:10.2f}\nu: {extremal_val:10.2f}\nup: {extremal_pos:9d}\nus: {extremal_start:9d}\nue: {extremal_end:9d}\nua: {extremal_area:9.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
                                ax.legend(loc='upper left')
                                ax.set(
                                        title=f'Pulser DAC = {pulserDAC}, Record {rec[0]}, Channel {chnum}, Peak at {loc}',
                                        xlabel='Time (microseconds)',
                                        ylabel='ADC Counts',
                                        xticks=np.arange(round(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE]), round(x[-1]), 2),
                                        )
                                fig.savefig(f'{plots_dir}/{pulserDAC}_{rec[0]}_{chnum}_{loc}.png')
                                plt.close()
                            if firstonly:
                                break

        alldata2 = {k: pd.DataFrame(v).stack() for k, v in alldata.items()}
        all_pulserDACs_data[pulserDAC] = pd.DataFrame(alldata2)
        first = False
    all_pulserDACs_df = pd.concat(all_pulserDACs_data)
    all_pulserDACs_df.to_pickle(output)

if __name__ == '__main__':
    main()

