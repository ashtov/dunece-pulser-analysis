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

existing_data = True

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

    if existing_data:
        all_pulserDACs_df = pd.read_pickle(f'{alldatafname}.pkl')
    all_pulserDACs_data = {}

    # very DEBUG
    plot = False
    writeout = True
    crp5 = True

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
        records = h5_file.get_all_record_ids()
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
                    if data.shape != (256, TIMESTAMPS_PER_FRAME):
                        logging.warning('Bad data shape? shape: %s for fragment %s, record %s', str(data.shape), fragpath, str(rec))
                    ##logging.debug('* * * data shape: %s', str(data.shape))
                    #assert data.shape == (256, TIMESTAMPS_PER_FRAME)
                    if negative:
                        data = -data - 49152    # flip over data for negative peaks
                    #for i in range(1):
                    ## DEBUG
                    #n = 245 if fragnum == 1 else 0
                    chrange = (0, 256)
                    #if crp5 and frameheader.link:
                    #    if frameheader.slot == 0:
                    #        chrange = (64, 256)
                    #        #logging.debug('Skipping FEMB 9 ASICs 4-7')
                    #    elif frameheader.slot == 2:
                    #        chrange = (0, 192)
                    #        #logging.debug('Skipping FEMB 2 ASICs 0-3')
                    for i in range(*chrange):
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
                        relevant_ind = MAX_PEAKS_PER_EVENT * (rec[0] - 1)
                        relevant = data_for_pulserDAC.loc[(chnum, slice(relevant_ind, relevant_ind + MAX_PEAKS_PER_EVENT - 1)), slice(None)].droplevel(0)
                        #print(relevant) # double DEBUG
                        #heights = relevant.loc[:, 'Peak Prominence']
                        #peak_locations = relevant.loc[:, 'Peak Position']
                        
                        for ind, relevanti in relevant.iterrows():
                            #print(relevanti)
                            loc, height, amp, tp, h, baseline, baselinestd, area = relevanti
                            loc, height = int(loc), int(height)
                            if loc == 0:
                                continue
                            WINDOW_BEFORE = -200
                            WINDOW_BEFORE_FIT = -WINDOW_BEFORE - 4      # relative to window start
                            WINDOW_AFTER = 200
                            WINDOW_AFTER_FIT = -WINDOW_BEFORE + 6
                            WINDOW_DISPLAY_BEFORE = -20                 # not relative ot window start
                            WINDOW_DISPLAY_AFTER = WINDOW_AFTER
                            WINDOW_EXTREMAL_START = WINDOW_AFTER_FIT    # relative to window start
                            WINDOW_UNDERSHOOT_END = WINDOW_EXTREMAL_START + 50  # window for undershoot position
                            WINDOW_EXTREMAL_END = WINDOW_EXTREMAL_START + 150   # window for recovery position
                            WINDOW_EXTREMAL_CORRECTION = WINDOW_EXTREMAL_START - WINDOW_BEFORE_FIT
                            # off by one or not? think about it
                            if loc < -WINDOW_BEFORE or loc > TIMESTAMPS_PER_FRAME - WINDOW_AFTER:
                                continue
                            ##logging.debug("loc: %s, height: %s", str(loc), str(height))
                            #window = data[i][loc - 6:loc + 7]# - data[i][loc - 6]
                            # size of window to draw on plot and to use for fit
                            window = data[i][loc + WINDOW_BEFORE:loc + WINDOW_AFTER].astype(np.float64)# - data[i][loc - 6]
                            fitwindow = window[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT]
                            # window for baseline
                            bwindow = window[:WINDOW_BEFORE_FIT]
                            baseline = relevant.at[ind, 'Baseline']
                            baselinestd = relevant.at[ind, 'Baseline Standard Deviation']
                            #popt, pcov = fit_peak(fitwindow, height)   # only fit rising edge?
                            #popt, pcov = fit_peak(fitwindow, height, baseline)
                            #popt, pcov = fit_peak2(fitwindow - baseline, height)
                            real_amplitude = amp / 10.11973588
                            ##logging.debug("popt: %s\npcov: %s", str(popt), str(pcov))
                            #area = scipy.integrate.quad(filtfunc.f_fast, 0., 8., args=(popt[0], popt[1]))
                            ##logging.debug("area: %s", str(area))

                            postwindow = window[WINDOW_EXTREMAL_START:WINDOW_EXTREMAL_END]
                            fitrel = filtfunc.f_no_b(np.arange(WINDOW_EXTREMAL_CORRECTION, postwindow.size + WINDOW_EXTREMAL_CORRECTION) * TIME_PER_SAMPLE, amp, tp, h)
                            postwindow_diff = postwindow - fitrel - baseline
                            postwindow_abs = np.abs(postwindow_diff)
                            extremal_ind = np.argmax(postwindow_abs)  # uncorrected
                            extremal_pos = extremal_ind + WINDOW_EXTREMAL_CORRECTION
                            extremal_val = postwindow_diff[extremal_ind]
                            extremal_sign = np.sign(extremal_val)
                            postwindow_signed = postwindow_diff * extremal_sign
                            esc = np.flatnonzero(postwindow_signed[:extremal_ind] < 0)
                            extremal_start = esc[-1] + WINDOW_EXTREMAL_CORRECTION if esc.size > 0 else WINDOW_EXTREMAL_CORRECTION
                            eec = np.flatnonzero(postwindow_signed[extremal_ind:] < 0)
                            extremal_end = eec[0] + extremal_pos if eec.size > 0 else postwindow.size - 1 + WINDOW_EXTREMAL_CORRECTION

                            # earlier version (with even earlier verison inside)
                            #postwindow = window[WINDOW_BEFORE_FIT:]
                            #postwindow_neg = postwindow < baseline
                            #postwindow_neg[:13] = False
                            #postwindow_neg_inds = np.flatnonzero(postwindow_neg)
                            ###print(postwindow_neg_inds)
                            #firstneg_ind = postwindow_neg_inds[0] if postwindow_neg_inds.size > 0 else postwindow.size
                            ##firstpos_after_firstneg_ind = np.flatnonzero(postwindow[firstneg_ind:] > baseline)[0] + firstneg_ind
                            ##undershoot_pos = np.argmin(postwindow[:firstpos_after_firstneg_ind])
                            ###overshoot_pos = np.argmax(
                            ##undershoot = postwindow[undershoot_pos] - baseline
                            #wstart = min(firstneg_ind, 13)  # this used to be 12 for first try on 47-50 batch
                            #postwindow_abs = np.abs(postwindow[wstart:] - baseline)
                            #extremal_pos = np.argmax(postwindow_abs) + wstart
                            #extremal_val = postwindow[extremal_pos] - baseline
                            #extremal_sign = np.sign(extremal_val)
                            #postwindow_signed = (postwindow - baseline) * extremal_sign
                            #esc = np.flatnonzero(postwindow_signed[wstart:extremal_pos] < 0) + wstart
                            #extremal_start = esc[-1] if esc.size > 0 else wstart
                            #eec = np.flatnonzero(postwindow_signed[extremal_pos:] < 0) + extremal_pos
                            #extremal_end = eec[0] if eec.size > 0 else postwindow.size - 1
                            ###logging.debug('firstneg_ind: %d\nfirstpos_after_firstneg_ind: %d\nundershoot_pos: %d\nundershoot: %.2f\nextremal_pos: %d\nextremal_val: %.2f', firstneg_ind, firstpos_after_firstneg_ind, undershoot_pos, undershoot, extremal_pos, extremal_val)
                            ###logging.debug('extremal_pos: %d\nextremal_val: %.2f\nextremal_start: %d\nextremal_end: %d', extremal_pos, extremal_val, extremal_start, extremal_end)
                            ##alldata['Undershoot'][chnum, ind] = undershoot
                            ##alldata['Undershoot Position'][chnum, ind] = undershoot_pos
                            ##alldata['Recovery Position'][chnum, ind] = firstpos_after_firstneg_ind
                            alldata['Undershoot'][chnum, ind] = extremal_val
                            alldata['Undershoot Position'][chnum, ind] = extremal_pos
                            alldata['Recovery Position'][chnum, ind] = extremal_end
                            alldata['Undershoot Start'][chnum, ind] = extremal_start

                            # saving data to giant arrays
                            #recind = MAX_PEAKS_PER_EVENT * (rec[0] - 1) + ind
                            #alldata['Peak Position'][chnum, recind] = loc
                            #alldata['Peak Prominence'][chnum, recind] = height
                            #alldata['Amplitude'][chnum, recind] = popt[0]
                            #alldata['Peaking Time'][chnum, recind] = popt[1]
                            #alldata['Horizontal Offset'][chnum, recind] = popt[2]
                            #alldata['Baseline'][chnum, recind] = baseline
                            #alldata['Baseline Standard Deviation'][chnum, recind] = baselinestd
                            #alldata['Peak Area'][chnum, recind] = area[0]
                            # is this really the best way to do this?

                            if plot:
                                # DEBUG / plot-drawing
                                x = (np.arange(len(window)) - WINDOW_BEFORE_FIT) * TIME_PER_SAMPLE
                                # fix to actually use WINDOW_DISPLAY_AFTER ?
                                x2 = np.linspace(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE], x[-1], 2048)
                                #y = filtfunc.f(x2, popt[0], popt[1], popt[2], popt[3])
                                y = filtfunc.f(x2, amp, tp, h, baseline)
                                #logging.debug("Min vs. max fitted: %.2f", np.max(y) - np.min(y))
                                fig, ax = plt.subplots(figsize=(24, 8))
                                ax.scatter(x[WINDOW_AFTER_FIT:], window[WINDOW_AFTER_FIT:], s=10, label='Data')
                                ax.scatter(x[WINDOW_BEFORE_FIT:WINDOW_AFTER_FIT], fitwindow, s=10, label='Data used for fit')
                                ax.scatter(x[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE:WINDOW_BEFORE_FIT], bwindow[WINDOW_DISPLAY_BEFORE - WINDOW_BEFORE:], s=10, label='Data used for baseline')
                                ax.scatter(x=x[extremal_start + WINDOW_BEFORE_FIT], y=postwindow[extremal_start - WINDOW_EXTREMAL_CORRECTION], s=10, label='Undershoot start')
                                ax.scatter(x=x[extremal_pos + WINDOW_BEFORE_FIT], y=postwindow[extremal_ind], s=10, label='Undershoot')
                                ax.scatter(x=x[extremal_end + WINDOW_BEFORE_FIT], y=postwindow[extremal_end - WINDOW_EXTREMAL_CORRECTION], s=10, label='Recovery')
                                #ax.scatter(x=x[extremal_pos + WINDOW_BEFORE_FIT], y=postwindow[extremal_pos], s=10, label='Extremum')
                                ax.plot(x2, y, linewidth=0.5, label='Fit')
                                pmax = np.max(window)
                                #ax.text(7, pmax, f'A0: {popt[0]:8.2f}\nrA: {real_amplitude:8.2f}\ntp: {popt[1]:8.2f}\nh:  {popt[2]:8.2f}\nb:  {popt[3]:8.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
                                ax.text(7, pmax, f'A0: {amp:8.2f}\nrA: {real_amplitude:8.2f}\ntp: {tp:8.2f}\nh:  {h:8.2f}\nb:  {baseline:8.2f}\nbs: {baselinestd:8.2f}\na: {area:9.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
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
                                break   # DEBUG

        alldata2 = {k: pd.DataFrame(v).stack() for k, v in alldata.items()}
        all_pulserDACs_data[pulserDAC] = pd.DataFrame(alldata2)
        # COMMENTED FOR PLOTFITTING TO FIRST CHANNEL ONLY
        #allpeakheights_flatter = allpeakheights.reshape((CHANNELS_PER_CRP, len(records) * MAX_PEAKS_PER_EVENT))
        #allpeakheights_masked = apply_mask(allpeakheights_flatter, BAD_CHANNELS)
        #write_peaksfile(parameters.at[run_number, 'Peaksfile'], allpeakheights_masked)
        first = False
    if writeout:
        all_pulserDACs_df2 = pd.concat(all_pulserDACs_data)
        all_pulserDACs_df_new = all_pulserDACs_df.join(all_pulserDACs_df2)
        all_pulserDACs_df_new.to_pickle(f'{alldatafname}_bonus2.pkl')
    ## DEBUG
    #all_pulserDACs_df.to_csv(f'{alldatafname}.tsv', sep='\t')#, float_format='.2f')

if __name__ == '__main__':
    main()

