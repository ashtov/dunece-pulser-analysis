import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import findpeaks

def find_averages_per_channel(config, parameters):
    """Converts from raw peaks data files to average, std deviation, possibly
    other statistics per channel."""
    # maybe just set first dimension to 64 manually?
    allavgs = np.zeros((parameters.loc[slice(None), 'Pulser DAC'].max() + 1, findpeaks.CHANNELS_PER_CRP))
    allstds = np.zeros((parameters.loc[slice(None), 'Pulser DAC'].max() + 1, findpeaks.CHANNELS_PER_CRP))
    allcounts = np.zeros((parameters.loc[slice(None), 'Pulser DAC'].max() + 1, findpeaks.CHANNELS_PER_CRP), dtype=np.int32)
    for run_number in parameters.index:
        peaksfname = parameters.at[run_number, 'Peaksfile']
        logging.info('Reading peaks info from %s . . .', peaksfname)
        peaks = findpeaks.read_peaksfile(peaksfname, config)
        logging.info('Done reading peaks info')
        #print(peaks)
        #print(peaks.shape)
        pulserDAC = parameters.at[run_number, 'Pulser DAC']
        peaksavg = peaks.mean(axis=1)
        #print(peaksavg)
        #print(peaksavg.shape)
        #print(np.argwhere(peaksavg < 1.00))
        #print(peaksavg[141])
        ## this section necessary to eliminate incorrect values due to
        ## attempting to find prominence of a peak at the ends of the signal
        ## while generating peaks data.
        #residuals = peaks - np.expand_dims(peaksavg, 1)
        ## usually range is nowhere near 1000, I think? is this safe? should rerun peak-finding
        #np.ma.masked_where(np.abs(residuals) > 500, peaks, copy=False)
        #peaksavg = peaks.mean(axis=1)
        ## end special section
        peaksstd = peaks.std(axis=1)
        peakscount = peaks.count(axis=1)
        allavgs[pulserDAC] = peaksavg
        allstds[pulserDAC] = peaksstd
        allcounts[pulserDAC] = peakscount

    print(np.argwhere(allavgs[1:11] < 1.00))
    np.save(f'{config["results_dir"]}allavgs.npy', allavgs)
    np.save(f'{config["results_dir"]}allstds.npy', allstds)
    np.save(f'{config["results_dir"]}allcounts.npy', allcounts)
    np.savetxt(f'{config["results_dir"]}allavgs.tsv', allavgs, fmt='%.2f', delimiter='\t')
    np.savetxt(f'{config["results_dir"]}allstds.tsv', allstds, fmt='%.2f', delimiter='\t')
    np.savetxt(f'{config["results_dir"]}allcounts.tsv', allcounts, fmt='%d', delimiter='\t')

def plots_one_pulserDAC_channel_range(config, parameters, pulserDAC, chmin, chmax):
    """quick and dirty fix later or don't use"""
    parameters2 = parameters.set_index('Pulser DAC')
    peaksfname = parameters2.at[pulserDAC, 'Peaksfile']
    logging.info('Reading peaks info from %s . . .', peaksfname)
    peaks = findpeaks.read_peaksfile(peaksfname, config)
    logging.info('Done reading peaks info')
    for i in range(chmin, chmax):
        logging.info('Generating bonus plots at Pulser DAC = %d for channel %d . . .', pulserDAC, i)
        fig, ax = plt.subplots(2, figsize=(8, 8), layout='constrained')
        ax[0].hist(peaks[i], bins=20)
        #ax[1].plot(peaks[i], linewidth=0.5)
        ax[1].scatter(np.arange(len(peaks[i])), peaks[i], s=4)
        fig.suptitle(f'Distribution of peaks at Pulser DAC = {pulserDAC}, channel {i}')
        ax[0].set(
                title='Histogram of peak heights',
                xlabel='Peak height (14-bit ADC counts)',
                ylabel=f'Count (total = {peaks[i].count()})'
                )
        ax[1].set(
                title='Peak heights by ordinal position',
                xlabel='Ordinal position (should be meaningless)',
                ylabel='Peak height (14-bit ADC counts)'
                )
        fig.savefig(f'{config["bonus_plots_dir"]}{pulserDAC}_{i}.png')
        plt.close()


def load_averages_per_channel(config):
    """Loads averages, stds, and counts from npy files already generated with
    find_averages_per_channel()"""
    allcounts = np.ma.masked_equal(np.load(f'{config["results_dir"]}allcounts.npy'), 0, copy=False)
    allavgs = np.ma.MaskedArray(np.load(f'{config["results_dir"]}allavgs.npy'), mask=allcounts.mask)
    allstds = np.ma.MaskedArray(np.load(f'{config["results_dir"]}allstds.npy'), mask=allcounts.mask)
    return allavgs, allstds, allcounts

def calc_gain(config, parameters):
    """Calculates gain???"""
    allavgs, allstds, allcounts = load_averages_per_channel(config)
    #allcounts = np.ma.masked_equal(allcount, 0, copy=False)
    print(allavgs)
    print(allavgs.shape)
    allerrors = allstds / np.sqrt(allcounts)
    #fullavg = allavgs[1:].mean(axis=1)
    fullavg = allavgs.mean(axis=1)

    x = parameters.loc[slice(None), 'Pulser DAC']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, fullavg)
    fig.savefig(config["average_vs_pulserDAC_fname"])
    plt.close()

    diffs = np.diff(allstds, axis=1)
    np.savetxt(f'{config["results_dir"]}alldiffs.tsv', diffs, fmt='%.2f', delimiter='\t')
    whichbig = np.nonzero(np.abs(diffs) > 5.0)
    whichbignext = (whichbig[0], whichbig[1] + 1)
    bigdiffs = diffs[whichbig]
    bigvals = allstds[whichbig]
    bigvalsnext = allstds[whichbignext]
    bigdiffcount = {}
    for i in range(len(bigvals)):
        if i != 0 and whichbig[0][i] != whichbig[0][i - 1]:
            logging.info('')  # terrible way to print newline
        logging.info('Anomalously large STD: run %2d, channel %4d, diff %6.2f, std %5.2f, nextstd %5.2f', whichbig[0][i], whichbig[1][i], bigdiffs[i], bigvals[i], bigvalsnext[i])
        bigdiffcount.setdefault(whichbig[1][i], []).append(whichbig[0][i])
    bdclk = list(bigdiffcount.keys())
    bdclk.sort(key=lambda x: len(bigdiffcount[x]), reverse=True)
    for k in bdclk:
        logging.info('%4d: %s, %s', k, str(bigdiffcount[k]), str(diffs[bigdiffcount[k], k]))

    for i in range(allavgs.shape[0]):
        logging.info('Generating plots for Pulser DAC = %d . . .', i)
        fig, ax = plt.subplots(4, 2, figsize=(16, 16))
        ax[0][0].hist(allavgs[i], bins=30)
        ax[1][0].hist(allerrors[i], bins=30)
        ax[2][0].hist(allstds[i], bins=30)
        ax[3][0].hist(allcounts[i], bins=30)
        ax[0][1].plot(allavgs[i], linewidth=0.5)
        ax[1][1].plot(allerrors[i], linewidth=0.5)
        ax[2][1].plot(allstds[i], linewidth=0.5)
        ax[3][1].plot(allcounts[i], linewidth=0.5)
        fig.savefig(f'{config["plots_by_pulserDAC_dir"]}avgs_{i}.png')
        plt.close()


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(allstds[:60].mean(axis=0), linewidth=0.7)
    ax.set(
            title='CRP4 Standard Deviation of Pulse Heights by Channel (average over Pulser DAC=1-59)',
            xlabel='CRP Channel',
            ylabel='Standard Deviation of Pulse Height (14-bit ADC Counts)',
            )
    fig.savefig(config["average_std_fname"])
    plt.close()

def plot_std_vs_pulserDAC(config, channels, outfname):
    """Plot standard deviation vs. pulser DAC setting for list of channels."""
    allavgs, allstds, allcounts = load_averages_per_channel(config)
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(64)
    for i in channels:
        ax.plot(x, allstds[:, i], linewidth=0.7, label=str(i))
    ax.set(
            title='Pulse Height Standard Deviation vs. Pulser DAC Setting',
            xlabel='Pulser DAC Setting (base 10)',
            ylabel='Pulse Height Standard Deviation (ADC Counts)',
            yticks=np.arange(0, np.max(allstds[:, channels]) // 10 * 10 + 3, 10),
            xticks=np.arange(0, 65, 5),
            xlim=(0, 64),
            )
    ax.legend()
    fig.savefig(outfname)
    plt.close()


def main():
    findpeaks.set_loglevel()
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} CONFIGFILE')
        return 0
    configfname = pathlib.Path(sys.argv[1])
    if not configfname.exists():
        print(f'No such file: {configfname}')
        return 0
    config = findpeaks.read_config(configfname)
    parametersfname = pathlib.Path(config['parametersfname'])
    if not parametersfname.exists():
        print(f'No such file: {parametersfname}')
        return 0
    parameters = findpeaks.read_parameters(parametersfname)

    badchans = [1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 2853, 2856, 2859, 2862, 2865, 2871, 2874]

    # lol fix this
    if len(sys.argv) < 6:
        #find_averages_per_channel(config, parameters)
        calc_gain(config, parameters)
        #plot_std_vs_pulserDAC(config, badchans, 'badchans_plot.png')
    else:
        plots_one_pulserDAC_channel_range(config, parameters, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

if __name__ == '__main__':
    main()
