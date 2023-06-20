import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import findpeaks

def find_averages_per_channel(parameters):
    """Converts from raw peaks data files to average, std deviation, possibly
    other statistics per channel."""
    # maybe just set first dimension to 64 manually?
    allavgs = np.zeros((parameters.loc[slice(None), 'Pulser DAC'].max() + 1, findpeaks.CHANNELS_PER_CRP))
    allstds = np.zeros((parameters.loc[slice(None), 'Pulser DAC'].max() + 1, findpeaks.CHANNELS_PER_CRP))
    allcounts = np.zeros((parameters.loc[slice(None), 'Pulser DAC'].max() + 1, findpeaks.CHANNELS_PER_CRP), dtype=np.int32)
    for run_number in parameters.index:
        peaksfname = parameters.at[run_number, 'Peaksfile']
        logging.info('Reading peaks info from %s . . .', peaksfname)
        peaks = findpeaks.read_peaksfile(peaksfname)
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
    np.save('allavgs.npy', allavgs)
    np.save('allstds.npy', allstds)
    np.save('allcounts.npy', allcounts)
    #np.savetxt('allavgs.tsv', allavgs, fmt='%.2f', delimiter='\t')
    #np.savetxt('allstds.tsv', allstds, fmt='%.2f', delimiter='\t')
    #np.savetxt('allcounts.tsv', allcounts, fmt='%d', delimiter='\t')

def plots_one_pulserDAC_channel_range(parameters, pulserDAC, chmin, chmax):
    """quick and dirty fix later or don't use"""
    peaksfname = parameters.at[pulserDAC + 20268, 'Peaksfile']
    logging.info('Reading peaks info from %s . . .', peaksfname)
    peaks = findpeaks.read_peaksfile(peaksfname)
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
        fig.savefig(f'bonusplots/{pulserDAC}_{i}.png')
        plt.close()


def load_averages_per_channel():
    """Loads averages, stds, and counts from npy files already generated with
    find_averages_per_channel()"""
    allcounts = np.ma.masked_equal(np.load('allcounts.npy'), 0, copy=False)
    allavgs = np.ma.MaskedArray(np.load('allavgs.npy'), mask=allcounts.mask)
    allstds = np.ma.MaskedArray(np.load('allstds.npy'), mask=allcounts.mask)
    return allavgs, allstds, allcounts

def calc_gain(parameters):
    """Calculates gain???"""
    allavgs, allstds, allcounts = load_averages_per_channel()
    #allcounts = np.ma.masked_equal(allcount, 0, copy=False)
    print(allavgs)
    print(allavgs.shape)
    allerrors = allstds / np.sqrt(allcounts)
    fullavg = allavgs[1:].mean(axis=1)

    x = parameters.loc[slice(None), 'Pulser DAC']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, fullavg)
    fig.savefig('average_gain.png')
    plt.close()

    diffs = np.diff(allstds, axis=1)
    np.savetxt('alldiffs.tsv', diffs, fmt='%.2f', delimiter='\t')
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
    #    logging.info('Generating plots for Pulser DAC = %d . . .', i)
    #    fig, ax = plt.subplots(4, 2, figsize=(16, 16))
    #    ax[0][0].hist(allavgs[i], bins=30)
    #    ax[1][0].hist(allerrors[i], bins=30)
    #    ax[2][0].hist(allstds[i], bins=30)
    #    ax[3][0].hist(allcounts[i], bins=30)
    #    ax[0][1].plot(allavgs[i], linewidth=0.5)
    #    ax[1][1].plot(allerrors[i], linewidth=0.5)
    #    ax[2][1].plot(allstds[i], linewidth=0.5)
    #    ax[3][1].plot(allcounts[i], linewidth=0.5)
    #    fig.savefig(f'plots/avgs_{i}.png')
    #    plt.close()


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(allstds[:60].mean(axis=0), linewidth=0.7)
    ax.set(
            title='CRP4 Standard Deviation of Pulse Heights by Channel (average over Pulser DAC=1-59)',
            xlabel='CRP Channel',
            ylabel='Standard Deviation of Pulse Height (14-bit ADC Counts)',
            )
    fig.savefig('average_std_across_all_pulser_settings_by_channel.png')
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} FILELIST')
        return 0
    parametersfname = pathlib.Path(sys.argv[1])
    if not parametersfname.exists():
        print(f'No such file: {parametersfname}')
        return 0

    logging.info('Reading parameters file %s . . .', parametersfname)
    parameters = findpeaks.read_parameters(parametersfname)
    logging.info('Successfully read parameters file')

    # lol fix this
    if len(sys.argv) < 5:
        find_averages_per_channel(parameters)
        #calc_gain(parameters)
    else:
        plots_one_pulserDAC_channel_range(parameters, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

if __name__ == '__main__':
    main()
