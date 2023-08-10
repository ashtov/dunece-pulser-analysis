import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy

CAP = 0.185e-12         # capcitance of LArASIC pulser's capacitor
V_PER_BIT = 8.08e-3     # at 14 mV / fC LArASIC setting
CHARGE_PER_PULSERDAC = V_PER_BIT * CAP          # injected charge per PulserDAC setting (in theory)
C_fC = CHARGE_PER_PULSERDAC * 1e15              # injected charge per PulserDAC setting in fC
C_e = CHARGE_PER_PULSERDAC / scipy.constants.e  # injected charge per PulserDAC in electrons

@click.group(invoke_without_command=True, chain=True)
@click.argument('filename', type=click.Path(exists=True, dir_okay=False))
@click.option('--statsfile', '-s', type=click.Path(exists=True, dir_okay=False))
@click.option('--statsout', '-o', type=click.Path(writable=True))
@click.option('--charge/--daclevel', '-c/-d', default=True)
@click.option('--crp5/--crp4', default=False)
@click.pass_context
def main(ctx, filename, statsfile, statsout, charge, crp5):
    global C, BAD_CHANNELS, DACLEVELS, CRPNAME
    ctx.ensure_object(dict)
    alldata = pd.read_pickle(filename)
    alldata['Real Amplitude'] = alldata['Amplitude'] / 10.11973588
    ctx.obj['alldata'] = alldata
    click.echo(alldata)
    if statsfile:
        #ctx.obj['areastats'] = pd.read_pickle(statsfile)
        ctx.obj['allstats'] = pd.read_pickle(statsfile)
    else:
        #grouped = alldata.loc[:, 'Peak Area'].groupby(level=['Pulser DAC', 'Channel'], sort=False)
        grouped = alldata.drop(columns='Peak Position').groupby(level=['Pulser DAC', 'Channel'], sort=False)
        #areastats = grouped.agg(['mean', 'std', 'count'])
        allstats = grouped.agg(['mean', 'std', 'count'])
        errs = allstats.loc[:, (slice(None), 'std')].droplevel(1, axis=1) / allstats.loc[:, (slice(None), 'count')].droplevel(1, axis=1).apply('sqrt')
        errs.columns = pd.MultiIndex.from_product((errs.columns, ('err',)))
        click.echo(errs)
        ctx.obj['allstats'] = allstats.join(errs)
        #allstats.loc[:, (slice(None), 'err')] = allstats.loc[:, (slice(None), 'std')] / allstats.loc[:, (slice(None), 'count')].apply('sqrt')
    click.echo(ctx.obj['allstats'])
    ctx.obj['areastats'] = ctx.obj['allstats'].loc[:, 'Peak Area']
    #areastats = ctx.obj['allstats'].loc[:, 'Peak Area']
    #areastats.loc[:, 'err'] = areastats.loc[:, 'std'] / np.sqrt(areastats.loc[:, 'count'])
    #ctx.obj['areastats'] = areastats
    click.echo(ctx.obj['areastats'])
    if statsout:
        #ctx.obj['areastats'].to_pickle(statsout)
        ctx.obj['allstats'].to_pickle(statsout)
    if charge:
        C = C_fC
    else:
        C = 1
    if crp5:
        BAD_CHANNELS = [182, 1424, 1849, 2949, 2997, 3019]
        DACLEVELS = [np.array(i) for i in [
                list(range(2, 31)),
                list(range(2, 14)) + list(range(15, 32)) + list(range(33, 61)),
                ]]
        CRPNAME = 'CRP5'
    else:
        BAD_CHANNELS = [1958, 2250, 2868]
        DACLEVELS = [np.array(i) for i in [
                list(range(2, 4)) + list(range(7, 31)),
                #list(range(2, 31)),                    # DEBUG
                #list(range(2, 11)) + list(range(32, 38)) + list(range(39, 41)) + list(range(42, 50)) + list(range(52, 55)) + [56],  # for CRP4 channels sharing ASIC with 2868
                list(range(2, 20)) + list(range(21, 38)) + list(range(39, 41)) + list(range(42, 51)) + list(range(52, 61)),
                #list(range(2, 31)),
                ]]
        CRPNAME = 'CRP4'


@main.command()
@click.option('--daclevels', nargs=2, type=click.IntRange(1, 64), default=(1, 64))
@click.option('--out-prefix', required=True)
@click.pass_context
def pulserdac_info(ctx, daclevels, out_prefix):
    """Generates plots of summary statistics at daclevels.

    Plots are generated for Pulser DAC settings in range specified (endpoint
    exclusive). Output is saved to files named 1.png, 2.png, etc, appended to
    prefix out-prefix."""
    if daclevels[0] >= daclevels[1]:
        raise click.BadOptionUsage('daclevels', 'Invalid range')
    for i in range(daclevels[0], daclevels[1]):
        if i not in ctx.obj['areastats'].index.levels[0]:
            click.echo(f'No data for daclevel = {i}, skipping . . .')
        else:
            click.echo(f'Generating plots for daclevel = {i} . . .')
            # DEBUG
            click.echo(ctx.obj['allstats'].columns.get_level_values(0).drop_duplicates())
            for stattype in ctx.obj['allstats'].columns.get_level_values(0).drop_duplicates():
                #a = ctx.obj['areastats'].loc[i]
                a = ctx.obj['allstats'].loc[i, stattype]
                fig, ax = plt.subplots(4, 2, figsize=(16, 16))
                ax[0][0].hist(a.loc[:, 'mean'], bins=30)
                ax[1][0].hist(a.loc[:, 'err'], bins=30)
                ax[2][0].hist(a.loc[:, 'std'], bins=30)
                ax[3][0].hist(a.loc[:, 'count'], bins=30)
                ax[0][1].plot(a.loc[:, 'mean'], linewidth=0.5)
                ax[1][1].plot(a.loc[:, 'err'], linewidth=0.5)
                ax[2][1].plot(a.loc[:, 'std'], linewidth=0.5)
                ax[3][1].plot(a.loc[:, 'count'], linewidth=0.5)
                fig.savefig(f'{out_prefix}{stattype}_{i}.png')
                plt.close()

@main.command()
@click.option('--daclevels', nargs=2, type=click.IntRange(1, 64), default=(1, 64))
@click.option('--channels', nargs=2, type=click.IntRange(0, 3072), required=True)
@click.option('--out-prefix', required=True)
@click.pass_context
def channel_pulserdac_info(ctx, daclevels, channels, out_prefix):
    if daclevels[0] >= daclevels[1]:
        raise click.BadOptionUsage('daclevels', 'Invalid range')
    if channels[0] >= channels[1]:
        raise click.BadOptionUsage('channels', 'Invalid range')
    for i in range(daclevels[0], daclevels[1]):
        if i not in ctx.obj['areastats'].index.levels[0]:
            click.echo(f'No data for daclevel = {i}, skipping . . .')
            continue
        for j in range(channels[0], channels[1]):
            if j not in ctx.obj['areastats'].index.levels[1]:
                click.echo(f'No data for channel = {j}, skipping . . .')
                continue
            a = ctx.obj['alldata'].loc[(i, j, slice(None)), 'Peak Area']
            click.echo(f'Generating plots for daclevel = {i}, channel {j} . . .')
            fig, ax = plt.subplots(2, figsize=(8, 8), layout='constrained')
            ax[0].hist(a, bins=20)
            ax[1].scatter(np.arange(len(a)), a, s=4)
            fig.suptitle(f'Distribution of pulse areas at Pulser DAC = {i}, channel {j}')
            ax[0].set(
                    title='Histogram of pulse areas',
                    xlabel='Pulse area (ADC Count * us)',
                    ylabel=f'Count (total = {ctx.obj["areastats"].at[(i, j), "count"]})'
                    )
            ax[1].set(
                    title='Pulse areas by ordinal position',
                    xlabel='Ordinal position (should be meaningless)',
                    ylabel='Pulse area (ADC Count * us)'
                    )
            fig.savefig(f'{out_prefix}{i}_{j}.png')
            plt.close()

@main.command()
@click.option('--channel', '-c', type=click.IntRange(0, 3072), required=True)
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--type', '-t', 'ytype', type=click.STRING, default='Peak Area')
@click.pass_context
def gain(ctx, channel, out, ytype):
    """Draws box plot of average pulse area for channels vs. pulser DAC level."""
    a = ctx.obj['allstats'].loc[:, (ytype, 'mean')].unstack('Pulser DAC')
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    ax.boxplot(a, positions=a.columns, whis=(0, 100))
    ax.set(
            title=f'Distribution of average pulse area over all channels at each Pulser DAC setting for {CRPNAME}\n'
                    '(boxes at quartiles, whiskers at full range)',
            xlabel='Pulser DAC setting',
            ylabel='Average pulse area (ADC counts * us)',
            #xticks=np.arange(0, 64),
            )
    fig.savefig(out)
    plt.close()

#@main.command()
#@click.option('--out', '-o', type=click.Path(), required=True)
#@click.pass_context
def calc_gain(ctx, out=None, ytype='Peak Area'):
    def fit(s):
        chnum = s.index.values[0][1]
        if chnum > 1903:
            #lastdaclevel = 60
            #firstdaclevel = 2
            #lastdaclevel = 60
            daclevels = DACLEVELS[1]
        else:
            #firstdaclevel = 2
            #lastdaclevel = 30
            daclevels = DACLEVELS[0]
        #s2 = s.loc[firstdaclevel:lastdaclevel]
        s2 = s.loc[daclevels]
        x = s2.index.get_level_values('Pulser DAC').to_numpy() * C
        y = s2.to_numpy()
        return scipy.stats.linregress(x, y, alternative='greater')
    #a = ctx.obj['areastats'].loc[:, 'mean']
    a = ctx.obj['allstats'].loc[:, (ytype, 'mean')]
    b = a.groupby(level='Channel', sort=False).agg(fit)
    c = {}
    for i in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr']:
        c[i] = b.apply(getattr, args=(i,))
    c['gain_e_per_ADC'] = 1e-15 / scipy.constants.e / c['slope']
    d = pd.DataFrame(c)
    click.echo(d)
    if out:
        d.to_pickle(out)
    return d

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--type', '-t', 'ytype', type=click.STRING, default='Real Amplitude')
@click.pass_context
def save_gain(ctx, out, ytype):
    reg = calc_gain(ctx, out=out, ytype=ytype)

names = ['Induction U', 'Induction V', 'Collection Z', 'Collection Z (y-range capped)']

@main.command()
@click.option('--out-prefix', '-o', type=click.Path(), required=True)
@click.option('--type', '-t', 'ytype', type=click.STRING, default='Peak Area')
#@click.option('--channel', '-c', type=int, required=True)
@click.argument('channels', nargs=-1, type=int)
@click.pass_context
def gain_plot_channel(ctx, out_prefix, channels, ytype):
    reg = calc_gain(ctx, None, ytype=ytype)
    for channel in channels:
        if channel in BAD_CHANNELS:
            click.echo(f'Channel {channel} is known bad channel! Skipping . . .')
            continue
        chfit = reg.loc[channel]
        click.echo(chfit)
        chas = ctx.obj['allstats'].loc[(slice(None), channel), (ytype, 'mean')]
        click.echo(chas)
        x = chas.index.get_level_values('Pulser DAC').to_numpy() * C
        y = chas.to_numpy()
        #x2 = np.arange(0, 64) * C
        x2 = np.arange(0, 64 if channel >= 1904 else 32) * C
        y2 = x2 * chfit.at['slope'] + chfit.at['intercept']
        residuals = np.empty(64 if channel >= 1904 else 32)
        residuals[0] = -y2[0]
        residuals[1:] = y - y2[1:]
        gain_e_per_ADC = chfit.at['gain_e_per_ADC']
        fig, ax = plt.subplots(2, figsize=(12, 16), layout='constrained')
        ax[0].scatter(x, y, s=10)
        ax[0].plot(x2, y2, linewidth=1)
        #xticks = np.arange(0, 64, 4)
        #ax[0].set_xticks(xticks, labels=ctx.obj['c'][xticks])
        ax[0].set(
                #xticks=np.arange(0, 64, 4),
                #yticks=np.arange(0, 45001, 5000),
                xlim=(0, 64 * C),
                #ylim=(0, 45000),
                )
        ax[0].grid()
        ax[0].text(20, ax[0].get_ylim()[1] * 0.9, f'Gain:         {chfit.at["slope"]:7.2f}\nIntercept:    {chfit.at["intercept"]:7.2f}\nGain (e/ADC): {gain_e_per_ADC:7.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
        ax[1].scatter(x2 * C, residuals, s=10)
        #ax[1].plot((0, 64), (0, 0), color='black', linewidth=1)
        ax[1].axhline(color='black', linewidth=1)
        #ax[1].set_xticks(xticks, labels=ctx.obj['c'][xticks])
        ax[1].set(
                #xticks=np.arange(0, 64, 4),
                xlim=(0, 64 * C),
                )
        fig.savefig(out_prefix + f'_{channel}.png')
        plt.close()

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--e-per-adc', is_flag=True)
@click.pass_context
def gain_residuals(ctx, out, e_per_adc):
    reg = calc_gain(ctx, None, ytype='Real Amplitude')
    intercepts = reg.loc[:, 'intercept'].to_numpy()
    slopes = reg.loc[:, 'slope'].to_numpy()
    x = np.arange(64)
    predicted = np.outer(x * C, slopes) + intercepts
    click.echo(predicted)
    actual_df = ctx.obj['allstats'].loc[:, ('Real Amplitude', 'mean')].unstack(level=0)
    click.echo(actual_df)
    actual_df.insert(0, 0, np.zeros(actual_df.index.size))
    click.echo(actual_df)
    actual = actual_df.to_numpy()
    click.echo(actual)
    if e_per_adc:   #this makes no sense
        residuals = 1e-15 / scipy.constants.e / actual - 1e-15 / scipy.constants.e / predicted.T
    else:
        residuals = actual - predicted.T
    click.echo(residuals)
    residuals_df = pd.DataFrame(residuals)
    residuals_df.index = actual_df.index
    residuals_df.columns = actual_df.columns
    print(residuals_df)
    residuals_df.to_pickle('fitresults/residuals_of_fit.pkl')
    #fig, ax = plt.subplots(4, figsize=(16, 48), layout='constrained')
    fig, ax = plt.subplots(ncols=3, figsize=(24, 8), layout='constrained')
    # NOTE!!! THESE NUMBERS WON'T WORK IF THERE ARE MISSING CHANNELS ON INDUCTION PLANES!!!
    ax[0].boxplot(residuals[:952, :32], positions=x[:32])
    ax[1].boxplot(residuals[952:1904, :32], positions=x[:32])
    ax[2].boxplot(residuals[1904:], positions=x)
    #ax[3].boxplot(residuals[1904:], positions=x)
    #ax[3].set(
    #        yticks=np.arange(-100, 101, 50),
    #        ylim=(-100, 100),
    #        )
    for ind, a in enumerate(ax):
        #a.grid(axis='y')
        a.axhline(color='black', linewidth=0.5)
        a.set(
                title=f'{names[ind]}\n',
                xlabel='Pulser DAC setting',
                yticks=np.arange(-100, 101, 50),
                ylim=(-100, 100),
                )
    ax[0].set_ylabel('Residuals from linear gain fit (ADC counts)')
    fig.suptitle('Residuals vs. Pulser DAC setting\n'
                 'Residuals of actual amplitude from predicted amplitude due to linear gain fit. '
                 'Gain fit from channels 2-30 for induction planes, 2-60 for collection plane (except 14, 32)\n'
                 #'Gain fit from channels 2-30 for induction planes (except 4, 5, 6), '
                 #'channels 2-60 for collection plane (except 20, 38, 41, 51).\n'
                 '(boxplot over channels. y-axis capped at (-100, 100) for visibility.)'
                )
    fig.savefig(out)
    plt.close()

def split_by_plane(df, level=1):
    """assumes channel number is in level=1 of index unless otherwise"""
    if level == 1:
        ind1 = df.loc[(slice(None), slice(0, 951))]
        ind2 = df.loc[(slice(None), slice(952, 1903))]
        col = df.loc[(slice(None), slice(1904, None))]
    elif level == 0:
        ind1 = df.loc[slice(0, 951)]
        ind2 = df.loc[slice(952, 1903)]
        col = df.loc[slice(1904, None)]
    return ind1, ind2, col

def tp_stats(ctx, diff=False):
    tp = ctx.obj['allstats'].loc[:, ('Peaking Time', 'mean')]
    print(tp)
    ind1, ind2, col = split_by_plane(tp)
    #tp_avg_inds = [i.loc[:31].drop([4, 5, 6], level='Pulser DAC').groupby(level='Channel', sort=False).mean() for i in (ind1, ind2)]
    tp_avg_inds = [i.loc[DACLEVELS[0]].groupby(level='Channel', sort=False).mean() for i in (ind1, ind2)]
    print(tp_avg_inds)
    tp_avg_col = col.loc[DACLEVELS[1]].groupby(level='Channel', sort=False).mean()
    print(tp_avg_col)
    tp_avgs = [tp_avg_inds[0], tp_avg_inds[1], tp_avg_col]
    if diff:
        tp_diffs = (tp_i.unstack(level=0).subtract(tp_avgs[i], axis='index') for i, tp_i in enumerate((ind1, ind2, col)))
        return tp_diffs
    else:
        return tp_avgs

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.pass_context
def save_tp_avgs(ctx, out):
    tp_avgs = tp_stats(ctx)
    tp_avgs_all = pd.concat(tp_avgs)
    click.echo(tp_avgs_all)
    tp_avgs_all.to_pickle(out)

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--type', '-t', 'ytype', type=click.STRING, default='Peak Area')
@click.pass_context
def peaking_time_vs_daclevel(ctx, out, ytype):
    tp_diffs = tp_stats(ctx, True)
    fig, ax = plt.subplots(ncols=3, figsize=(24, 8), layout='constrained')
    for i, tp_diff in enumerate(tp_diffs):
        if i < 2:
            ax[i].boxplot(tp_diff.loc[:, :31])
        else:
            ax[i].boxplot(tp_diff)
        ax[i].set(
                title=names[i],
                ylabel='Peaking time difference from average for channel (us)',
                )
        ax[i].axhline(color='black', linewidth=0.5)
    ax[2].set_xlabel('Pulser DAC setting')
    fig.suptitle('Peaking time difference from average for channel vs. Pulser DAC\n'
            'Peaking time average over Pulser DAC 1-31 for Induction, except 4, 5, 6\n'
            'Peaking time average over Pulser DAC 1-63 for Collection, except 38, 41, 51\n'
            'Boxplot over channels (3072 data points per Pulser DAC)')
    fig.savefig(out)
    plt.close()

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--type', '-t', 'ytype', type=click.STRING, default='Peak Area')
@click.pass_context
def gain_vs_peaking_time(ctx, out, ytype):
    reg = calc_gain(ctx, ytype=ytype)
    gains = split_by_plane(reg.loc[:, 'gain_e_per_ADC'], level=0)
    tp = ctx.obj['allstats'].loc[:, ('Peaking Time', 'mean')]
    print(tp)
    ind1, ind2, col = split_by_plane(tp)
    tp_avg_inds = [i.loc[:31].drop([4, 5, 6], level='Pulser DAC').groupby(level='Channel', sort=False).mean() for i in (ind1, ind2)]
    print(tp_avg_inds)
    tp_avg_col = col.drop([38, 41, 51], level='Pulser DAC').groupby(level='Channel', sort=False).mean()
    print(tp_avg_col)
    tp_avgs = [tp_avg_inds[0], tp_avg_inds[1], tp_avg_col]
    #fig, ax = plt.subplots(3, figsize=(12, 24), layout='constrained')
    fig, ax = plt.subplots(ncols=3, figsize=(18, 8), layout='constrained')
    for i in range(3):
        ax[i].scatter(tp_avgs[i], gains[i], s=5)
        ax[i].set(
                title=names[i],
                ylabel=f'Gain (e per ADC)',
                )
    ax[2].set_xlabel('Average peaking time (us)')
    fig.suptitle(f'Gain (from {ytype}) vs. average peaking time for each channel over Pulser DAC settings\n'
            'Peaking time average over Pulser DAC 1-31 for Induction, except 4, 5, 6\n'
            'Peaking time average over Pulser DAC 1-63 for Collection, except 38, 41, 51\n'
            'Scatterplot over channels (3072 data points per plot)')
    fig.savefig(out)
    plt.close()

def load_channel_map(filename):
    chmap = pd.read_csv(filename, sep='\t', header=0, index_col='offlchan')#['femb', 'asic', 'asicchan'])
    return chmap
RANGES = ((0, 952), (952, 1904), (1904, 3072))
TP_YLIM = (1.95, 2.25)
NBINS = 30

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.pass_context
def tp_distribution(ctx, out):
    """Draw plot of peaking time distribution (by plane) and save to out"""
    tp_avgs = tp_stats(ctx)
    fig, ax = plt.subplots(2, 3, figsize=(24, 12), layout='constrained')
    #bins = np.arange(*strs[4]['binsrange'])
    bins = np.linspace(*TP_YLIM, NBINS)
    fitx = np.linspace(*TP_YLIM, 100)
    means = [g.mean() for g in tp_avgs]
    stds = [g.std() for g in tp_avgs]
    counts = [g.count() for g in tp_avgs]
    errs = [std / np.sqrt(count) for std, count in zip(stds, counts)]
    stdpct = [std / mean * 100 for mean, std in zip(means, stds)]
    for i in range(3):
        ax[0][i].hist(tp_avgs[i].to_numpy(), bins=bins)
        ax[0][i].set(
                title=names[i],
                xlabel='Average Peaking Time (us)',
                )
        ax[0][i].plot(fitx, scipy.stats.norm.pdf(fitx, means[i], stds[i]) * counts[i] * (bins[1] - bins[0]))
        #ax[0][i].text(bins[-9], ax[0][i].get_ylim()[1] * 0.8, f'mean: {means[i]:6.4f}\nstd: {stds[i]:7.4f}\nerr: {errs[i]:7.4f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
        ax[0][i].text(bins[-9], ax[0][i].get_ylim()[1] * 0.8, f'mean: {means[i]:6.4f}\nstd: {stds[i]:7.4f}\nstd%: {stdpct[i]:6.4f}%', fontsize=15, fontfamily='monospace', verticalalignment='top')
        ax[1][i].plot(tp_avgs[i].index.to_numpy(), tp_avgs[i].to_numpy())
        ax[1][i].set(
                xlabel='Channel number',
                xlim=RANGES[i],
                yticks=np.linspace(*TP_YLIM, 7),
                ylim=TP_YLIM,
                )
    ax[0][0].set_ylabel('Count')
    ax[1][0].set_ylabel('Peaking time')
    fig.suptitle(f'{CRPNAME} Peaking Time Distribution by Plane')
    fig.savefig(out)
    plt.close()

def triple_plot(out, stats, binrange=None, statname="UNSPECIFIED", fit=True):
    """Draw histogram + channel plot, by plane, of some stat, and save to out"""
    fig, ax = plt.subplots(2, 3, figsize=(24, 12), layout='constrained')
    if binrange:
        bins = np.linspace(*binrange)
        binedges = bins
        fitx = np.linspace(*binrange[:2], 100)
    else:
        bins = 30
    means = [g.mean() for g in stats]
    stds = [g.std() for g in stats]
    counts = [g.count() for g in stats]
    errs = [std / np.sqrt(count) for std, count in zip(stds, counts)]
    stdpct = [std / mean * 100 for mean, std in zip(means, stds)]
    for i in range(3):
        _, binedges, _ = ax[0][i].hist(stats[i].to_numpy(), bins=bins)
        ax[0][i].set(
                title=names[i],
                xlabel=f'Average {statname}',
                )
        if not binrange:
            fitx = np.linspace(binedges[0], binedges[-1], 100)
        if fit:
            ax[0][i].plot(fitx, scipy.stats.norm.pdf(fitx, means[i], stds[i]) * counts[i] * (binedges[1] - binedges[0]))
            ax[0][i].text(binedges[-9], ax[0][i].get_ylim()[1] * 0.8, f'mean: {means[i]:7.2f}\nstd: {stds[i]:8.2f}\nstd%: {stdpct[i]:7.2f}%', fontsize=15, fontfamily='monospace', verticalalignment='top')
        ax[1][i].scatter(stats[i].index.to_numpy(), stats[i].to_numpy(), s=3)
        ax[1][i].axhline(color='black', linewidth=0.5)
        ax[1][i].set(
                xlabel='Channel number',
                xlim=RANGES[i],
                )
        if binrange:
            ax[1][i].set(
                #yticks=np.linspace(*binrange[:2], 7),
                yticks=binedges[::4],
                ylim=binrange[:2],
                )
    ax[0][0].set_ylabel('Count')
    ax[1][0].set_ylabel(statname)
    fig.suptitle(f'{CRPNAME} {statname} Distribution by Plane')
    fig.savefig(out)
    plt.close()

undershoot_params = {
        'Undershoot': {
            'binrange': None,
            },
        'Undershoot Position': {
            'binrange': None,
            },
        'Recovery Position': {
            'binrange': None,
            },
        'Undershoot Start': {
            'binrange': None,
            },
        'Undershoot Area': {
            'binrange': None,
            },
        }

@main.command()
@click.option('--out-prefix', '-o', 'out', type=click.Path(), required=True)
@click.pass_context
def undershoot_hists(ctx, out):
    #threshold = ctx.obj['allstats'].loc[:, ('Baseline Standard Deviation', 'mean')] * 4
    #mask = ctx.obj['allstats'].loc[:, ('Undershoot', 'mean')].abs() > threshold
    #mask = ctx.obj['allstats'].loc[:, ('Undershoot Position', 'mean')] > 12.0
    #a = (ctx.obj['alldata'].loc[:, 'Undershoot Start'] == 13.0).groupby(['Pulser DAC', 'Channel']).sum()
    #print(a)
    #mask = a > 150
    #allstats_masked = ctx.obj['allstats'].loc[mask]
    allstats_masked = ctx.obj['allstats']
    for j in range(1, 64):
        if j not in allstats_masked.index:
            continue
        for i, statname in enumerate(undershoot_params.keys()):
            params = undershoot_params[statname]
            #stats_full = allstats_masked.loc[j, (statname, 'mean')]
            #click.echo(stats_full)
            #stats = split_by_plane(stats_full, level=0)
            #triple_plot(f'{out}_{statname}_{j}.png', stats, binrange=params['binrange'], statname=statname)
            stats_full = allstats_masked.loc[j, (statname, 'mean')]
            click.echo(stats_full)
            stats = split_by_plane(stats_full, level=0)
            triple_plot(f'{out}_{statname}_mean_{j}.png', stats, binrange=params['binrange'], statname=statname)
        fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
        ax.scatter(allstats_masked.loc[j, ('Undershoot Position', 'mean')].to_numpy(), allstats_masked.loc[j, ('Undershoot', 'mean')].to_numpy(), s=3)
        ax.axhline(color='black', linewidth=0.5)
        ax.axvline(color='black', linewidth=0.5)
        fig.savefig(f'{out}_scatter_{j}.png')
        plt.close()

@main.command()
@click.option('--out-prefix', '-o', 'out', type=click.Path(), required=True)
@click.pass_context
def undershoot_analysis(ctx, out):
    datanames = ['Undershoot', 'Undershoot Position', 'Recovery Position', 'Undershoot Start']
    rdata = ctx.obj['alldata'].loc[:, datanames]
    startsfirst = rdata.loc[:, 'Undershoot Start'] == 13
    for i in range(10):
        a = rdata.loc[(50, i)]
        fig, ax = plt.subplots(ncols=4, figsize=(24, 8))
        for j in range(4):
            ax[j].hist(a.loc[:, datanames[j]])
            ax[j].set_title(datanames[j])
        fig.savefig(out + f'_{i}.png')
        plt.close()

@main.command()
@click.option('--out', '-o', 'out', type=click.Path(), required=True)
@click.pass_context
def undershoot_analysis_2(ctx, out):
    gain_amplitude = calc_gain(ctx, ytype="Real Amplitude").loc[:, 'slope']
    gain_undershoot = calc_gain(ctx, ytype="Undershoot").loc[:, 'slope']
    #gain_amplitude = calc_gain(ctx, ytype="Peak Area").loc[:, 'slope']
    #gain_undershoot = calc_gain(ctx, ytype="Undershoot Area").loc[:, 'slope']
    pct_undershoot = gain_undershoot / gain_amplitude * 100
    stats = split_by_plane(pct_undershoot, level=0)
    triple_plot(out, stats, statname='Undershoot %', binrange=(-8, 5, 53), fit=False)
    pct_undershoot.to_pickle(out[:-4] + '.pkl')
    click.echo(pct_undershoot.nlargest(20))
    click.echo(pct_undershoot.nsmallest(20))

gain_hist_strs = {
        'Peak Area': {
            2: {
                'ylabel': 'Gain (ADC count * us / pulserDAC setting',
                'ylim': (600, 740),
                },
            4: {
                'binsrange': (600, 740, 5),
                'ylim': (600, 740),
                },
            5: {
                'pedestal_offset': 600,
                'ylim': (600, 740),
                'yticks': (600, 741, 20),
                },
            },
        'Amplitude': {
            2: {
                'ylabel': 'Gain (ADC count / pulserDAC setting',
                'ylim': (2350, 2650),
                },
            4: {
                'binsrange': (2350, 2650, 10),
                'ylim': (2350, 2650),
                },
            5: {
                'pedestal_offset': 2350,
                'ylim': (2350, 2650),
                'yticks': (2350, 2651, 50),
                },
            },
        'Peaking Time': {
            2: {
                'ylabel': 'Gain (ADC count / pulserDAC setting',
                'ylim': (2350, 2650),
                },
            4: {
                'binsrange': (2350, 2650, 10),
                'ylim': (2350, 2650),
                },
            5: {
                'pedestal_offset': 2350,
                'ylim': (2350, 2650),
                'yticks': (2350, 2651, 50),
                },
            },
        'e_per_ADC': {
            2: {
                'ylabel': 'Gain (electrons / ADC count)',
                'ylim': (35.5, 40),
                },
            4: {
                'binsrange': (35.5, 40.1, 0.25),
                'ylim': (35.5, 40),
                },
            5: {
                'pedestal_offset': 20,
                'ylim': (35.5, 40),
                'yticks': (30, 41, 1),
                },
            },
    }

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.argument('channels', nargs=-1, type=int)
@click.pass_context
def plot_std_vs_pulserDAC(ctx, out, channels):
    """Plot standard deviation vs. pulser DAC setting for list of channels."""
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    for chnum in channels:
        y = ctx.obj['allstats'].loc[(slice(None), chnum), ('Real Amplitude', 'std')]
        x = y.index.get_level_values('Pulser DAC').to_numpy()
        ax.plot(x, y, linewidth=0.7, label=str(chnum))
    ax.set(
            title='Amplitude Standard Deviation vs. Pulser DAC Setting',
            xlabel='Pulser DAC Setting (base 10)',
            ylabel='Amplitude Standard Deviation (ADC Counts)',
            yticks=np.arange(0, ctx.obj['allstats'].loc[(slice(None), channels), ('Real Amplitude', 'std')].max() // 10 * 10 + 3, 10),
            xticks=np.arange(0, 65, 5),
            xlim=(0, 64),
            )
    ax.legend()
    fig.savefig(out)
    plt.close()

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--channel-map', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--type', '-t', 'ytype', type=click.STRING, default='Peak Area')
@click.pass_context
def gain_hist(ctx, out, channel_map, ytype):
    strs = gain_hist_strs.get(ytype, gain_hist_strs['Peak Area'])
    if ytype == 'e_per_ADC':
        ytype = 'Real Amplitude'
        fitp = 'gain_e_per_ADC'
    else:
        fitp = 'slope'
    chmap = load_channel_map(channel_map)
    chmap.sort_values(['femb', 'asic', 'asicchan'], inplace=True)
    print(chmap)
    pedestal_rms = pd.Series(np.loadtxt('ped_crp4/stds.tsv', delimiter='\t'))
    pedestal_rms.drop(BAD_CHANNELS, inplace=True)
    print(pedestal_rms)
    pedestal_rms_by_asic = pedestal_rms.reindex_like(chmap)
    print(pedestal_rms_by_asic)
    reg = calc_gain(ctx, ytype=ytype)
    gains_all = reg.loc[:, fitp]
    gains_ind = reg.loc[:1903, fitp]
    gains_col = reg.loc[1904:, fitp]
    gains_ind1 = reg.loc[:951, fitp]
    gains_ind2 = reg.loc[952:1903, fitp]
    print(gains_ind1.describe())
    print(gains_ind2.describe())
    print(gains_col.describe())
    gains_by_asic = reg.loc[:, fitp].reindex_like(chmap)
    print(gains_by_asic)
    index_by_asic = pd.MultiIndex.from_frame(chmap.loc[:, ['femb', 'asic', 'asicchan']])
    gains_by_asic_mi = gains_by_asic.set_axis(index_by_asic, copy=False)
    print(gains_by_asic_mi)
    stats_by_asic = gains_by_asic_mi.groupby(level=['femb', 'asic'], sort=False).agg(['mean', 'std'])
    print(stats_by_asic)
    largest_stds = stats_by_asic.nlargest(20, 'std')
    print(largest_stds)
    all_asics_by_largest = stats_by_asic.sort_values('std', ascending=False)
    all_asics_by_largest.to_csv('all_asics_by_largest.csv', sep='\t')
    print(stats_by_asic.loc[9])
    print(stats_by_asic.loc[21])
    print(stats_by_asic.loc[12])
    fig, ax = plt.subplots(2, 3, figsize=(24, 12), layout='constrained')
    ax[0][0].hist(gains_ind.to_numpy(), bins=40)
    ax[1][0].hist(gains_col.to_numpy(), bins=40)
    ax[0][1].plot(gains_ind.index.to_numpy(), gains_ind.to_numpy())
    ax[1][1].plot(gains_col.index.to_numpy(), gains_col.to_numpy())
    ax[0][2].hist(gains_by_asic.to_numpy(), bins=40)
    ax[1][2].plot(np.arange(len(gains_by_asic))[1024:1024 + 256], gains_by_asic.to_numpy()[1024:1024 + 256])
    ax[1][2].set_xticks(np.arange(1024, 1024+256, 16))
    ax[1][2].grid(axis='x')
    fig.savefig(out)
    plt.close()
    fig, ax = plt.subplots(figsize=(20, 8), layout='constrained')
    ax.plot(np.arange(len(gains_by_asic)), gains_by_asic.to_numpy(), linewidth=1.0)
    ax.set_xticks(np.arange(0, len(gains_by_asic) + 1, 128), labels=np.arange(1, 26))
    ax.set_xticks(np.arange(0, len(gains_by_asic) + 1, 16), minor=True)
    ax.grid(which='major', axis='x', color='black')
    ax.grid(which='minor', axis='x')
    ax.set(
            title=f'{CRPNAME} Gain by ASIC (gain from {ytype})',
            xlabel='FEMB # (grey line = 1 ASIC)',
            ylabel=strs[2]['ylabel'],
            xlim=(0, 3072),
            ylim=strs[2]['ylim'],
            )
    fig.savefig(out[:-4] + '_2' + '.png')
    plt.close()
    fig, ax = plt.subplots(ncols=3, figsize=(20, 6), layout='constrained')
    ax[0].hist(stats_by_asic.loc[:, 'mean'], bins=30)
    ax[1].hist(stats_by_asic.loc[:, 'std'], bins=30)
    ax[2].scatter(stats_by_asic.loc[:, 'mean'], stats_by_asic.loc[:, 'std'])
    fig.savefig(out[:-4] + '_3' + '.png')
    plt.close()
    fig, ax = plt.subplots(2, 3, figsize=(24, 12), layout='constrained')
    bins = np.arange(*strs[4]['binsrange'])
    fitx = np.linspace(bins[0], bins[-1], 100)
    means = [g.mean() for g in (gains_ind1, gains_ind2, gains_col)]
    stds = [g.std() for g in (gains_ind1, gains_ind2, gains_col)]
    counts = [g.count() for g in (gains_ind1, gains_ind2, gains_col)]
    errs = [std / np.sqrt(count) for std, count in zip(stds, counts)]
    stdpct = [std / mean * 100 for mean, std in zip(means, stds)]
    ax[0][0].hist(gains_ind1.to_numpy(), bins=bins)
    ax[0][0].set(
            title='Induction U',
            xlabel='Gain',
            ylabel='Count',
            )
    ax[0][1].hist(gains_ind2.to_numpy(), bins=bins)
    ax[0][1].set(
            title='Induction V',
            xlabel='Gain',
            )
    ax[0][2].hist(gains_col.to_numpy(), bins=bins)
    ax[0][2].set(
            title='Collection Z',
            xlabel='Gain',
            )
    for i in range(3):
        ax[0][i].plot(fitx, scipy.stats.norm.pdf(fitx, means[i], stds[i]) * counts[i] * strs[4]['binsrange'][2])
        #ax[0][i].text(bins[-9], ax[0][i].get_ylim()[1] * 0.8, f'mean: {means[i]:7.2f}\nstd: {stds[i]:8.2f}\nerr: {errs[i]:8.2f}', fontsize=15, fontfamily='monospace', verticalalignment='top')
        ax[0][i].text(bins[-6], ax[0][i].get_ylim()[1] * 0.8, f'mean: {means[i]:7.2f}\nstd: {stds[i]:8.2f}\nstd%: {stdpct[i]:7.2f}%', fontsize=15, fontfamily='monospace', verticalalignment='top')
    ax[1][0].plot(gains_ind1.index.to_numpy(), gains_ind1.to_numpy())
    ax[1][0].set(
            xlabel='Channel number',
            xlim=(0, 952),
            ylim=strs[4]['ylim'],
            ylabel='Gain',
            )
    ax[1][1].plot(gains_ind2.index.to_numpy(), gains_ind2.to_numpy())
    ax[1][1].set(
            xlabel='Channel number',
            xlim=(952, 1904),
            ylim=strs[4]['ylim'],
            )
    ax[1][2].plot(gains_col.index.to_numpy(), gains_col.to_numpy())
    ax[1][2].set(
            xlabel='Channel number',
            xlim=(1904, 3072),
            ylim=strs[4]['ylim'],
            )
    fig.suptitle(f'{CRPNAME} Gain Distribution by Plane (gain from {ytype})')
    fig.savefig(out[:-4] + '_4' + '.png')
    plt.close()
    fig, ax = plt.subplots(figsize=(120, 8), layout='constrained')
    ax.plot(np.arange(len(gains_by_asic)), gains_by_asic.to_numpy(), linewidth=1.0)
    #ax.plot(np.arange(len(pedestal_rms_by_asic)), pedestal_rms_by_asic.to_numpy() + strs[5]['pedestal_offset'], linewidth=1.0)
    ax.set_xticks(np.arange(0, len(gains_by_asic) + 1, 128), labels=np.arange(1, 26))
    ax.set_xticks(np.arange(0, len(gains_by_asic) + 1, 16), minor=True)
    ax.set_xlim((0, len(gains_by_asic)))
    ax.set_yticks(np.arange(*strs[5]['yticks']))
    ax.set_ylim(strs[5]['ylim'])
    ax.grid(which='major', axis='x', color='black')
    ax.grid(which='minor', axis='x')
    fig.savefig(out[:-4] + '_5' + '.png')
    plt.close()
    #fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    #ax.scatter(pedestal_rms[gains_all.index.to_numpy()], gains_all.to_numpy(), s=5)
    #fig.savefig(out[:-4] + '_6' + '.png')
    #plt.close()

if __name__ == '__main__':
    main(obj={})
