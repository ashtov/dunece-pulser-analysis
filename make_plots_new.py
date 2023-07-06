import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy

@click.group(invoke_without_command=True)
@click.argument('filename', type=click.Path(exists=True, dir_okay=False))
@click.option('--statsfile', '-s', type=click.Path(exists=True, dir_okay=False))
@click.option('--statsout', '-o', type=click.Path(writable=True))
@click.pass_context
def main(ctx, filename, statsfile, statsout):
    ctx.ensure_object(dict)
    alldata = pd.read_pickle(filename)
    ctx.obj['alldata'] = alldata
    click.echo(alldata)
    if statsfile:
        ctx.obj['areastats'] = pd.read_pickle(statsfile)
    else:
        grouped = alldata.loc[:, 'Peak Area'].groupby(level=['Pulser DAC', 'Channel'], sort=False)
        areastats = grouped.agg(['mean', 'std', 'count'])
        areastats['err'] = areastats.loc[:, 'std'] / np.sqrt(areastats.loc[:, 'count'])
        ctx.obj['areastats'] = areastats
    click.echo(ctx.obj['areastats'])
    if statsout:
        ctx.obj['areastats'].to_pickle(statsout)

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
            a = ctx.obj['areastats'].loc[i]
            click.echo(f'Generating plots for daclevel = {i} . . .')
            fig, ax = plt.subplots(4, 2, figsize=(16, 16))
            ax[0][0].hist(a.loc[:, 'mean'], bins=30)
            ax[1][0].hist(a.loc[:, 'err'], bins=30)
            ax[2][0].hist(a.loc[:, 'std'], bins=30)
            ax[3][0].hist(a.loc[:, 'count'], bins=30)
            ax[0][1].plot(a.loc[:, 'mean'], linewidth=0.5)
            ax[1][1].plot(a.loc[:, 'err'], linewidth=0.5)
            ax[2][1].plot(a.loc[:, 'std'], linewidth=0.5)
            ax[3][1].plot(a.loc[:, 'count'], linewidth=0.5)
            fig.savefig(f'{out_prefix}{i}.png')
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
@click.pass_context
def gain(ctx, channel, out):
    """Draws box plot of average pulse area for channels vs. pulser DAC level."""
    a = ctx.obj['areastats'].loc[:, 'mean'].unstack('Pulser DAC')
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    ax.boxplot(a, positions=a.columns, whis=(0, 100))
    ax.set(
            title='Distribution of average pulse area over all channels at each Pulser DAC setting for CRP4\n'
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
def calc_gain(ctx, out):
    def fit(s):
        chnum = s.index.values[0][1]
        if chnum > 1903:
            lastdaclevel = 60
        else:
            lastdaclevel = 30
        s2 = s.loc[:lastdaclevel]
        x = s2.index.get_level_values('Pulser DAC').to_numpy()
        y = s2.to_numpy()
        return scipy.stats.linregress(x, y, alternative='greater')
    a = ctx.obj['areastats'].loc[:, 'mean']
    b = a.groupby(level='Channel', sort=False).agg(fit)
    c = {}
    for i in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr']:
        c[i] = b.apply(getattr, args=(i,))
    d = pd.DataFrame(c)
    click.echo(d)
    if out:
        d.to_pickle(out)
    return d

def load_channel_map(filename):
    chmap = pd.read_csv(filename, sep='\t', header=0, index_col='offlchan')#['femb', 'asic', 'asicchan'])
    return chmap

@main.command()
@click.option('--out', '-o', type=click.Path(), required=True)
@click.option('--channel-map', type=click.Path(exists=True, dir_okay=False), required=True)
@click.pass_context
def gain_hist(ctx, out, channel_map):
    chmap = load_channel_map(channel_map)
    chmap.sort_values(['femb', 'asic', 'asicchan'], inplace=True)
    print(chmap)
    pedestal_rms = pd.Series(np.loadtxt('ped_crp4/stds.tsv', delimiter='\t'))
    pedestal_rms.drop([1958, 2250, 2868], inplace=True)
    print(pedestal_rms)
    pedestal_rms_by_asic = pedestal_rms.reindex_like(chmap)
    print(pedestal_rms_by_asic)
    reg = calc_gain(ctx, None)
    gains_all = reg.loc[:, 'slope']
    gains_ind = reg.loc[:1903, 'slope']
    gains_col = reg.loc[1904:, 'slope']
    gains_ind1 = reg.loc[:951, 'slope']
    gains_ind2 = reg.loc[952:1903, 'slope']
    print(gains_ind1.describe())
    print(gains_ind2.describe())
    print(gains_col.describe())
    gains_by_asic = reg.loc[:, 'slope'].reindex_like(chmap)
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
    fig.savefig(out[:-4] + '_2' + '.png')
    plt.close()
    fig, ax = plt.subplots(ncols=3, figsize=(20, 6), layout='constrained')
    ax[0].hist(stats_by_asic.loc[:, 'mean'], bins=30)
    ax[1].hist(stats_by_asic.loc[:, 'std'], bins=30)
    ax[2].scatter(stats_by_asic.loc[:, 'mean'], stats_by_asic.loc[:, 'std'])
    fig.savefig(out[:-4] + '_3' + '.png')
    plt.close()
    fig, ax = plt.subplots(2, 3, figsize=(24, 12), layout='constrained')
    ax[0][0].hist(gains_ind1.to_numpy(), bins=np.arange(600, 740, 5))
    ax[0][1].hist(gains_ind2.to_numpy(), bins=np.arange(600, 740, 5))
    ax[0][2].hist(gains_col.to_numpy(), bins=np.arange(600, 740, 5))
    ax[1][0].plot(gains_ind1.index.to_numpy(), gains_ind1.to_numpy())
    ax[1][1].plot(gains_ind2.index.to_numpy(), gains_ind2.to_numpy())
    ax[1][2].plot(gains_col.index.to_numpy(), gains_col.to_numpy())
    ax[1][0].set_ylim((600, 740))
    ax[1][1].set_ylim((600, 740))
    ax[1][2].set_ylim((600, 740))
    fig.savefig(out[:-4] + '_4' + '.png')
    plt.close()
    fig, ax = plt.subplots(figsize=(120, 8), layout='constrained')
    ax.plot(np.arange(len(gains_by_asic)), gains_by_asic.to_numpy(), linewidth=1.0)
    ax.plot(np.arange(len(pedestal_rms_by_asic)), pedestal_rms_by_asic.to_numpy() + 600, linewidth=1.0)
    ax.set_xticks(np.arange(0, len(gains_by_asic) + 1, 128), labels=np.arange(1, 26))
    ax.set_xticks(np.arange(0, len(gains_by_asic) + 1, 16), minor=True)
    ax.grid(which='major', axis='x', color='black')
    ax.grid(which='minor', axis='x')
    fig.savefig(out[:-4] + '_5' + '.png')
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    ax.scatter(pedestal_rms[gains_all.index.to_numpy()], gains_all.to_numpy(), s=5)
    fig.savefig(out[:-4] + '_6' + '.png')
    plt.close()

if __name__ == '__main__':
    main(obj={})
