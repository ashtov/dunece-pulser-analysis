import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

@click.group(invoke_without_command=True)
@click.argument('filename', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def main(ctx, filename):
    ctx.ensure_object(dict)
    alldata = pd.read_pickle(filename)
    ctx.obj['alldata'] = alldata
    click.echo(alldata)
    grouped = alldata.loc[:, 'Peak Area'].groupby(level=['Pulser DAC', 'Channel'])
    areastats = grouped.agg(['mean', 'std', 'count'])
    areastats['err'] = areastats.loc[:, 'std'] / np.sqrt(areastats.loc[:, 'count'])
    print(areastats)
    ctx.obj['areastats'] = areastats

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

if __name__ == '__main__':
    main(obj={})
