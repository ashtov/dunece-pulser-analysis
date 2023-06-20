import glob
DIRPREFIX = 'crp4_pulser_runs'
PEAKSFILEDIR = 'crp4_peaks'
FIRSTRUN = 20268
with open('filelistcalib.tsv', 'w') as f:
    f.write('Run number\tFilename\tPulser DAC\tPeaksfile\n')
    for i in range(64):
        fnames = glob.glob(f'{DIRPREFIX}/*{FIRSTRUN + i}*')
        assert len(fnames) == 1
        fname = fnames[0]
        f.write(f'{FIRSTRUN + i}\t{fname}\t{i}\t{PEAKSFILEDIR}/{FIRSTRUN + i}_peaks.tsv\n')
