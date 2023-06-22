import glob
DIRPREFIX = 'crp5_pulser_runs'
PEAKSFILEDIR = 'crp5_peaks'
FIRSTRUN = 21010
for i in range(64):
    if i % 10 == 0:
        if i != 0:
            f.close()
        f = open(f'filelistcalib_crp5_part{i // 10}.tsv', 'w')
        f.write('Run number\tFilename\tPulser DAC\tPeaksfile\n')
    fnames = glob.glob(f'{DIRPREFIX}/*run0{FIRSTRUN + i}*')
    assert len(fnames) == 1
    fname = fnames[0]
    f.write(f'{FIRSTRUN + i}\t{fname}\t{i}\t{PEAKSFILEDIR}/{FIRSTRUN + i}_peaks.tsv\n')
