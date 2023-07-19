#!/usr/bin/env python

RANGE = (11, 64)
STEP = 4

for i in range(RANGE[0], RANGE[1], STEP):
    with open(f'_config_multirun_{i}.tsv', 'w') as f:
        f.write('Parameter\tValue\tComment\n'
                'parametersfname\tfilelistcalib_test2.tsv\t\n'
                'bad_channels\t1958, 2250, 2868\t\n'
                'fitplotsdir\tfitplots_all/\t\n'
                'negative\tno\t\n'
               f'alldatafname\tfitresults/alldata_{i}\t\n'
               f'pulserDACs\t{str([j for j in range(i, min(i + STEP, RANGE[1]))])[1:-1]}\t\n')
