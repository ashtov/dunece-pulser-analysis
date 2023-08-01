#!/usr/bin/env python

import glob
from pprint import pprint

import pandas as pd

files = glob.glob('fitresults/*bonus3.pkl')
#files.sort(key=lambda x: int(x[x.rfind('_') + 1:x.rfind('.')]))
files.sort(key=lambda x: int(x[x.find('_', x.rfind('/')) + 1:x.rfind('_')]))
pprint(files)

dfs = [pd.read_pickle(i) for i in files]
alldf = pd.concat(dfs)
print(alldf)
firstcol = alldf.iloc[:, 0]
print(firstcol)
is0 = firstcol != 0
print(is0)
alldf2 = alldf.loc[is0]
alldf2.rename_axis(['Pulser DAC', 'Channel', 'Pulse No.'], axis='index', inplace=True)
alldf2.index = alldf2.index.set_levels(alldf2.index.levels[1].astype('int16'), level=1)
print(alldf2)
alldf2.to_pickle('fitresults/all_bonus3.pkl')
