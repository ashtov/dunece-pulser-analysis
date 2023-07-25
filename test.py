import sys
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
import scipy
from hdf5libs import HDF5RawDataFile
import detchannelmaps
import daqdataformats
import detdataformats
import fddetdataformats
import rawdatautils.unpack.wib2

from findpeaks import read_parameters

# sys.argv:
# 1: rec (index from 0)
# 2: fragment number
# 3: channel number
# 4: start record number
# 5: end record number
# 6: pulser DAC

#FNAME = 'np02_bde_coldbox_run020331_0000_dataflow0_datawriter_0_20230309T180027.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020268_0000_dataflow0_datawriter_0_20230309T161417.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020269_0000_dataflow0_datawriter_0_20230309T161601.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020275_0000_dataflow0_datawriter_0_20230309T162604.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020289_0000_dataflow0_datawriter_0_20230309T164942.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020291_0000_dataflow0_datawriter_0_20230309T165303.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020318_0000_dataflow0_datawriter_0_20230309T173834.hdf5'
#FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020326_0000_dataflow0_datawriter_0_20230309T175201.hdf5'

CHANNEL_MAP = 'VDColdboxChannelMap'

ch_map = detchannelmaps.make_map(CHANNEL_MAP)

parametersfname = 'filelistcalib_crp5_high.tsv'
parameters = read_parameters(parametersfname)
if len(sys.argv) > 6:
    pulserDAC = int(sys.argv[6])
    #FNAME = parameters.at[pulserDAC + 20268, 'Filename']
    #FNAME = parameters.at[pulserDAC + 21010, 'Filename']
    FNAME = parameters.at[pulserDAC + 21082, 'Filename']
else:
    print('probably should give all parameters')
    FNAME = 'crp4_pulser_runs/np02_bde_coldbox_run020291_0000_dataflow0_datawriter_0_20230309T165303.hdf5'

print(FNAME)
h5_file = HDF5RawDataFile(FNAME)
print(h5_file)
dp = h5_file.get_all_record_ids()
print(dp)
rec = dp[int(sys.argv[1])]
print(rec)
gids = h5_file.get_geo_ids(rec)
print(gids)
gids2 = h5_file.get_geo_ids_for_subdetector(rec, detdataformats.DetID.Subdetector.kVD_BottomTPC)
print(gids2)
fragpaths = h5_file.get_fragment_dataset_paths(rec)
print(fragpaths)
sids = h5_file.get_source_ids(rec)
print(sids)
fragnum = int(sys.argv[2])
frag = h5_file.get_frag(fragpaths[fragnum])
print(frag)
h = frag.get_header()
print(h)
#pprint([f'{i}: {getattr(h, i)}' for i in dir(h)])
d = frag.get_data()
print(d)
frame = fddetdataformats.WIB2Frame(d)
print(frame)
frameheader = frame.get_header()
print(frameheader)
pprint([f'{i}: {getattr(frameheader, i)}' for i in dir(frameheader)])
chnums = [ch_map.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, i) for i in range(256)]
print(chnums)
adcdata = rawdatautils.unpack.wib2.np_array_adc(frag)
print(adcdata)
print(adcdata.shape)
timestamp = rawdatautils.unpack.wib2.np_array_timestamp(frag)
print(timestamp)

#with open('fragtypes.txt', 'w') as f:
#    for i in fragpaths:
#        frag = h5_file.get_frag(i)
#        header = frag.get_header()
#        f.write('------------------------------\n\n')
#        f.write(i)
#        f.write('\n\n')
#        for j in dir(header):
#            f.write(f'{j}: {getattr(header, j)}\n')
#        f.write('\n\n')

channelnum = int(sys.argv[3])
MYCHAN = ch_map.get_crate_slot_fiber_chan_from_offline_channel(channelnum)
pprint([f'{i}: {getattr(MYCHAN, i)}' for i in dir(MYCHAN)])
#fragnum = 0
while not (MYCHAN.crate == frameheader.crate and MYCHAN.slot == frameheader.slot and MYCHAN.fiber == frameheader.link):
    fragnum += 1
    print(f'Wrong fragment! Trying fragment {fragnum} . . .')
    frag = h5_file.get_frag(fragpaths[fragnum])
    frame = fddetdataformats.WIB2Frame(frag.get_data())
    frameheader = frame.get_header()
adcdata = rawdatautils.unpack.wib2.np_array_adc(frag)
#assert MYCHAN.crate == frameheader.crate and MYCHAN.slot == frameheader.slot and MYCHAN.fiber == frameheader.link
x = timestamp - timestamp[0]
y = adcdata[:, MYCHAN.channel]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, linewidth=0.3)
fig.savefig('plot2.png')
plt.close()

#peaks, peaksinfo = scipy.signal.find_peaks(y, height=np.max(y) - 100, prominence=(None, None), wlen=20)#width=(None, 100))
# change prominence limit when necessary, maybe
peaks, peaksinfo = scipy.signal.find_peaks(y, prominence=(500, None), wlen=20, width=(None, None), height=(None, None))
print(peaks)
print(peaksinfo)

fig, ax = plt.subplots(ncols=len(peaks), figsize=(6 * len(peaks), 6), layout='constrained')
for i in range(len(peaks)):
    r = slice(peaks[i] - 20, peaks[i] + 20)
    #ax.plot(x[r], y[r], linewidth=0.3)
    x2 = np.arange(peaks[i] - 20, peaks[i] + 20)
    ax[i].scatter(x2, y[r], s=3)
    ax[i].set_xlabel(f'peak: {i}, prominence: {peaksinfo["prominences"][i]}')
    #ax[i].set_ylim((800, 1150))
fig.savefig(f'plot3.png')
plt.close()

if len(sys.argv) > 5:
    jstart = int(sys.argv[4])
    jend = int(sys.argv[5])
    for j in range(jstart, jend):
        rec = dp[j]
        print(f'\nrec: {rec}')
        fragpaths = h5_file.get_fragment_dataset_paths(rec)
        frag = h5_file.get_frag(fragpaths[fragnum])
        frame = fddetdataformats.WIB2Frame(frag.get_data())
        frameheader = frame.get_header()
        pprint([f'{i}: {getattr(frameheader, i)}' for i in ('crate', 'slot', 'link')])
        adcdata = rawdatautils.unpack.wib2.np_array_adc(frag)
        y = adcdata[:, MYCHAN.channel]
        fig, ax = plt.subplots(figsize=(32, 8), layout='constrained')
        ax.plot(y, linewidth=0.3)
        fig.savefig(f'rawsignalplots_crp5_high/signal_{pulserDAC}_{channelnum}_{j}.png')
        plt.close()
        peaks, peaksinfo = scipy.signal.find_peaks(y, prominence=(500, None), wlen=20, width=(None, None), height=(None, None))
        print(peaks)
        print(peaksinfo)
        fig, ax = plt.subplots(ncols=len(peaks), figsize=(6 * len(peaks), 6), layout='constrained')
        for k in range(len(peaks)):
            indstart = max(peaks[k] - 20, 0)
            indend = min(peaks[k] + 20, adcdata.shape[0])
            #r = slice(peaks[k] - 20, peaks[k] + 20)
            x2 = np.arange(indstart, indend)
            #print(f'Lengths: x2: {len(x2)}, y[r]: {len(y[r])}')
            ax[k].scatter(x2, y[indstart:indend], s=10)
            ax[k].set_xlabel(f'peak: {k}, prominence: {peaksinfo["prominences"][k]}')
            ax[k].set_yticks(np.arange(0, 16385, 2048))
            ax[k].set_yticks(np.arange(0, 16385, 512), minor=True)
            #ax[k].set_yticks(np.arange(0, 4097, 512))
            #ax[k].set_yticks(np.arange(0, 4097, 128), minor=True)
            ax[k].set_xticks(np.arange(x2[0], x2[-1] + 1, 5))
            ax[k].set_xticks(np.arange(x2[0], x2[-1] + 1), minor=True)
            ax[k].grid(which='major', linewidth=1)
            ax[k].grid(which='minor', linewidth=0.3)
            #ax[k].set_ylim(0, 16384)
            #ax[k].set_ylim((800, 1150))
        fig.savefig(f'rawsignalplots_crp5_high/peaks_{pulserDAC}_{channelnum}_{j}.png')
        plt.close()


#gid = h5_file.get_geo_ids(rec, daqdataformats.GeoID.SystemType.kTPC)
#print(gid)
#rhdp = h5_file.get_trigger_record_header_dataset_paths()
#print(rhdp)
#fdp = h5_file.get_all_fragment_dataset_paths()
#print(fdp)
#tsids = h5_file.get_all_timeslice_ids()
#print(tsids)
#trids = h5_file.get_all_trigger_record_ids()
#print(trids)
