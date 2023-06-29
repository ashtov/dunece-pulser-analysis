import logging
import sys
import pathlib

import numpy as np
import pandas as pd

from hdf5libs import HDF5RawDataFile
import detchannelmaps
import daqdataformats
import fddetdataformats
import rawdatautils.unpack.wib2

import findpeaks

def main():
    findpeaks.set_loglevel()
    parametersfname = pathlib.Path(sys.argv[1])
    if not parametersfname.exists():
        print(f'No such file: {parametersfname}')
        return 0
    parameters = findpeaks.read_parameters(parametersfname)

    # run number with minimal pulser DAC setting
    min_run_number = parameters.loc[slice(None), 'Pulser DAC'].idxmin()
    min_DAC_setting = parameters.at[min_run_number, 'Pulser DAC']
    if min_DAC_setting != 0:
        logging.warning('No run with DAC setting 0 in parameters file! Minimal found is %d, run #%d', min_DAC_setting, min_run_number)

    # really should put this stuff into a function somehow
    # very bad copy paste from findpeaks.py
    fname = parameters.at[min_run_number, 'Filename']
    logging.info('Reading file %s . . .', fname)
    h5_file = HDF5RawDataFile(fname)
    records = h5_file.get_all_record_ids()
    # below line differs from coypaste
    # alldata is array with dimension (channel, record, timestamp of signal)
    alldata = np.empty((findpeaks.CHANNELS_PER_CRP, len(records), findpeaks.TIMESTAMPS_PER_FRAME), dtype=np.int16)
    for rec_ind, rec in enumerate(records): # enumerate differs from copypaste
        logging.info('* Record: %s', str(rec))
        assert rec[1] == 0
        # One "fragment" is either data and metadata from a single WIB or some
        # other metadata about the event.
        fragpaths: list[str] = h5_file.get_fragment_dataset_paths(rec)
        for fragpath in fragpaths:
            logging.debug('* * Fragment: %s', fragpath)
            frag = h5_file.get_frag(fragpath)
            fragheader = frag.get_header()
            assert fragheader.version == 5
            # Only attempt to process WIB data, not metadata fragments
            logging.debug('* * * FragmentType: %s', fragheader.fragment_type)
            if fragheader.fragment_type == daqdataformats.FragmentType.kWIB.value:
                # WIB2Frame object stores data and another level of metadata
                # for a single WIB
                frame = fddetdataformats.WIB2Frame(frag.get_data())
                frameheader = frame.get_header()
                assert frameheader.version == 4
                ## differs from copypaste from here
                #alldata[rec_ind] = rawdatautils.unpack.wib2.np_array_adc(frag).T
                data = rawdatautils.unpack.wib2.np_array_adc(frag).T
                # don't need assert since above will just fail otherwise?
                #it's not true sometimes in CRP5???
                #assert data.shape == (256, findpeaks.TIMESTAMPS_PER_FRAME)
                # below 3 lines from copypaste though
                chnums = np.empty(256, dtype=np.int16)
                for i in range(256):
                    chnums[i] = findpeaks.CHANNEL_MAP.get_offline_channel_from_crate_slot_fiber_chan(frameheader.crate, frameheader.slot, frameheader.link, i)
                    #logging.debug('* * * * Channel number: %d', chnum)
                #alldata[chnums, rec_ind] = data
                alldata[chnums, rec_ind] = data[:, :8192]  # again CRP5 being wacky
    alldata_2d = alldata.reshape((alldata.shape[0], alldata.shape[1] * alldata.shape[2]))
    logging.info('Running calculations . . .')
    pedestals = np.mean(alldata_2d, axis=1)
    stds = np.std(alldata_2d, axis=1)
    rmss = np.sqrt(np.mean(alldata_2d.astype(np.int64)**2, axis=1))
    logging.info('Saving ped/pedestals.tsv')
    np.savetxt('ped/pedestals.tsv', pedestals, fmt='%.2f', delimiter='\t')
    logging.info('Saving ped/stds.tsv')
    np.savetxt('ped/stds.tsv', stds, fmt='%.2f', delimiter='\t')
    logging.info('Saving ped/rmss.tsv')
    np.savetxt('ped/rmss.tsv', rmss, fmt='%.2f', delimiter='\t')
    logging.info('Saving ped/alldata_2d.npy')
    np.save('ped/alldata_2d.npy', alldata_2d)

if __name__ == '__main__':
    main()
