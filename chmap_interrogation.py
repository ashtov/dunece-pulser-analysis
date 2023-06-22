import sys
from pprint import pprint

import detchannelmaps

CHANNEL_MAP = 'VDColdboxChannelMap'

ch_map = detchannelmaps.make_map(CHANNEL_MAP)

def main():
    if sys.argv[1] == 'offline':
        for i in range(int(sys.argv[2]), int(sys.argv[3])):
            chan = ch_map.get_crate_slot_fiber_chan_from_offline_channel(i)
            print(f'\nOffline channel: {i}')
            pprint([f'{j}: {getattr(chan, j)}' for j in dir(chan)[-4:]])
    else:
        print('offline?')

if __name__ == '__main__':
    main()
