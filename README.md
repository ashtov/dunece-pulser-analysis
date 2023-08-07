# ProtoDUNE II Coldbox Pulser Runs Analysis

The scripts in this repository can be used to analyse ProtoDUNE II Coldbox pulser run data, generating statistics on pulse amplitude, peaking time, etc. for each channel. It also finds parameters characterising the under/overshoot following each pulse. A variety of routines to generate informative plots from the analysed data are also included.

Currently only supports CRP coldbox data with LArASIC set to 14 mV/fC gain and 2 us shaping time. Using data with a different gain setting is not tested, but may work with some minor issues, such as incorrect axes on generated plots. Using data with a different shaping time will not work without modifications as many of the hardcoded fitting parameters assume a 2 us shaping time. APA support may be added in the future. Assumes 512 ns per sample.

## Requirements

- DUNE DAQ v4.1.0 Python venv
  - v4.0.0 and older will not work without modifications to the data processing code. Newer versions have not been tested.
  - Required only for `findpeaks.py`
- scipy
- pandas
- matplotlib (for plots)
- click

## How to use

To provide information to the script about the runs to be analysed and the pulser settings used for each run, an "inputs file" in a specific format must be created. The file must be a text file in the following format, with fields separated by TAB characters (including the header line as shown):

```
Run number  Filename    Pulser DAC
20269   pulser_runs/20269.hdf5  1
...
```

The script `genfilelistcalib.py` provides a template for generating such a file.

The main data-processing script is `findpeaks.py`. This script supports command-line arguments using click. A listing of arguments and their usage can be obtained with the `--help` option. The output of this script is a pickled pandas DataFrame with the filename specified by the `--output` argument. This DataFrame can be read with any python pandas installation using `pd.read_pickle()`. The DataFrame contains several pieces of information about each processed pulse. It has the following format:

- Column index: 3 levels (Pulser DAC, Channel, Pulse No.)
- Column names: 1 level (Parameter name)

The parameters found for each pulse are:
- Peak Position (in timestamps of the sample window)
- Peak Prominence (raw prominence of highest sample in pulse, in ADC counts)
- Amplitude (of response function fitted to peak. Divide by 10.11973588 to get amplitude in ADC counts (I do not know why the fit function is off by this factor))
- Peaking time (of response function fitted to peak, in microseconds)
- Horizontal offset (of response function relative to "first sample of peak", which is the first sample drawn in orange if the `--plot` option is passed)
- Baseline (average of 200 samples before peak, in ADC counts)
- Baseline Standard Deviation (standard deviation of above)
- Peak Area (integral of response function fitted to peak, in ADC counts * us)
- Undershoot (magnitude of under/overshoot following peak, in ADC counts relative to pre-peak baseline)
- Undershoot Position (index of most-deviating sample used for "Undershoot" relative to start of peak)
- Recovery Position (index of first sample after return to baseline following "Undershoot Position" relative to start of peak)
- Undershoot Start (index of sample at which under/overshoot was calculated to start relative to start of peak)
- Undershoot Area ("integral" (discrete sum) of deviation from baseline between Undershoot Start and Recovery Position)

Note that `findpeaks.py` may take a long time to run (~1 hour per Pulser DAC setting). It can be simply parallelised by running multiple instances with different values of the `--pulser-dac` option. The resulting output files can be easily combined using pandas; `combine_fitresults.py` is an example of such a combining script.

If raw signal plots are to be generated, it is highly recommended to use the `--firstonly` option as well to generate only one plot per channel and Pulser DAC setting. With `--plot` and `--firstonly` specified, runtime was found to be ~15 minutes per Pulser DAC setting.

## Generating plots

Once pulser runs have been initially processed using `findpeaks.py`, a variety of plots can be quickly generated using the various commands in `make_plots_new.py`. The first time this script is run, it will take a few seconds to generate statistics on each of the statistics in its input; this file can be saved using the `-o` option and on future runs loaded with the `-s` option, rather than being regenerated, to save time. Obviously, the same file should not be reused when analysing different datasets.

`make_plots_new.py` remains not fully polished; some values are hardcoded. An understanding of the plots it can generate can be gained by reading the code.

## Other files

All scripts other than `findpeaks.py` and `make_plots_new.py` were WIP or old versions that will likely not work without some modification.

## Author

Alexander Shtov (UC Berkeley / LBNL Neutrino group)

Inspired by previous pulser run analysis scripts by Hanjie Liu and Roger Huang.
