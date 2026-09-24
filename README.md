# Python Companion to Emitter Detection and Geolocation for Electronic Warfare

<img src="graphics/cover_emitterDet.png" height=200 /><img src="graphics/cover_practicalGeo.png" height=200 />

This repository is a port of the [MATLAB software companion](https://github.com/nodonoughue/emitter-detection-book/) to *Emitter Detection and Geolocation for Electronic Warfare,* by Nicholas A. O'Donoughue, Artech House, 2019.

This repository contains the Python code, released under the MIT License, and when it is complete, it will generate all the figures and implements all the algorithms and many of the performance calculations within the texts *Emitter Detection and Geolocation for Electronic Warfare,* by Nicholas A. O'Donoughue, Artech House, 2019 and *Practical Geolocation for Electronic Warfare using MATLAB,* by Nicholas A. O'Donoughue, Artech House, 2022.

The textbooks can be purchased from Artech House directly at the following links: **[Emitter Detection and Geolocation for Electronic Warfare](https://us.artechhouse.com/Emitter-Detection-and-Geolocation-for-Electronic-Warfare-P2291.aspx)**, and **[Practical Geolocation for Electronic Warfare using MATLAB](https://us.artechhouse.com/Practical-Geolocation-for-Electronic-Warfare-Using-MATLAB-P2292.aspx)** Both are also available from Amazon.

## Installation

### PyPI Install (recommended)
Use pip to install the package from the PyPI repository
```
pip install ewgeo
```

All the tools will be installed and available by importing the `ewgeo` package.
```
import ewgeo
```

### Local Install
After cloning or downloading the git repository, you can install it locally in any virtual environment.

If the path to your downloaded copy of the repository is `<PATH_TO_EWGEO>`, then issue the following commands in a terminal window.
```
cd <PATH_TO_EWGEO>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

This repository has been tested with Python 3.12 and 3.13. We recommend using a 
virtual environment for package/dependency handling (the virtual environment 
does not need to be named `.venv`, however).

### Dependencies

This repository is dependent on the following packages, and was written with Python 3.12.
+ matplotlib
+ numpy
+ scipy
+ seaborn

## Figures
The **make_figures/** folder contains the code to generate all the figures in the textbook. The subfolder **make_figures/practical_geo** generates figures for the second textbook.

To generate all figures, run the file **make_figures.py**. To run figures for an individual chapter, use a command such as the following:
```python
import make_figures
chap1_figs = make_figures.chapter1.make_all_figures()
```

## Examples
The **examples/** folder contains the code to execute each of the examples in the textbook. The subfolder **examples/practical_geo** has examples from the second textbook.

## Utilities
A number of utilities are provided in this repository, under the following modules:

+ **ewgeo.aoa** Code to execute angle-of-arrival estimation, as discussed in Chapter 7
+ **ewgeo.array_df** Code to execute array-based direction-finding and angle-of-arrival estimation, as discussed in Chapter 8
+ **ewgeo.atm** Code to model atmospheric loss, as discussed in Appendix C
+ **ewgeo.detector** Code to model detection performance, as discussed in Chapter 3-4
+ **ewgeo.fdoa** Code to execute Frequency Difference of Arrival (FDOA) geolocation processing, as discussed in Chapter 12.
+ **ewgeo.hybrid** Code to execute hybrid geolocation processing, as discussed in Chapter 13.
+ **ewgeo.noise** Code to model noise power, as discussed in Appendix D.
+ **ewgeo.prop** Code to model propagation losses, as discussed in Appendix B.
+ **ewgeo.tdoa** Code to execute Time Difference of Arrival (TDOA) geolocation processing, as discussed in Chapter 11.
+ **ewgeo.triang** Code to model triangulation from multiple AOA measurements, as discussed in Chapter 10.
+ **ewgeo.tracker** Code to track emitter position over time using a Kalman filter, as discussed in Chapter 9.
+ **ewgeo.utils** Generic utilities, including numerical solvers used in geolocation algorithms.

## Optional SNR-based Covariance Matrix Support
New in v1.2 is the option to automatically adjust the covariance matrix of a PSS, based on the position of a source.
This was implemented to supplement the default behavior of a single fixed covariance matrix in CRLB calculations, which fails to account for reductions in signal strength at long range.

There is a new utility method, `ewgeo.utils.snr.compute_snr_per_sensor()` that will accept a number of arguments and return the SNR for each sensor in a PSS.
This function is called internally; users may access it directly, or they can create a PSS object with SNR parameters, which will prompt that object to update its covariance matrix dynamically.
```python
pss = TDOAPassiveSurveillanceSystem(
    x_sensor, cov,
    has_snr_cov=True,
    erp_dbw=..., 
    mds_dbw=..., 
    freq_hz=...,   # activates SNR mode
    coord_system='enu', 
    enu_ref_lla=(lat, lon, alt))
```

The necessary parameters are:
+ `erp_dbw`: Effective Radiated Power [dBW] for the source. Inclusive of all transmitter gain and loss terms.
+ `mds_dbw`: Minimum detectable signal for the PSS receivers. Must be scalar. At this power level, signals will have SNR=0 dB. Inclusive of all noise, gain, and loss terms.
+ `freq_hz`: Center frequency of the source [Hz]
+ `coord_system`: Coordinate system in use for the PSS; must be either ECEF or ENU. Used to determine atmospheric loss (optional).
+ `enu_ref_lla`: LLA reference point for the local ENU coordinate system; used to determine atmospheric loss (optional).
If only some of the SNR parameters are provided at creation, the rest will need to be provided when either `compute_cov`, `compute_crlb`, or `compute_snr` are called.

```python
snr_overrides = dict(...)
pss.compute_cov(x_source, **snr_overrides)
```
## Feedback
Please submit any suggestions, bugs, or comments as issues in this git repository.
