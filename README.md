# Python Companion to Emitter Detection and Geolocation for Electronic Warfare

<img src="graphics/cover_emitterDet.png" height=200 /><img src="graphics/cover_practicalGeo.png" height=200 />

This repository is a port of the [MATLAB software companion](https://github.com/nodonoughue/emitter-detection-book/) to *Emitter Detection and Geolocation for Electronic Warfare,* by Nicholas A. O'Donoughue, Artech House, 2019.

This repository contains the Python code, released under the MIT License, and when it is complete, it will generate all the figures and implements all the algorithms and many of the performance calculations within the texts *Emitter Detection and Geolocation for Electronic Warfare,* by Nicholas A. O'Donoughue, Artech House, 2019 and *Practical Geolocation for Electronic Warfare using MATLAB,* by Nicholas A. O'Donoughue, Artech House, 2022.

The textbooks can be purchased from Artech House directly at the following links: **[Emitter Detection and Geolocation for Electronic Warfare](https://us.artechhouse.com/Emitter-Detection-and-Geolocation-for-Electronic-Warfare-P2291.aspx)**, and **[Practical Geolocation for Electronic Warfare using MATLAB](https://us.artechhouse.com/Practical-Geolocation-for-Electronic-Warfare-Using-MATLAB-P2292.aspx)** Both are also available from Amazon.

## Installation

coming soon...

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
A number of utilities are provided in this repository, under the following namespaces:

+ **aoa/** Code to execute angle-of-arrival estimation, as discussed in Chapter 7
+ **array/** Code to execute array-based angle-of-arrival estimation, as discussed in Chapter 8
+ **atm/** Code to model atmospheric loss, as discussed in Appendix Carlo
+ **detector/** Code to model detection performance, as discussed in Chapter 3-4
+ **fdoa/** Code to execute Frequency Difference of Arrival (FDOA) geolocation processing, as discussed in Chapter 12.
+ **hybrid/** Code to execute hybrid geolocation processing, as discussed in Chapter 13.
+ **noise/** Code to model noise power, as discussed in Appendix D.
+ **prop/** Code to model propagation losses, as discussed in Appendix B.
+ **tdoa/** Code to execute Time Difference of Arrival (TDOA) geolocation processing, as discussed in Chapter 11.
+ **triang/** Code to model triangulation from multiple AOA measurements, as discussed in Chapter 10.
+ **utils/** Generic utilities, including numerical solvers used in geolocation algorithms.

## Feedback
Please submit any suggestions, bugs, or comments to nicholas [dot] odonoughue [at] ieee [dot] org.
