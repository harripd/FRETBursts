Overview
=======

**FRETBursts** is an open-source toolkit for analysis of single-molecule FRET
data acquired by single and multi-spot confocal systems.

FRETBursts aims to be a reference implementation for all the state-ot-the-art
algorithms commonly used in smFRET burst analysis.

As input data, both single laser excitation and 2-laser **Al**ternating **Ex**citation 
(ALEX) are supported. 

Several background estimation and FRET efficiency fitting routines are
implemented. The burst search is an efficient version of the classical
sliding-window search and can be applied al different selection of timestamps
(all, d-only, a-only, etc...). A large number of post-search burst selection 
criteria are included and ready to use (see `select_bursts.py`). Moreover, 
defining a new burst selection criterium requires only a couple of lines of code.

A variety of preset plotting functions are already defined. Just to mention a few:
time-traces, scatter-plots of any burst data (i.e. E vs S, E vs size, etc...),
histograms (FRET, stoichiometry, inter-photon waiting times, 2D ALEX histogram, etc..),
kernel density estimations and much more (see `burst_plot.py`).

FRETBursts can load one of the sample datasets (soon to be released) or any arbitrary
binary timestamps data. To load a new binary file format 

For bug reports please use the GitHub issue tracker. Also, fixes and/or enhancements 
are welcome: just send a [pull request (PR)](https://help.github.com/articles/using-pull-requests).

For more info contact me at tritemio @ gmail.com.

Environment
===========

FRETBursts is written in the [python programming language](http://www.python.org/) using the standard 
scientific stack of libraries (numpy, scipy, matplotlib).

Usage examples are provided as IPython notebooks. 
[IPython Notebook](http://ipython.org/notebook.html) is an interactive web-based environment that allows 
mixing rich text, math and graphics with (live) code, similarly to the Mathematica environment. 
You can find a static HTML version of the notebooks below in section **[Usage examples](#usage-examples)**. 

For a tutorial on using python for scientific computing:

* [Python Scientific Lecture Notes](http://scipy-lectures.github.io/)

Another useful resources for the IPython Notebook:

* [The IPython Notebook](http://ipython.org/ipython-doc/stable/interactive/notebook.html)
* [Notebook examples](http://nbviewer.ipython.org/github/ipython/ipython/blob/master/examples/Notebook/Index.ipynb)
* [A gallery of interesting IPython Notebooks](https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks)

#Installation

##MS Windows

In order to run the code you need to install a scientific python
distribution like [Anaconda](https://store.continuum.io/cshop/anaconda/).
The free version of Anaconda includes all the needed dependencies.
Any other scientific python distribution (for example 
[Enthought Canopy](https://www.enthought.com/products/canopy/)) 
will work as well.
 
Once a python distribution is installed, download the latest version
of [FRETBursts](https://github.com/tritemio/FRETBursts) from *GitHub*. 

The most user friendly way to use FRETBursts is through an IPython Notebook. 
The following paragraph shows how to configure it.

###Configuring IPython Notebook

We can create a launcher to start the IPython Notebook server on any local folder.

It is suggested to create a folder with all the notebooks and to put there a subfolder
for the FRETBursts notebooks.

You can
right click on the *IPython Notebook icon* -> *Properties* and paste 
the notebook folder in *Start in*. Apply and close.

Now, double click on the icon and a browser should pop up showing the list
of notebooks. Chrome browser is suggested.

##Linux and Mac OS X

On Linux or Mac OS X you can also use the [Anaconda](https://store.continuum.io/cshop/anaconda/) distribution.

Alternatively, these are the software dependencies (hint: on Mac OS X you can use MacPorts):

 - python 2.7.x
 - IPython 1.x (2.x suggested)
 - matplotlib 1.3.x or greater
 - numpy/scipy (any version from 2013 on)
 - cython (optional, to speedup burst search)
 - pytables 3.x (optional)
 - a modern browser (Chrome suggested)

#Usage examples

The following links will open (a static version of) the notebooks provided
with FRETBursts. These notebooks shows typical workflows for smFRET analysis
and illustrate some of the basic features of FRETBursts.

* [usALEX - Workflow](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/FRETBursts/master/notebooks/usALEX%2520-%2520Workflow.ipynb)

FRETBursts is a standard python packaged and therefore can be also used as a library and integrated in other software.

#Acknowledgements

This work was supported by NIH grants R01 GM069709 and R01 GM095904.

#License and Copyrights

FRETBursts - A bursts analysis toolkit for single and multi-spot smFRET data.

Copyright (C) 2014  Antonino Ingargiola - <tritemio @ gmail.com>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    version 2, as published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You can find a full copy of the license in the file LICENSE.txt
