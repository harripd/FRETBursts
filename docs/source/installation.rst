.. _installation:

FRETBursts Installation
=======================

FRETBursts can be installed as a standard python package either via `conda`
or PIP (see below). Being written in python, FRETBursts runs on OS X,
Windows and Linux.

For updates on the latest FRETBursts version please refer to the
:doc:`Release Notes (What's new?) <releasenotes>`.

.. _package_install:

Installing latest stable version
--------------------------------

The preferred way to to install and keep FRETBursts updated is through
`conda`, a package manager used by Anaconda scientific python distribution.
If you haven't done it already, please install the python3. We recommend using `Anaconda <https://docs.anaconda.com/free/anaconda/install/>`_.

Then, you can install or upgrade FRETBursts with::

    conda install fretbursts -c conda-forge

After the installation, it is recommended that you download and run the
`FRETBursts notebooks <https://github.com/OpenSMFS/FRETBursts_notebooks/archive/master.zip>`__
to get familiar with the workflow. If you don't know what a Jupyter Notebooks is
and how to launch it please see:

* `Jupyter/IPython Notebook Quick Start Guide <http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/>`__

See also the FRETBursts documentation section: :ref:`running_fretbursts`.

Install from yaml file
----------------------

With anaconda, you can manage different environments, allowing specific versions to be installed ensuring compatibility.
Which packages you need will depend on your use case.

Environments can be build from yaml files with::

    conda env create -f <envfile.yml>

And activate with::

    conda activate <envname>

Below are environment files that we have *_**verified to work**
#. Minimal environment: :download:`frbmin.yml<downloads/frbmin.yml>` which will create an environment named `frbmin` which contains just the essential packages for running the notebooks
#. Complete environment :download:`frbcmpt.yml<downloads/frbcmplt.yml>` which will create an environment named `frbcmplt` which also includes cython, testing packages and numba, which are not necesary for running FRETBursts, but can come in helpful in other circumstances

For packages build off of FRETBursts, check their respective documentation for similar yaml files.

To create an environemnt from a downloaded yml file (like those above) run the command in your terminal::

    conda env create <filenam.yml>

Then simply activate the environment with::

    conda activate <envname>


Alternative methods: using PIP
------------------------------

Users that prefer using `PIP <https://pypi.python.org/pypi/pip>`__, have to
make sure that all the non-pure python dependencies are properly installed
(i.e. numpy, scipy, pandas, matplotlib, pyqt, pytables), then use the
usual::

    pip install fretbursts --upgrade

The previous command installs or upgrades FRETBursts to the latest stable release.


Install FRETBursts in a stand-alone environment
-----------------------------------------------

For reproducibiltity, it is better to install FRETBursts in a dedicated environment.
The instructions below create a new conda environment with python 3.7:

First, add the ``conda-forge`` channel
containing the fretbursts (do it only once after installing Anaconda)::

    conda config --append channels conda-forge

Then create a new conda environment with python 3.10 and FRETbursts::

    conda create -n py310-fb python=3.10 fretbursts
    conda activate py310-fb
    conda install pyqt   # optional
    pip install pybroom  # optional
    python -m ipykernel install --user --name py310-fb --display-name "Python 3.10 (FB)"

The last command installs the
`jupyter kernel <https://ipython.readthedocs.io/en/latest/install/kernel_install.html>`__
so that you can use the new environment from jupyter notebooks.

This method allows to easily backup and reinstall a working environment, or install
it on a different machine (with same OS). This is useful for replicating
an environment on multiple machine, for recovering from a broken anaconda
installation or for reproducibility of published results.

More info:

- `Using conda environments <https://conda.io/docs/using/envs.html>`__
- `Managing conda channels <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html>`__
- `Installing a jupyter kernel <https://ipython.readthedocs.io/en/latest/install/kernel_install.html>`__


.. _source_install:

Install latest development version
----------------------------------

As a rule, all new development takes place on separate "feature branches".
The master branch should always be stable and releasable.
The advantage of installing from the master branch is that you can
get updates without waiting for a formal release.
If there are some errors you can always roll back to the latest
released version to get your job done. Since you have the full version
down to the commit level printed in the notebook you will know which version
works and which does not.

You can install the latest development version directly from GitHub with::

    pip install git+git://github.com/OpenSMFS/FRETBursts.git

.. note ::
    Note that the previous command fails if `git <http://git-scm.com/>`__
    is not installed.

Alternatively you can do an "editable" installation, i.e. executing
FRETBursts from the source folder. In this case, modifications in the source
files are immediately available on the next FRETBursts import.
To do so, clone FRETBursts and install it as follows::

    git clone https://github.com/OpenSMFS/FRETBursts.git
    cd FRETBursts
    pip install -e .

It is recommended that you install `cython <http://cython.org/>`__ before
FRETBursts so that the optimized C routines are installed as well.
Also, make sure you have `lmfit` and `seaborn` installed before running
FRETBursts.
