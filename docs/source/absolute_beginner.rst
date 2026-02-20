.. _absolute_beginner:

Getting started for the absolute python beginner
================================================

Before running FRETBursts you need to install a python distribution that
includes the Jupyter/IPython Notebook application.

We recomend using Anaconda, which you can find instructions for downloading and installing here: `Anaconda installation instructions <https://docs.anaconda.com/free/anaconda/install/index.html>`_.

You can find a guide for jupyter notebooks here:

- |jupyter_quick_guide|

.. |jupyter_quick_guide| raw:: html

   <a href="http://jupyter-notebook-beginner-guide.readthedocs.org/" target="_blank">Jupyter/IPython Notebook Quick Start Guide</a>

Once you are able start Jupyter Notebook application and open
a notebook you can move to the next section.

Installing FRETBursts
---------------------

To install FRETBursts, make sure you close Jupyter Notebook, then
type the following commands in a terminal
(i.e. ``cmd`` on Windows or ``Terminal`` on OSX)::

    conda install fretbursts -c conda-forge

The installation should take a few seconds.
If you notice any error please report it by opening a new issue on the
`FRETBursts GitHub Issues <https://github.com/OpenSMFS/FRETBursts/issues>`_.

Alternatively create an environment from one of our yaml files where we have verified compatibility of all versions of the software: :download:`frbmin.yml <downloads/frbmin.yml>`

First download `frbmin.yml`

Then run the following in your terminal::

    conda env create -f frbmin.yml
    conda activate frbmin

.. note::
    You may need to replace frbmin.yml with the path to the file you downloaded.
    :ref:`instalation` provides other yaml files for more complete environments.

Running FRETBursts tutorial notebook
------------------------------------

Download the ZIP file of
`FRETBursts notebooks <https://github.com/OpenSMFS/FRETBursts_notebooks/archive/master.zip>`__
and extract it inside a folder accessible by the Jupyter Notebook App.

Next, in the new Jupyter Notebook Dashboard click on the folder containing
the FRETBursts notebooks.

For first time users, we recommend to start from the notebook:

- `FRETBursts - us-ALEX smFRET burst analysis <http://nbviewer.ipython.org/urls/raw.github.com/OpenSMFS/FRETBursts_notebooks/master/notebooks/FRETBursts%2520-%2520us-ALEX%2520smFRET%2520burst%2520analysis.ipynb>`__

and follow the instructions therein.

Remember, to run the notebooks step-by-step (one cell a time) keep pressing
*shift + enter*. To run the entire notebook in a single step click on menu
*Cell -> Run All*.

For more info how to run/edit a notebook see |run_notebook|.

.. |run_notebook| raw:: html

   <a href="http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/execute.html" target="_blank">Running the Jupyter Notebook</a>
