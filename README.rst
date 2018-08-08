Imaging Spectrometer Optimal FITting (ISOFIT) Overview
======================================================

Test

This codebase contains a set of routines and utilities for fitting surface,
atmosphere and instrument models to imaging spectrometer data.  It is
written primarily in Python, with JSON format configuration files and some
dependencies on widely-available numerical and scientific libraries such as
scipy, scikit-learn, and numba.  It is designed for maximum flexibility, so
that users can swap in and evaluate model components based on different
radiative transfer models (RTMs) and various statistical descriptions of
surface, instrument, and atmosphere.  It can run on individual radiance
spectra in text format, or imaging spectrometer data cubes.

The subdirectories contain:

* data/      - shared data files
* docs/      - documentation
* examples/  - example runs packaged with input data and configuration files
* isofit/    - the main Python codebase, with the top-level program isofit.py
* utils/     - general purpose utilities and routines for the advanced user
* tests/     - unit tests for use by pytest

If you use ISOFIT in your research or production, we ask that you cite the 
precursor publication:

  David R. Thompson, Vijay Natraj, Robert O. Green, Mark C. Helmlinger, Bo-Cai Gao, and Michael L. Eastwood, (2018). “Optimal Estimation for Imaging Spectrometer Atmospheric Correction.” *Remote Sensing of Environment*, in press.


Installation Instructions
-------------------------

From Github
***********

The code repository, development branches, and user community are found on
`GitHub <https://github.com/davidraythompson/isofit>`_. To install:

1. Download or clone the git repo located at https://github.com/davidraythompson/isofit.

2. Install the ISOFIT dependencies using pip

.. code::

  python3 -m pip install scipy
  python3 -m pip install numba
  python3 -m pip install matplotlib
  python3 -m pip install scikit-learn
  python3 -m pip install spectral
  python3 -m pip install pytest 
  python3 -m pip install pep8 

3. Make sure the isofit/ and utils/ subdirectories are in your Python path like this:

.. code::

    export PYTHONPATH="${PYTHONPATH}:/path/to/isofit/isofit:/path/to/isofit/utils"

From PyPI
*********

Also, the latest release is always hosted on `PyPI <https://pypi.python.org/pypi/isofit>`_,
so if you have `pip` installed, you can install ISOFIT from the command line with

.. code::

    pip install isofit

This will install the "isofit" package into your environment as well as its dependencies.

Quick Start using MODTRAN 6.0
-----------------------------

This quick start presumes that you have an installation of the MODTRAN 6.0
radiative transfer model.  This is the preferred radiative transfer option if available, though we have also included an interface to the open source LibRadTran RT code.  Other open source options and neural network emulators will be integrated in the future. 

1. Configure your environment with the variables ISOFIT_BASE pointing to the base checkout directory of ISOFIT, and also MODTRAN_DIR pointing to the base MODTRAN 6.0 directory.

2. Run the following code

.. code::

    cd examples/20171108_Pasadena
    ./run_examples_modtran.sh

3. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist.  It will call MODTRAN to rebuild them, which will take a few minutes.

4. Look for output data in examples/20171108_Pasadena/output/.  Each retrieval writes diagnostic images to examples/20171108_Pasadena/images/ as it runs.

Quick Start with LibRadTran 2.0.x
---------------------------------

This quick start presumes that you have an installation of the open source libRadTran radiative transfer model (`LibRadTran <http://www.libradtran.org/doku.php>`)_ .  We have tested with the 2.0.2 release.  You will need the "REPTRAN" absorption parameterization - follow the instructions on the libradtran installation page to get that data.

1. Configure your environment with the variables ISOFIT_BASE pointing to the base checkout directory of ISOFIT, and also LIBRADTRAN_DIR pointing to the base libRadTran directory.

2. Run the following code

.. code::

    cd examples/20171108_Pasadena
    ./run_example_libradtran.sh

3. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist.  It will call libRadTran to rebuild them, which will take a few minutes.

4. Look for output data in examples/20171108_Pasadena/output/.  Diagnostic images are written to examples/20171108_Pasadena/images/.

Additional Installation Info for Mac OSX
------------------------------------------

1. Install the command-line compiler

.. code::

  xcode-select --install

2. Download the python3 installer from https://www.python.org/downloads/mac-osx/
