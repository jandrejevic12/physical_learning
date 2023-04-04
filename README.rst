Physical learning with elastic networks
=======================================

Repository for physical learning with elastic networks.

Documentation
-------------
The documentation for this repo is made using `Read the Docs <https://physical-learning.readthedocs.io/en/latest/index.html>`_.

Setup
-----
1. Clone a local copy of the repository:

.. code-block:: bash
   
   $ git clone https://github.com/jandrejevic12/physical_learning.git

2. The main directory contains an :code:`environment.yml` file for easily setting up a conda environment, named :code:`lrn`, with all the package dependencies:

.. code-block:: bash
   
   $ conda env create --file=environment.yml

3. Visualization of 3D networks makes use of the `vapory package <https://github.com/Zulko/vapory>`_, which relies on the `POV-Ray software <http://www.povray.org>`_. POV-Ray must be installed separately, for example via Homebrew on a Mac:

.. code-block:: bash
   
   $ brew install povray

4. Training networks at nonzero temperature makes use of the `LAMMPS software <https://www.lammps.org/#gsc.tab=0>`_. Please see the instructions in `lammps_ext <https://github.com/jandrejevic12/physical_learning/tree/main/lammps_ext>`_ for incorporating the extended LAMMPS code.

5. Several jupyter notebooks are provided for getting started generating and training elastic networks, and can be viewed on the `documentation page <https://physical-learning.readthedocs.io/en/latest/examples.html>`_.


